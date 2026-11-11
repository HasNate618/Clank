import base64
import email.message
import unittest
from unittest import mock

from clank import brain
from clank.brain import Ground, Plan
from clank.frames import Frame
from clank.intent import SKILLS


def fake_frame(index: int = 0) -> Frame:
    return Frame(index=index, t_s=float(index), jpeg=b"\xff\xd8\xff\xd9", source="test")


def not_none(value, message: str = ""):
    """Narrow an Optional for the type checker. `assertIsNotNone` does not narrow."""
    assert value is not None, message
    return value


class ChatPatchMixin:
    reply: str = ""
    reply_error: str = ""
    reply_prompt_ms: float = 0.0
    reply_decode_ms: float = 0.0
    _orig_chat = None

    def setUp(self) -> None:
        self.chat_calls: list[dict] = []
        self._orig_chat = brain._chat
        brain._chat = self._fake_chat

    def _fake_chat(self, config, messages, max_tokens=None):
        self.chat_calls.append({"messages": messages, "max_tokens": max_tokens})
        return brain.ModelReply(
            text=self.reply,
            prompt_ms=self.reply_prompt_ms,
            decode_ms=self.reply_decode_ms,
            error=self.reply_error,
        )

    def tearDown(self) -> None:
        brain._chat = self._orig_chat

    def user_text(self, call: int = -1) -> str:
        """The text sent with the still on call `call` (default: the last)."""
        content = self.chat_calls[call]["messages"][-1]["content"]
        if isinstance(content, str):
            return content
        return "\n".join(part["text"] for part in content if part["type"] == "text")

    def image_sent(self, call: int = -1) -> str:
        """The data URL of the still sent on call `call`."""
        content = self.chat_calls[call]["messages"][-1]["content"]
        image = next(part for part in content if part["type"] == "image_url")
        return image["image_url"]["url"]


class ScriptedChat(ChatPatchMixin):
    """`ChatPatchMixin` with one reply per call, so PLAN/GROUND/VERIFY differ."""

    script: list[str] = []

    def setUp(self) -> None:
        super().setUp()
        self.script = list(type(self).script)
        self.calls = 0

    def _fake_chat(self, config, messages, max_tokens=None):
        self.chat_calls.append({"messages": messages, "max_tokens": max_tokens})
        text = self.script[min(self.calls, len(self.script) - 1)] if self.script else ""
        self.calls += 1
        return brain.ModelReply(text=text)


class ExtractJson(unittest.TestCase):
    def test_extracts_a_fenced_object(self) -> None:
        text = 'sure\n```json\n{"skill":"reach","side":"right","target":"card"}\n```'
        parsed = brain.extract_json_object(text)
        assert parsed is not None
        self.assertEqual(parsed["skill"], "reach")


class PlanParser(ChatPatchMixin, unittest.TestCase):
    def test_action_is_accepted_as_skill(self) -> None:
        self.reply = '{"action":"reach","target":"card"}'
        result = brain.plan(
            brain.BrainConfig(),
            seed="",
            frame=fake_frame(),
            live="heard: hi",
        )
        self.assertEqual(result.skill, "reach")
        self.assertIn("side missing", result.note)

    def test_skill_as_a_key_is_folded(self) -> None:
        self.reply = '{"reach":"right","target":"card"}'
        result = brain.plan(
            brain.BrainConfig(),
            seed="",
            frame=fake_frame(),
            live="heard: hi",
        )
        self.assertEqual(result.skill, "reach")
        self.assertEqual(result.side, "right")

    def test_insert_is_rejected(self) -> None:
        self.reply = '{"skill":"insert","side":"right","target":"slot"}'
        result = brain.plan(
            brain.BrainConfig(),
            seed="",
            frame=fake_frame(),
            live="heard: hi",
        )
        self.assertFalse(result.ok)


class PlannerSkills(unittest.TestCase):
    def test_motor_tools_are_a_subset_of_the_intent_schema(self) -> None:
        # `menu` is in both: it is a motor tool the model may choose and a legal Intent, because
        # the controller's system button is a real thing to press - it just cannot be reached by
        # a finger, so the adapter sends it as a request instead of a motion.
        self.assertEqual(set(brain.MOTOR_SKILLS),
                         {"hold", "look", "reach", "grab", "release", "menu"})
        self.assertLessEqual(set(brain.MOTOR_SKILLS), set(SKILLS))

    def test_sense_tools_are_not_intent_skills(self) -> None:
        # The whole point of `locate` being planner-only: the adapter cannot compile it, so an
        # Intent that names it must stay impossible even though the model may choose it.
        self.assertEqual(set(brain.SENSE_SKILLS), {"locate"})
        self.assertTrue(set(brain.SENSE_SKILLS).isdisjoint(SKILLS))
        self.assertEqual(set(brain.PLANNER_SKILLS),
                         set(brain.MOTOR_SKILLS) | set(brain.SENSE_SKILLS))

    def test_the_constitution_offers_the_sense_tool_as_find(self) -> None:
        # MEASURED 2026-11-10: offered as `locate` the model chose it 0 of 3 stills; offered as
        # `find` it chose it on the clearest frame. The canonical skill stays `locate`, so the
        # word the model uses and the word the code uses are joined by SENSE_ALIASES.
        self.assertIn('"find"', brain.CONSTITUTION)
        self.assertNotIn('"locate"', brain.CONSTITUTION)
        self.assertEqual(brain.SENSE_ALIASES["find"], "locate")

    def test_find_is_read_as_the_canonical_locate(self) -> None:
        class FindReply(ChatPatchMixin, unittest.TestCase):
            reply = '{"skill":"find","target":"card"}'

        case = FindReply("run")
        case.setUp()
        try:
            result = brain.plan(config=brain.BrainConfig(), seed="", frame=fake_frame(),
                                live="heard: take my card")
        finally:
            case.tearDown()
        self.assertTrue(result.ok)
        self.assertEqual(result.skill, "locate")
        self.assertEqual(result.side, "none")
        self.assertIn("find", result.note)

    def test_the_planner_cannot_emit_rest(self) -> None:
        # `rest` exists on the intent schema so a writer can ask for the clip
        # return, but a brain that says "hold" means freeze. Nothing in the
        # constitution offers `rest`, and PLAN rejects it as an unknown skill.
        self.assertIn("rest", SKILLS)
        self.assertNotIn("rest", brain.PLANNER_SKILLS)
        self.assertNotIn("rest", brain.CONSTITUTION)

    def test_locate_is_accepted_with_a_target_and_no_side(self) -> None:
        class LocateReply(ChatPatchMixin, unittest.TestCase):
            reply = '{"skill":"locate","side":"none","target":"card"}'

        case = LocateReply("run")
        case.setUp()
        try:
            result = brain.plan(
                config=brain.BrainConfig(), seed="", frame=fake_frame(),
                live="heard: take my card",
            )
        finally:
            case.tearDown()
        self.assertTrue(result.ok)
        self.assertEqual(result.skill, "locate")
        self.assertEqual(result.target, "card")
        self.assertEqual(result.side, "none")

    def test_locate_without_a_target_is_rejected(self) -> None:
        class NoTarget(ChatPatchMixin, unittest.TestCase):
            reply = '{"skill":"locate","side":"none","target":null}'

        case = NoTarget("run")
        case.setUp()
        try:
            result = brain.plan(
                config=brain.BrainConfig(), seed="", frame=fake_frame(), live="heard: hi",
            )
        finally:
            case.tearDown()
        self.assertFalse(result.ok)
        self.assertIn("needs a target", result.error)

    def test_locate_with_a_side_is_coerced_to_none(self) -> None:
        # A sense tool has no hand. The model adds a side anyway, and a right decision is not
        # thrown away over a field the tool does not use.
        class WithSide(ChatPatchMixin, unittest.TestCase):
            reply = '{"skill":"locate","side":"right","target":"card"}'

        case = WithSide("run")
        case.setUp()
        try:
            result = brain.plan(
                config=brain.BrainConfig(), seed="", frame=fake_frame(), live="heard: hi",
            )
        finally:
            case.tearDown()
        self.assertTrue(result.ok, result.error)
        self.assertEqual(result.side, "none")
        self.assertIn("side ignored", result.note)

    def test_rest_is_rejected_by_the_planner_parser(self) -> None:
        class RestReply(ChatPatchMixin, unittest.TestCase):
            reply = '{"skill":"rest","side":"none","target":null}'

        case = RestReply("run")
        case.setUp()
        try:
            result = brain.plan(
                config=brain.BrainConfig(),
                seed="",
                frame=fake_frame(),
                live="heard: hi",
            )
        finally:
            case.tearDown()
        self.assertFalse(result.ok)
        self.assertIn("rest", result.error)


class MenuSkill(unittest.TestCase):
    """JobBot asks for the menu button early. Nothing on the body can press it."""

    def test_the_menu_action_is_offered_only_when_the_voice_asks_for_it(self) -> None:
        # MEASURED 2026-11-10: with the menu line always in the list, "take my card" came back
        # as menu(card). One extra action captures an unrelated instruction on this model, so the
        # action exists exactly when the instruction names it.
        self.assertNotIn('"menu"', brain.CONSTITUTION)
        asked = brain.constitution_for("press the menu button")
        self.assertIn('"menu"', asked)
        self.assertIn("press the controller menu button", asked)
        self.assertNotIn('"menu"', brain.constitution_for("Hey, take my card."))

    def test_a_menu_instruction_reaches_the_model_through_think(self) -> None:
        class PressReply(ChatPatchMixin, unittest.TestCase):
            reply = '{"skill":"menu","side":"right","target":"menu button"}'

        case = PressReply("run")
        case.setUp()
        try:
            thought = brain.think(
                brain.BrainConfig(), brain.DEFAULT_SEED, fake_frame(),
                "press the menu button", [], brain.WorkingMemory())
            system = case.chat_calls[0]["messages"][0]["content"]
        finally:
            case.tearDown()
        self.assertEqual(thought.plan.skill, "menu")
        self.assertEqual(thought.decision.intent.skill, "menu")
        self.assertIn('"menu"', system)

    def test_a_menu_plan_assembles_into_a_legal_intent(self) -> None:
        from clank.intent import validate_intent

        decision = brain.assemble(Plan(skill="menu", side="right"), None, seq=0, t_mono_ns=0)
        self.assertTrue(decision.accepted, decision.reason)
        self.assertEqual(decision.intent.skill, "menu")
        self.assertEqual(decision.intent.side, "right")
        self.assertEqual(validate_intent(decision.intent), "")


class WorkingMemoryTests(unittest.TestCase):
    def test_card_finds_jobbot_card(self) -> None:
        mem = brain.WorkingMemory()
        mem.remember("JobBot card", brain.Sighting(
            noun="JobBot card", present=True, pixel=(0.8, 0.4)))
        found = mem.find("card")
        self.assertIsNotNone(found)
        self.assertIsNotNone(found)
        assert found is not None
        self.assertEqual(found.pixel, (0.8, 0.4))

    def test_missing_noun_is_none(self) -> None:
        self.assertIsNone(brain.WorkingMemory().find("card"))

    def test_a_near_miss_is_not_a_match(self) -> None:
        mem = brain.WorkingMemory()
        mem.remember("cartridge", brain.Sighting(noun="cartridge", present=True))
        self.assertIsNone(mem.find("card"))


class AssembleLocate(unittest.TestCase):
    def test_locate_does_not_emit_a_locate_intent(self) -> None:
        d = brain.assemble(Plan(skill="locate", side="none", target="card"),
                           None, seq=0, t_mono_ns=0)
        self.assertEqual(d.intent.skill, "hold")
        self.assertEqual(d.note, "locate")
        self.assertTrue(d.accepted)

    def test_reach_without_a_sighting_is_not_located(self) -> None:
        d = brain.assemble(Plan(skill="reach", side="right", target="card"),
                           None, seq=0, t_mono_ns=0, memory=brain.WorkingMemory())
        self.assertFalse(d.accepted)
        self.assertIn("not_located", d.reason)
        self.assertEqual(d.intent.skill, "hold")

    def test_reach_after_an_absent_sighting_is_not_located(self) -> None:
        mem = brain.WorkingMemory()
        mem.remember("card", brain.Sighting(
            noun="card", present=False, outcome="not_visible"))
        d = brain.assemble(Plan(skill="reach", side="right", target="card"),
                           None, seq=0, t_mono_ns=0, memory=mem)
        self.assertFalse(d.accepted)
        self.assertIn("not_visible", d.reason)

    def test_reach_with_a_sighting_fills_the_pixel(self) -> None:
        mem = brain.WorkingMemory()
        mem.remember("card", brain.Sighting(
            noun="card", present=True, pixel=(0.81, 0.45)))
        d = brain.assemble(Plan(skill="reach", side="right", target="card"),
                           None, seq=0, t_mono_ns=0, memory=mem)
        self.assertTrue(d.accepted)
        self.assertEqual(d.intent.skill, "reach")
        self.assertEqual(d.intent.pixel, (0.81, 0.45))

    def test_grab_with_a_sighting_keeps_the_target(self) -> None:
        mem = brain.WorkingMemory()
        mem.remember("card", brain.Sighting(
            noun="card", present=True, pixel=(0.81, 0.45)))
        d = brain.assemble(Plan(skill="grab", side="right", target="card"),
                           None, seq=0, t_mono_ns=0, memory=mem)
        self.assertTrue(d.accepted, d.reason)
        self.assertEqual(d.intent.skill, "grab")

    def test_live_block_does_not_carry_coordinates(self) -> None:
        text = brain.live_block(
            "take my card", "empty", "empty", 0.0, "unknown",
            "locate none target=card → present")
        self.assertNotIn("0.8", text)
        self.assertNotIn("px", text.lower())
        self.assertNotIn("TOP-LEFT", text)


class ThinkTests(ChatPatchMixin, unittest.TestCase):
    """Think is the chooser: PLAN plus ASSEMBLE, and nothing else. It never locates."""

    def run_think(self, reply: str, heard: str, memory=None):
        self.reply = reply
        return brain.think(
            brain.BrainConfig(), "", fake_frame(), heard, [],
            memory if memory is not None else brain.WorkingMemory())

    def test_take_my_card_can_choose_locate(self) -> None:
        thought = self.run_think(
            '{"skill":"locate","side":"none","target":"card"}',
            "Hey, Hillman, take my card.")
        self.assertEqual(thought.sense, "locate")
        self.assertEqual(thought.plan.target, "card")
        self.assertEqual(thought.decision.intent.skill, "hold")
        self.assertEqual(thought.decision.note, "locate")

    def test_reach_without_memory_is_not_located(self) -> None:
        thought = self.run_think(
            '{"skill":"reach","side":"right","target":"card"}', "take my card")
        self.assertIsNone(thought.sense)
        self.assertIn("not_located", thought.decision.reason)

    def test_reach_after_remembering_emits_reach(self) -> None:
        mem = brain.WorkingMemory()
        mem.remember("card", brain.Sighting(
            noun="card", present=True, pixel=(0.8, 0.4)))
        thought = self.run_think(
            '{"skill":"reach","side":"right","target":"card"}', "take my card", mem)
        self.assertIsNone(thought.sense)
        self.assertEqual(thought.decision.intent.skill, "reach")
        self.assertEqual(thought.decision.intent.pixel, (0.8, 0.4))

    def test_think_does_not_send_audio(self) -> None:
        self.run_think('{"skill":"hold","side":"none","target":null}', "take my card")
        content = self.chat_calls[0]["messages"][-1]["content"]
        self.assertNotIn("input_audio", [part["type"] for part in content])

    def test_the_prompt_is_the_constitution_not_a_skill_enum(self) -> None:
        self.run_think('{"skill":"hold","side":"none","target":null}', "take my card")
        system = self.chat_calls[0]["messages"][0]["content"]
        self.assertEqual(system, brain.CONSTITUTION)
        self.assertNotIn("grab|place|reach|release", system)

    def test_the_last_line_reaches_the_model(self) -> None:
        self.reply = '{"skill":"hold","side":"none","target":null}'
        brain.think(brain.BrainConfig(), "", fake_frame(), "take my card", [],
                    brain.WorkingMemory(),
                    last="locate none target=card \u2192 not_visible")
        self.assertIn("not_visible", self.user_text())


class PromptText(unittest.TestCase):
    def test_the_seed_carries_no_choreography(self) -> None:
        # MEASURED 2026-11-10: a seed naming the cartridge, the slot and the lever overrode what
        # JobBot said - asked for his card, the model answered `release ... cartridge` on three
        # stills. The seed now names the world and no errand, and this test is what keeps a later
        # hand from quietly putting the walkthrough back.
        # Naming the world's things is fine and is what makes the model name the right object.
        # Naming an errand is not: the seed may not say what to do with them.
        s = brain.DEFAULT_SEED.lower()
        for banned in ("slot", "lever", "ghost", "glow", "store clerk", "goes in",
                       "put it", "insert", "pull the"):
            self.assertNotIn(banned, s)
        self.assertIn("job simulator", s)

    def test_seed_and_constitution_do_not_list_positions(self) -> None:
        blob = brain.CONSTITUTION + "\n" + brain.DEFAULT_SEED
        for word in ("TOP-LEFT", "TOP-RIGHT", "BOTTOM-LEFT", "BOTTOM-RIGHT"):
            self.assertNotIn(word, blob)

    def test_constitution_schema_has_no_pixel_key(self) -> None:
        self.assertIn('"skill"', brain.CONSTITUTION)
        self.assertNotIn("pixel", brain.CONSTITUTION.lower())
        for banned in ("fire", "strike", "dodge", "twitch", "insert", "pull"):
            self.assertNotIn(banned, brain.CONSTITUTION.lower())

    def test_live_block_carries_glow_and_ghost(self) -> None:
        text = brain.live_block("take my card", "empty", "holding", 0.0, "yes", "reach right → done", ghost="no")
        self.assertIn("glow=yes", text)
        self.assertIn("ghost=no", text)
        self.assertIn("right=holding", text)
        self.assertNotIn("current goal", text.lower())


class Occupancy(unittest.TestCase):
    def test_release_on_an_empty_hand_is_held(self) -> None:
        d = brain.assemble(
            Plan(skill="release", side="right"),
            None,
            seq=0,
            t_mono_ns=0,
            right="empty",
        )
        self.assertFalse(d.accepted)
        self.assertEqual(d.intent.skill, "hold")

    def test_grab_on_a_holding_hand_is_held(self) -> None:
        d = brain.assemble(
            Plan(skill="grab", side="right", target="card"),
            None,
            seq=0,
            t_mono_ns=0,
            right="holding",
        )
        self.assertFalse(d.accepted)

    def test_not_visible_reach_becomes_a_look_scan(self) -> None:
        d = brain.assemble(
            Plan(skill="reach", side="right", target="card"),
            Ground(noun="card", error="not visible"),
            seq=0,
            t_mono_ns=0,
        )
        self.assertEqual(d.intent.skill, "look")
        self.assertEqual(d.note, "downgraded to scan")
        # A refused plan still returns the intent to send, and that fallback must
        # be a legal scan rather than a look with no direction.
        self.assertEqual(d.intent.side, "none")
        self.assertIsNotNone(d.intent.yaw_rad)
        self.assertNotEqual(d.intent.yaw_rad, 0.0)


class CueGates(unittest.TestCase):
    def test_grab_with_glow_no_is_held(self) -> None:
        d = brain.assemble(
            Plan(skill="grab", side="right", target="card"),
            Ground(noun="card", pixel=(0.5, 0.5)),
            seq=0,
            t_mono_ns=0,
            glow="no",
        )
        self.assertFalse(d.accepted)
        self.assertEqual(d.intent.skill, "hold")
        self.assertIn("glow is no", d.reason)

    def test_unknown_glow_does_not_refuse_grab(self) -> None:
        d = brain.assemble(
            Plan(skill="grab", side="right", target="card"),
            Ground(noun="card", pixel=(0.5, 0.5)),
            seq=0,
            t_mono_ns=0,
            glow="unknown",
        )
        self.assertTrue(d.accepted)
        self.assertEqual(d.intent.skill, "grab")

    def test_glow_yes_does_not_emit_grab_without_a_plan(self) -> None:
        d = brain.assemble(
            Plan(skill="hold", side="none"),
            None,
            seq=0,
            t_mono_ns=0,
            glow="yes",
        )
        self.assertEqual(d.intent.skill, "hold")

    def test_release_into_slot_without_ghost_is_held(self) -> None:
        d = brain.assemble(
            Plan(skill="release", side="right", target="slot"),
            None,
            seq=0,
            t_mono_ns=0,
            right="holding",
            ghost="no",
        )
        self.assertFalse(d.accepted)
        self.assertIn("ghost is no", d.reason)

    def test_release_into_slot_with_unknown_ghost_is_accepted(self) -> None:
        # Pins that `unknown` never refuses. Without this, an edit to
        # `if ghost != "yes"` passes the rest of this class and freezes the
        # replay harness, which carries no game cues at all.
        d = brain.assemble(
            Plan(skill="release", side="right", target="slot"),
            None,
            seq=0,
            t_mono_ns=0,
            right="holding",
            ghost="unknown",
        )
        self.assertTrue(d.accepted)
        self.assertEqual(d.intent.skill, "release")

    def test_release_of_a_card_does_not_need_ghost(self) -> None:
        d = brain.assemble(
            Plan(skill="release", side="right", target="card"),
            None,
            seq=0,
            t_mono_ns=0,
            right="holding",
            ghost="no",
        )
        self.assertTrue(d.accepted)
        self.assertEqual(d.intent.skill, "release")


class CueGatesInTurn(ChatPatchMixin, unittest.TestCase):
    def test_run_turn_passes_glow_to_assemble(self) -> None:
        self.reply = '{"skill":"grab","side":"right","target":"card"}'
        record = brain.run_turn(
            brain.BrainConfig(),
            seed="",
            frame=fake_frame(),
            heard="take my card",
            history=[],
            glow="no",
        )
        self.assertEqual(record.decision.intent.skill, "hold")
        self.assertIn("glow is no", record.decision.reason)

    def test_run_turn_passes_hand_and_ghost_cues_to_assemble(self) -> None:
        self.reply = '{"skill":"release","side":"right","target":"slot"}'
        record = brain.run_turn(
            brain.BrainConfig(),
            seed="",
            frame=fake_frame(),
            heard="put it in the slot",
            history=[],
            right="holding",
            ghost="no",
        )
        self.assertEqual(record.decision.intent.skill, "hold")
        self.assertIn("ghost is no", record.decision.reason)

    def test_run_turn_passes_right_so_a_side_less_plan_defaults_to_the_free_hand(
            self) -> None:
        # `right` is the hand cue, not just text: it chooses `default_side` for a
        # plan that omits its side, and a plan that names the right hand has to
        # see that hand holding. Both die if `right` stops reaching the turn.
        with mock.patch.object(brain, "_chat", side_effect=[
            brain.ModelReply(text='{"skill":"release","target":"card"}'),
            brain.ModelReply(
                text='{"skill":"release","side":"right","target":"card"}'),
        ]):
            side_less = brain.run_turn(
                brain.BrainConfig(), seed="", frame=fake_frame(),
                heard="put the card down", history=[],
                left="holding", right="holding",
            )
            named_right = brain.run_turn(
                brain.BrainConfig(), seed="", frame=fake_frame(),
                heard="put the card down", history=[], right="holding",
            )
        self.assertEqual(side_less.plan.side, "left")
        self.assertEqual(side_less.decision.intent.side, "left")
        self.assertIn("defaulted to left", side_less.plan.note)
        self.assertTrue(named_right.decision.accepted)
        self.assertEqual(named_right.decision.intent.side, "right")


class RunTurnCrop(ScriptedChat, unittest.TestCase):
    """`run_turn` crops the container box, grounds the crop, then verifies it."""

    #: The hand-drawn tray box from the 2026-11-01 measurement.
    box = (0.270, 0.325, 0.548, 0.566)
    reach = '{"skill":"reach","side":"right","target":"STORE CLERK"}'
    grab = '{"skill":"grab","side":"right","target":"STORE CLERK"}'

    def crop(self, jpeg: bytes, box: tuple[float, float, float, float],
             width: int = 1024) -> bytes:
        """Stand-in for `frames.crop_jpeg`: ffmpeg is not required in a unit test."""
        return jpeg + b"cropped"

    def run_turn(self, script: list[str], **kwargs):
        self.script = script
        with mock.patch.object(brain, "crop_jpeg", side_effect=self.crop) as crop:
            self.crop_mock = crop
            return brain.run_turn(
                brain.BrainConfig(),
                seed="",
                frame=fake_frame(),
                heard="grab the store clerk",
                history=[],
                **kwargs,
            )

    def ground_of(self, record):
        """The ground of a completed turn, narrowed.

        `TurnRecord.ground` is optional - a turn that never grounded has none - so every
        assertion here would otherwise be a possible-None access.
        """
        self.assertIsNotNone(record.ground)
        assert record.ground is not None
        return record.ground

    def cropped_url(self) -> str:
        jpeg = fake_frame().jpeg + b"cropped"
        return "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")

    def test_ground_is_asked_about_the_crop_and_composes_into_the_box(self) -> None:
        record = self.run_turn([self.reach, "TOP-RIGHT"], box=self.box)
        self.crop_mock.assert_called_once_with(fake_frame().jpeg, self.box)
        self.assertEqual(self.image_sent(1), self.cropped_url())
        self.assertEqual(self.ground_of(record).box, self.box)
        self.assertAlmostEqual(not_none(self.ground_of(record).pixel)[0], 0.270 + 0.278 * 0.75)
        self.assertAlmostEqual(not_none(self.ground_of(record).pixel)[1], 0.325 + 0.241 * 0.25)
        self.assertEqual(record.self_state["box"], list(self.box))
        self.assertEqual(record.self_state["crop_error"], "")

    def test_without_a_box_the_whole_frame_goes_out(self) -> None:
        record = self.run_turn([self.reach, "TOP-RIGHT"])
        self.crop_mock.assert_not_called()
        self.assertEqual(self.image_sent(1), fake_frame().data_url())
        self.assertEqual(self.ground_of(record).box, (0.0, 0.0, 1.0, 1.0))
        self.assertIsNone(record.self_state["box"])

    def test_a_failed_crop_grounds_the_whole_frame_instead_of_raising(self) -> None:
        self.script = [self.reach, "TOP-LEFT"]
        with mock.patch.object(brain, "crop_jpeg",
                               side_effect=RuntimeError("ffmpeg: not found")):
            record = brain.run_turn(
                brain.BrainConfig(), seed="", frame=fake_frame(),
                heard="grab the store clerk", history=[], box=self.box,
            )
        self.assertEqual(self.image_sent(1), fake_frame().data_url())
        self.assertEqual(self.ground_of(record).box, (0.0, 0.0, 1.0, 1.0))
        self.assertEqual(self.ground_of(record).pixel, (0.25, 0.25))
        self.assertEqual(record.decision.intent.skill, "reach")
        self.assertIn("not found", record.self_state["crop_error"])

    def test_the_container_reaches_the_ground_question(self) -> None:
        self.run_turn([self.reach, "TOP-LEFT"], container="register")
        self.assertIn("Which position in this register", self.user_text(1))

    def test_self_state_reports_the_cues_it_was_given(self) -> None:
        # Pin: glow is reported, never recomputed inside the turn.
        record = self.run_turn(['{"skill":"hold","side":"none"}'],
                               glow="yes", ghost="yes")
        self.assertEqual(record.self_state["glow"], "yes")
        self.assertEqual(record.self_state["ghost"], "yes")
        self.assertIn("glow=yes", record.live)
        self.assertIn("ghost=yes", record.live)

    def test_verify_mismatch_blocks_the_grab(self) -> None:
        record = self.run_turn([self.grab, "TOP-RIGHT", "GOURMET CHEF"],
                               glow="yes", box=self.box)
        self.assertEqual(len(self.chat_calls), 3)
        self.assertIn("verify mismatch", self.ground_of(record).error)
        self.assertIsNone(self.ground_of(record).pixel)
        # The real ground survives the mismatch - only its pixel goes. Without
        # these, a fabricated `heard`-based ground passes this test again.
        self.assertEqual(self.ground_of(record).quadrant, "TOP-RIGHT")
        self.assertEqual(self.ground_of(record).raw, "TOP-RIGHT")
        self.assertEqual(self.ground_of(record).box, self.box)
        self.assertFalse(record.to_dict()["verify"]["match"])
        self.assertIsNotNone(record.verify)
        self.assertFalse(not_none(record.verify).match)
        self.assertEqual(record.decision.intent.skill, "hold")
        self.assertIn("not visible", record.decision.reason)

    def test_a_transport_error_during_verify_is_not_a_content_mismatch(self) -> None:
        # A dead model server must not look like a wrong cartridge: the suffix
        # carries the transport message and the Verify keeps it too.
        with mock.patch.object(brain, "_chat", side_effect=[
            brain.ModelReply(text=self.grab),
            brain.ModelReply(text="TOP-RIGHT"),
            brain.ModelReply(text="", error="transport: timed out"),
        ]):
            record = self.run_turn([self.grab], glow="yes", box=self.box)
        self.assertIn("verify mismatch: ", self.ground_of(record).error)
        self.assertIn("transport: timed out", self.ground_of(record).error)
        self.assertTrue(not_none(record.verify).error)
        # The quadrant still came from the real ground, not a fabricated one.
        self.assertEqual(self.ground_of(record).quadrant, "TOP-RIGHT")
        self.assertEqual(self.ground_of(record).box, self.box)

    def test_a_matching_verify_is_recorded_on_the_turn(self) -> None:
        record = self.run_turn([self.grab, "TOP-RIGHT", "the store clerk cartridge"],
                               glow="yes", box=self.box)
        self.assertIsNotNone(record.verify)
        self.assertTrue(not_none(record.verify).match)
        self.assertTrue(record.to_dict()["verify"]["match"])

    def test_a_reach_at_glow_yes_is_never_verified(self) -> None:
        # The gate is grab-only. Widening it would pay a VERIFY call for a
        # reach, and a mismatch would overwrite the ground and downgrade it.
        record = self.run_turn([self.reach, "TOP-RIGHT"], glow="yes", box=self.box)
        self.assertEqual(len(self.chat_calls), 2)
        self.assertEqual(record.decision.intent.skill, "reach")
        self.assertIsNone(record.verify)

    def test_a_matching_verify_lets_the_grab_through(self) -> None:
        record = self.run_turn([self.grab, "TOP-RIGHT", "the store clerk cartridge"],
                               glow="yes", box=self.box)
        self.assertEqual(record.decision.intent.skill, "grab")
        self.assertIsNotNone(record.decision.intent.pixel)

    def test_verify_sees_the_same_crop_as_ground(self) -> None:
        self.run_turn([self.grab, "TOP-RIGHT", "GOURMET CHEF"],
                      glow="yes", box=self.box)
        self.assertEqual(self.image_sent(1), self.cropped_url())
        self.assertEqual(self.image_sent(2), self.cropped_url())

    def test_no_verify_without_glow_yes(self) -> None:
        record = self.run_turn([self.grab, "TOP-RIGHT"], glow="unknown", box=self.box)
        self.assertEqual(len(self.chat_calls), 2)
        self.assertEqual(record.decision.intent.skill, "grab")


class ComposePixel(unittest.TestCase):
    #: The hand-drawn tray box from the 2026-11-01 measurement.
    box = (0.270, 0.325, 0.548, 0.566)

    def test_top_right_of_the_measured_tray(self) -> None:
        u, v = brain.compose_pixel(self.box, "TOP-RIGHT")
        self.assertAlmostEqual(u, 0.270 + (0.548 - 0.270) * 0.75)
        self.assertAlmostEqual(v, 0.325 + (0.566 - 0.325) * 0.25)

    def test_bottom_left_takes_the_other_centre_of_each_axis(self) -> None:
        u, v = brain.compose_pixel(self.box, "BOTTOM-LEFT")
        self.assertAlmostEqual(u, 0.270 + (0.548 - 0.270) * 0.25)
        self.assertAlmostEqual(v, 0.325 + (0.566 - 0.325) * 0.75)

    def test_the_whole_frame_is_the_plain_image_quadrant(self) -> None:
        u, v = brain.compose_pixel((0.0, 0.0, 1.0, 1.0), "TOP-LEFT")
        self.assertEqual((u, v), (0.25, 0.25))


class GroundQuad(ChatPatchMixin, unittest.TestCase):
    box = (0.270, 0.325, 0.548, 0.566)

    def test_store_clerk_top_right_composes_into_the_box(self) -> None:
        self.reply = "TOP-RIGHT"
        g = brain.ground(brain.BrainConfig(), fake_frame(), "STORE CLERK", box=self.box)
        self.assertEqual(g.quadrant, "TOP-RIGHT")
        self.assertEqual(g.methods, "quad")
        self.assertIsNotNone(g.pixel)
        self.assertAlmostEqual(not_none(g.pixel)[0], 0.270 + (0.548 - 0.270) * 0.75)
        self.assertAlmostEqual(not_none(g.pixel)[1], 0.325 + (0.566 - 0.325) * 0.25)

    def test_one_call_asking_the_bare_question(self) -> None:
        self.reply = "TOP-LEFT"
        brain.ground(brain.BrainConfig(), fake_frame(), "STORE CLERK")
        self.assertEqual(len(self.chat_calls), 1)
        self.assertEqual(self.chat_calls[0]["max_tokens"], 60)
        self.assertEqual(
            self.user_text(),
            "Which position in this tray does the STORE CLERK occupy? "
            "Answer with one of: TOP-LEFT, TOP-RIGHT, BOTTOM-LEFT, BOTTOM-RIGHT.",
        )

    def test_container_is_interpolated(self) -> None:
        self.reply = "TOP-LEFT"
        brain.ground(brain.BrainConfig(), fake_frame(), "lever", container="register")
        self.assertIn("Which position in this register", self.user_text())

    def test_prompt_names_only_the_asked_noun_and_the_answer_words(self) -> None:
        self.reply = "TOP-LEFT"
        brain.ground(brain.BrainConfig(), fake_frame(), "STORE CLERK")
        blob = self.user_text()
        self.assertIn("STORE CLERK", blob)
        self.assertIn("TOP-LEFT", blob)  # answer vocabulary, not a layout table
        for unasked in (
            "AUTO MECHANIC",
            "GOURMET CHEF",
            "OFFICE WORKER",
            "BLUE",
            "GREEN",
            "RED",
        ):
            self.assertNotIn(unasked, blob)

    def test_top_right_with_a_space_still_parses(self) -> None:
        self.reply = "The STORE CLERK cartridge is in the top right position."
        g = brain.ground(brain.BrainConfig(), fake_frame(), "STORE CLERK")
        self.assertEqual(g.quadrant, "TOP-RIGHT")
        self.assertEqual(g.raw, self.reply)

    def test_none_is_not_visible(self) -> None:
        self.reply = "NONE"
        g = brain.ground(brain.BrainConfig(), fake_frame(), "card")
        self.assertEqual(g.error, "not visible")
        self.assertIsNone(g.pixel)
        self.assertIsNone(g.quadrant)

    def test_unparseable_is_not_visible(self) -> None:
        self.reply = "It is somewhere near the middle of the tray."
        g = brain.ground(brain.BrainConfig(), fake_frame(), "card")
        self.assertEqual(g.error, "not visible")
        self.assertIsNone(g.pixel)
        self.assertEqual(g.raw, self.reply)

    def test_transport_error_is_not_read_as_not_visible(self) -> None:
        self.reply_error = "transport: timed out"
        g = brain.ground(brain.BrainConfig(), fake_frame(), "card")
        self.assertEqual(g.error, "transport: timed out")
        self.assertIsNone(g.pixel)

    def test_cell_aliases_the_quadrant_and_to_dict_reports_the_box(self) -> None:
        self.reply = "BOTTOM-LEFT"
        g = brain.ground(brain.BrainConfig(), fake_frame(), "card", box=self.box)
        self.assertEqual(g.cell, "BOTTOM-LEFT")
        as_dict = g.to_dict()
        self.assertEqual(as_dict["quadrant"], "BOTTOM-LEFT")
        self.assertEqual(as_dict["cell"], "BOTTOM-LEFT")
        self.assertEqual(as_dict["box"], list(self.box))
        self.assertEqual(as_dict["pixel"], list(not_none(g.pixel)))
        self.assertEqual(as_dict["error"], "")


class VerifyCrop(ChatPatchMixin, unittest.TestCase):
    def test_store_clerk_matches_a_paraphrase(self) -> None:
        self.reply = "the store clerk cartridge"
        v = brain.verify(brain.BrainConfig(), fake_frame(), "STORE CLERK")
        self.assertTrue(v.match)
        self.assertEqual(v.heard, self.reply)
        self.assertEqual(v.raw, self.reply)
        self.assertEqual(v.error, "")

    def test_a_neighbour_does_not_match(self) -> None:
        self.reply = "GOURMET CHEF"
        v = brain.verify(brain.BrainConfig(), fake_frame(), "STORE CLERK")
        self.assertFalse(v.match)
        self.assertEqual(v.heard, "GOURMET CHEF")

    def test_one_word_missing_is_a_mismatch(self) -> None:
        self.reply = "the store cartridge"
        v = brain.verify(brain.BrainConfig(), fake_frame(), "STORE CLERK")
        self.assertFalse(v.match)

    def test_punctuation_and_case_do_not_decide_the_match(self) -> None:
        self.reply = "It's the Store-Clerk cartridge."
        v = brain.verify(brain.BrainConfig(), fake_frame(), "store clerk")
        self.assertTrue(v.match)

    def test_a_negated_reply_does_not_match(self) -> None:
        self.reply = "the store clerk is not visible"
        v = brain.verify(brain.BrainConfig(), fake_frame(), "STORE CLERK")
        self.assertFalse(v.match)

    def test_no_in_the_reply_does_not_match(self) -> None:
        self.reply = "no card here"
        v = brain.verify(brain.BrainConfig(), fake_frame(), "card")
        self.assertFalse(v.match)

    def test_none_does_not_match(self) -> None:
        self.reply = "NONE"
        v = brain.verify(brain.BrainConfig(), fake_frame(), "card")
        self.assertFalse(v.match)

    def test_empty_reply_does_not_match(self) -> None:
        self.reply = "   "
        v = brain.verify(brain.BrainConfig(), fake_frame(), "card")
        self.assertFalse(v.match)
        self.assertEqual(v.heard, "")
        self.assertEqual(v.raw, "")

    def test_a_noun_with_no_words_cannot_match(self) -> None:
        self.reply = "a card"
        v = brain.verify(brain.BrainConfig(), fake_frame(), "")
        self.assertFalse(v.match)

    def test_transport_error_does_not_match_and_keeps_the_message(self) -> None:
        self.reply_error = "transport: timed out"
        v = brain.verify(brain.BrainConfig(), fake_frame(), "card")
        self.assertFalse(v.match)
        self.assertEqual(v.error, "transport: timed out")

    def test_timings_are_reported_from_the_reply(self) -> None:
        self.reply = "card"
        self.reply_prompt_ms = 12.0
        self.reply_decode_ms = 34.56
        d = brain.verify(brain.BrainConfig(), fake_frame(), "card").to_dict()
        self.assertEqual(d["prompt_ms"], 12.0)
        self.assertEqual(d["decode_ms"], 34.6)

    def test_a_transport_error_keeps_the_timings(self) -> None:
        self.reply_error = "transport: timed out"
        self.reply_decode_ms = 5.0
        v = brain.verify(brain.BrainConfig(), fake_frame(), "card")
        self.assertEqual(v.to_dict()["decode_ms"], 5.0)

    def test_one_call_asking_the_bare_question_about_the_crop(self) -> None:
        self.reply = "card"
        brain.verify(brain.BrainConfig(), fake_frame(), "card")
        self.assertEqual(len(self.chat_calls), 1)
        self.assertEqual(self.chat_calls[0]["max_tokens"], 40)
        self.assertEqual(
            self.user_text(),
            "This is a close-up of one object. What is it? Reply with the name, or NONE.",
        )
        content = self.chat_calls[-1]["messages"][-1]["content"]
        self.assertEqual(content[0]["type"], "image_url")

    def test_to_dict_reports_the_match(self) -> None:
        self.reply = "the store clerk cartridge"
        v = brain.verify(brain.BrainConfig(), fake_frame(), "STORE CLERK")
        self.assertEqual(
            v.to_dict(),
            {
                "noun": "STORE CLERK",
                "heard": "the store clerk cartridge",
                "match": True,
                "raw": "the store clerk cartridge",
                "error": "",
                "prompt_ms": 0.0,
                "decode_ms": 0.0,
            },
        )


class NamesTheNoun(unittest.TestCase):
    def test_card_is_not_found_inside_discard(self) -> None:
        self.assertFalse(brain._names_the_noun("card", "discard"))

    def test_art_is_not_found_inside_cartridge(self) -> None:
        self.assertFalse(brain._names_the_noun("art", "cartridge"))


if __name__ == "__main__":
    unittest.main()


class OutcomeFeedback(unittest.TestCase):
    """P2b: a saturated reach has to be visible to the next PLAN turn.

    The executor computes FLAG_SATURATED on every tick and writes it into
    body_state.json, and until now nothing read it. The constitution promises
    "after each action you are told done, saturated, or timeout", so the words
    have to come from the flags rather than from a caller guessing.
    """

    def test_saturated_flag_is_the_word_saturated(self) -> None:
        self.assertEqual(
            brain.outcome_from_flags(brain.FLAG_SATURATED), "saturated"
        )

    def test_ttl_expired_is_the_word_timeout(self) -> None:
        self.assertEqual(brain.outcome_from_flags(brain.FLAG_TTL_EXPIRED), "timeout")

    def test_a_clean_tick_is_done(self) -> None:
        self.assertEqual(brain.outcome_from_flags(0), "done")

    def test_saturated_beats_timeout(self) -> None:
        # Both set: the body could not get there, which is the fact worth
        # retrying differently. "timeout" would invite the identical retry.
        self.assertEqual(
            brain.outcome_from_flags(brain.FLAG_SATURATED | brain.FLAG_TTL_EXPIRED),
            "saturated",
        )

    def test_fall_risk_and_unsupported_are_not_model_words(self) -> None:
        # The constitution offers exactly three words. Inventing a fourth would
        # break the promise; these flags belong in the trace, not the prompt.
        for flag in (brain.FLAG_FALL_RISK, brain.FLAG_UNSUPPORTED):
            self.assertEqual(brain.outcome_from_flags(flag), "done")

    def test_last_line_names_the_action_and_the_outcome(self) -> None:
        self.assertEqual(
            brain.last_line("reach", "right", "card", "saturated"),
            "reach right target=card → saturated",
        )

    def test_last_line_renders_a_missing_target_as_null(self) -> None:
        self.assertEqual(
            brain.last_line("look", "none", None, "done"),
            "look none target=null → done",
        )


class OutcomeReachesThePrompt(ChatPatchMixin, unittest.TestCase):
    def test_a_saturated_reach_appears_in_the_live_block(self) -> None:
        self.reply = '{"skill":"hold","side":"none","target":null}'
        record = brain.run_turn(
            brain.BrainConfig(),
            seed="",
            frame=fake_frame(),
            heard="take my card",
            history=[],
            last_action=("reach", "right", "card"),
            state_flags=brain.FLAG_SATURATED,
        )
        self.assertIn("reach right target=card → saturated", record.live)

    def test_the_planner_actually_sees_it(self) -> None:
        self.reply = '{"skill":"hold","side":"none","target":null}'
        brain.run_turn(
            brain.BrainConfig(),
            seed="",
            frame=fake_frame(),
            heard="take my card",
            history=[],
            last_action=("reach", "right", "card"),
            state_flags=brain.FLAG_SATURATED,
        )
        self.assertIn("saturated", self.user_text())

    def test_last_action_overrides_the_raw_last_string(self) -> None:
        self.reply = '{"skill":"hold","side":"none","target":null}'
        record = brain.run_turn(
            brain.BrainConfig(),
            seed="",
            frame=fake_frame(),
            heard="take my card",
            history=[],
            last="stale hand-written line",
            last_action=("reach", "right", "card"),
            state_flags=0,
        )
        self.assertNotIn("stale hand-written line", record.live)
        self.assertIn("→ done", record.live)

    def test_without_last_action_the_string_is_used_unchanged(self) -> None:
        self.reply = '{"skill":"hold","side":"none","target":null}'
        record = brain.run_turn(
            brain.BrainConfig(),
            seed="",
            frame=fake_frame(),
            heard="take my card",
            history=[],
            last="grab right target=card → done",
        )
        self.assertIn("grab right target=card → done", record.live)


class ReachOutcome(unittest.TestCase):
    """The game's own grab-range highlight is a better arrival oracle than a residual.

    Measured 2026-11-01 (`research/sonic-reach-envelope.md`): at the contract's
    0.02 m default tolerance `FLAG_SATURATED` is set on *every* reach, including
    ones the body tracked well, because ordinary SONIC tracking error is
    0.035-0.18 m. So the flag alone would tell the planner that every reach
    failed. Job Simulator lights the object blue when the hand is in grab range,
    which is the game answering the question instead of us inferring it.
    """

    def test_in_grab_range_is_done_even_when_the_flag_says_saturated(self) -> None:
        self.assertEqual(brain.reach_outcome("yes", brain.FLAG_SATURATED), "done")

    def test_out_of_grab_range_is_saturated(self) -> None:
        self.assertEqual(brain.reach_outcome("no", 0), "saturated")

    def test_an_unknown_cue_falls_back_to_the_flags(self) -> None:
        self.assertEqual(brain.reach_outcome("unknown", 0), "done")
        self.assertEqual(
            brain.reach_outcome("unknown", brain.FLAG_TTL_EXPIRED), "timeout"
        )

    def test_a_yes_cue_beats_a_timeout(self) -> None:
        # Both can be true at once: the command outlived its TTL and the hand is
        # nevertheless in range. Arrival is the fact that matters.
        self.assertEqual(brain.reach_outcome("yes", brain.FLAG_TTL_EXPIRED), "done")

    def test_a_recognised_yes_is_only_yes(self) -> None:
        # `yes`/`no` are the only two values that mean anything; anything else is
        # a cue we did not measure, and guessing would be worse than the flag.
        for cue in ("", "maybe", "YES", "1", "true"):
            self.assertEqual(
                brain.reach_outcome(cue, brain.FLAG_SATURATED), "saturated"
            )


class ReachOutcomeInThePrompt(ChatPatchMixin, unittest.TestCase):
    def test_a_reach_in_grab_range_reads_done(self) -> None:
        self.reply = '{"skill":"hold","side":"none","target":null}'
        record = brain.run_turn(
            brain.BrainConfig(), seed="", frame=fake_frame(),
            heard="take my card", history=[],
            last_action=("reach", "right", "card"),
            state_flags=brain.FLAG_SATURATED, glow="yes",
        )
        self.assertIn("reach right target=card → done", record.live)

    def test_a_reach_out_of_range_reads_saturated(self) -> None:
        self.reply = '{"skill":"hold","side":"none","target":null}'
        record = brain.run_turn(
            brain.BrainConfig(), seed="", frame=fake_frame(),
            heard="take my card", history=[],
            last_action=("reach", "right", "card"),
            state_flags=0, glow="no",
        )
        self.assertIn("reach right target=card → saturated", record.live)

    def test_glow_does_not_change_a_look_outcome(self) -> None:
        # The highlight is a hand-in-range signal. Letting it decide a look would
        # make "the object is lit" mean "the head arrived", which is not measured.
        self.reply = '{"skill":"hold","side":"none","target":null}'
        record = brain.run_turn(
            brain.BrainConfig(), seed="", frame=fake_frame(),
            heard="look at the human", history=[],
            last_action=("look", "none", None),
            state_flags=brain.FLAG_SATURATED, glow="yes",
        )
        self.assertIn("look none target=null → saturated", record.live)


class TaskIntent(unittest.TestCase):
    """The instruction names the verb and the object. The model was measured not to.

    JobBot's real line is "Hey human, take my card." Given that and one still, the model
    answered {"goal": "insert the card into the machine", "object": "INSERT CARTRIDGE
    machine"} - a later step, with an object it invented. So the words are read here.
    """

    def test_take_my_card(self):
        self.assertEqual(brain.task_intent("Hey human, take my card."), ("grab", "card"))

    def test_an_offer_is_a_grab(self):
        self.assertEqual(brain.task_intent("Here's my card."), ("grab", "card"))

    def test_the_walkthrough_line_names_a_cartridge(self):
        line = ("Alright. Why don't you grab one of the job cartridges off the table "
                "and let's get started.")
        self.assertEqual(brain.task_intent(line), ("grab", "job cartridge"))

    def test_a_place_keeps_its_noun(self):
        self.assertEqual(brain.task_intent("put the cartridge into the slot"),
                         ("place", "cartridge"))

    def test_two_word_verb(self):
        self.assertEqual(brain.task_intent("pick up the card from the table"),
                         ("grab", "card"))

    def test_a_participle_is_not_an_object(self):
        self.assertEqual(brain.task_intent("let's get started."), ("grab", ""))

    def test_no_verb_is_no_intent(self):
        self.assertEqual(brain.task_intent("Your hands are empty."), ("", ""))

    def test_nothing_is_not_an_object(self):
        self.assertEqual(brain.task_intent(""), ("", ""))


class _Response:
    """The tiny slice of urlopen's return value `server_status` uses."""

    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def read(self) -> bytes:
        return self.payload

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *exc) -> bool:
        return False


class ServerStatusTests(unittest.TestCase):
    """What the dashboard's Model card reports, from llama.cpp's own answers."""

    def probe(self, *outcomes):
        """`outcomes` are the urlopen results in call order."""
        queue = list(outcomes)
        calls = []

        def fake_urlopen(url, timeout=None):
            calls.append(url)
            outcome = queue.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        with mock.patch.object(brain.urllib.request, "urlopen", side_effect=fake_urlopen):
            status = brain.server_status("http://127.0.0.1:11434/")
        return status, calls

    def test_a_serving_server_names_its_weights(self) -> None:
        status, calls = self.probe(
            _Response(b'{"status":"ok"}'),
            _Response(b'{"models":[{"name":"/home/nate/AI/models/LFM3-Omni-2B-Q8_0.gguf"}]}'),
        )
        self.assertTrue(status.up)
        self.assertEqual(status.status, "ok")
        self.assertEqual(status.model, "LFM3-Omni-2B-Q8_0")
        self.assertEqual(calls[0], "http://127.0.0.1:11434/health")

    def test_loading_is_not_down(self) -> None:
        # llama.cpp answers 503 while it loads the weights: the operator should
        # see "loading", not a red dot for a server that is doing the work.
        error = brain.urllib.error.HTTPError(
            "http://x/health", 503, "loading", email.message.Message(), None
        )
        status, _ = self.probe(error)
        self.assertTrue(status.up)
        self.assertEqual(status.status, "loading")

    def test_a_stopped_server_is_down_and_asks_once(self) -> None:
        status, calls = self.probe(brain.urllib.error.URLError("connection refused"))
        self.assertFalse(status.up)
        self.assertEqual(status.status, "down")
        self.assertEqual(len(calls), 1)

    def test_a_bad_models_answer_still_reports_health(self) -> None:
        status, _ = self.probe(
            _Response(b'{"status":"ok"}'),
            brain.urllib.error.URLError("timed out"),
        )
        self.assertTrue(status.up)
        self.assertEqual(status.status, "ok")
        self.assertEqual(status.model, "")

    def test_short_model_drops_the_directory_and_extension(self) -> None:
        self.assertEqual(brain.short_model("/a/b/LFM3-Omni-2B-Q8_0.gguf"), "LFM3-Omni-2B-Q8_0")
        self.assertEqual(brain.short_model("omni"), "omni")
