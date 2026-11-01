import unittest

from clank import brain
from clank.brain import Ground, Plan
from clank.frames import Frame


def fake_frame(index: int = 0) -> Frame:
    return Frame(index=index, t_s=float(index), jpeg=b"\xff\xd8\xff\xd9", source="test")


class ChatPatchMixin:
    reply: str = ""
    _orig_chat = None

    def setUp(self) -> None:
        self._orig_chat = brain._chat
        brain._chat = lambda *a, **k: brain.ModelReply(text=self.reply)

    def tearDown(self) -> None:
        brain._chat = self._orig_chat


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


class PromptText(unittest.TestCase):
    def test_seed_names_the_green_cartridge_the_slot_and_the_lever(self) -> None:
        s = brain.DEFAULT_SEED.lower()
        self.assertIn("store clerk", s)
        self.assertIn("green", s)
        self.assertIn("ghost", s)
        self.assertIn("lever", s)

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


if __name__ == "__main__":
    unittest.main()
