import unittest

from clank import brain
from clank.brain import Ground, Plan
from clank.frames import Frame


def fake_frame(index: int = 0) -> Frame:
    return Frame(index=index, t_s=float(index), jpeg=b"\xff\xd8\xff\xd9", source="test")


class ChatPatchMixin:
    reply: str = ""
    reply_error: str = ""
    _orig_chat = None

    def setUp(self) -> None:
        self.chat_calls: list[dict] = []
        self._orig_chat = brain._chat
        brain._chat = self._fake_chat

    def _fake_chat(self, config, messages, max_tokens=None):
        self.chat_calls.append({"messages": messages, "max_tokens": max_tokens})
        return brain.ModelReply(text=self.reply, error=self.reply_error)

    def tearDown(self) -> None:
        brain._chat = self._orig_chat

    def user_text(self) -> str:
        """The text sent with the still on the last call."""
        content = self.chat_calls[-1]["messages"][-1]["content"]
        if isinstance(content, str):
            return content
        return "\n".join(part["text"] for part in content if part["type"] == "text")


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
        self.assertAlmostEqual(g.pixel[0], 0.270 + (0.548 - 0.270) * 0.75)
        self.assertAlmostEqual(g.pixel[1], 0.325 + (0.566 - 0.325) * 0.25)

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
        self.assertEqual(as_dict["pixel"], list(g.pixel))
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

    def test_none_does_not_match(self) -> None:
        self.reply = "NONE"
        v = brain.verify(brain.BrainConfig(), fake_frame(), "card")
        self.assertFalse(v.match)

    def test_empty_reply_does_not_match(self) -> None:
        self.reply = "   "
        v = brain.verify(brain.BrainConfig(), fake_frame(), "card")
        self.assertFalse(v.match)
        self.assertEqual(v.heard, "")

    def test_a_noun_with_no_words_cannot_match(self) -> None:
        self.reply = "a card"
        v = brain.verify(brain.BrainConfig(), fake_frame(), "")
        self.assertFalse(v.match)

    def test_transport_error_does_not_match_and_keeps_the_message(self) -> None:
        self.reply_error = "transport: timed out"
        v = brain.verify(brain.BrainConfig(), fake_frame(), "card")
        self.assertFalse(v.match)
        self.assertEqual(v.error, "transport: timed out")

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
            },
        )


if __name__ == "__main__":
    unittest.main()
