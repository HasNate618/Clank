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


if __name__ == "__main__":
    unittest.main()
