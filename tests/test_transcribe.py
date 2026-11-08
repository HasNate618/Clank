"""Hearing the game say it: the model's transcript, and the parsing around it.

`_chat` is patched, so nothing here needs the llama server on 11434 or an audio device.
"""

import base64
import unittest

from clank import brain


class ChatPatchMixin:
    reply: str = ""
    reply_error: str = ""
    reply_decode_ms: float = 0.0
    _orig_chat = None

    def setUp(self) -> None:
        self.chat_calls: list[dict] = []
        self._orig_chat = brain._chat
        brain._chat = self._fake_chat

    def _fake_chat(self, config, messages, max_tokens=None):
        self.chat_calls.append({"messages": messages, "max_tokens": max_tokens})
        return brain.ModelReply(text=self.reply, decode_ms=self.reply_decode_ms,
                                error=self.reply_error)

    def tearDown(self) -> None:
        brain._chat = self._orig_chat

    def audio_parts(self) -> list[dict]:
        parts: list[dict] = []
        for call in self.chat_calls:
            for message in call["messages"]:
                content = message["content"]
                if isinstance(content, list):
                    parts.extend(p for p in content if p.get("type") == "input_audio")
        return parts


class CleanTranscriptTest(unittest.TestCase):
    def test_a_quoted_reply_loses_its_quotes(self):
        self.assertEqual(brain.clean_transcript('"Hello human!"'), "Hello human!")

    def test_the_silence_marker_becomes_empty(self):
        for marker in ("(silence)", "silence", "[silence]", "No speech detected.", "..."):
            self.assertEqual(brain.clean_transcript(marker), "", marker)

    def test_a_real_line_is_kept_whole(self):
        line = "Hello human! Welcome to the Job Simulator."
        self.assertEqual(brain.clean_transcript(f'  "{line}"\n'), line)

    def test_a_cut_sentence_is_not_punished(self):
        self.assertEqual(brain.clean_transcript("Alright, why dont you grab one of the job -"),
                         "Alright, why dont you grab one of the job -")

    def test_newlines_collapse_to_one_line(self):
        self.assertEqual(brain.clean_transcript("take my card\nthank you"), "take my card thank you")


class TranscribeTest(ChatPatchMixin, unittest.TestCase):
    def test_the_clip_is_sent_as_audio(self):
        self.reply = "Hello human!"
        result = brain.transcribe(brain.BrainConfig(), b"RIFFxxxxWAVE")
        self.assertEqual(result.text, "Hello human!")
        parts = self.audio_parts()
        self.assertEqual(len(parts), 1)
        self.assertEqual(parts[0]["input_audio"]["format"], "wav")
        self.assertEqual(base64.b64decode(parts[0]["input_audio"]["data"]), b"RIFFxxxxWAVE")

    def test_it_asks_for_words_not_a_decision(self):
        self.reply = "hello"
        brain.transcribe(brain.BrainConfig(), b"RIFF")
        text = [p["text"] for m in self.chat_calls[0]["messages"]
                for p in (m["content"] if isinstance(m["content"], list) else [])
                if p.get("type") == "text"]
        self.assertEqual(len(text), 1)
        self.assertIn("verbatim", text[0])
        self.assertNotIn("json", text[0].lower())

    def test_silence_is_empty_not_an_error(self):
        self.reply = "(silence)"
        result = brain.transcribe(brain.BrainConfig(), b"RIFF")
        self.assertTrue(result.ok)
        self.assertTrue(result.empty)

    def test_an_unfinished_line_is_flagged(self):
        self.reply = "Alright, why dont you grab one of the job -"
        self.assertTrue(brain.transcribe(brain.BrainConfig(), b"RIFF").ellipsis)

    def test_a_complete_line_is_not_flagged(self):
        self.reply = "Hello human! Welcome to the Job Simulator."
        self.assertFalse(brain.transcribe(brain.BrainConfig(), b"RIFF").ellipsis)

    def test_no_audio_is_an_error_without_a_call(self):
        result = brain.transcribe(brain.BrainConfig(), b"")
        self.assertFalse(result.ok)
        self.assertEqual(result.error, "no audio")
        self.assertEqual(self.chat_calls, [])

    def test_a_transport_error_is_a_field(self):
        self.reply_error = "transport: connection refused"
        result = brain.transcribe(brain.BrainConfig(), b"RIFF")
        self.assertFalse(result.ok)
        self.assertIn("connection refused", result.error)

    def test_timings_survive_to_the_caller(self):
        self.reply = "hi"
        self.reply_decode_ms = 240.0
        self.assertEqual(brain.transcribe(brain.BrainConfig(), b"RIFF").decode_ms, 240.0)

    def test_dict_shape_is_stable(self):
        self.reply = "hi"
        keys = set(brain.transcribe(brain.BrainConfig(), b"RIFF").to_dict())
        self.assertEqual(keys, {"text", "raw", "ellipsis", "error", "prompt_ms", "decode_ms"})


if __name__ == "__main__":
    unittest.main()
