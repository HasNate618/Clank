"""The session mailbox: four small json files a 50 Hz body process and an adapter share.

Nothing here blocks. The body process ticks MuJoCo every 20 ms and must never wait for
an adapter that is spending milliseconds generating motion, so each direction is a
latest-wins file written with a pid-suffixed temp and `os.replace`. The tests pin the
three properties that make that safe: the pose ring is oldest-first and capped at 32,
a missing/empty/torn file reads as `None` instead of raising, and the play request
carries only `seq` and a clip path - joint-space motion, no world coordinates and no
clip bytes.
"""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import tempfile
import unittest
import unittest.mock

WBC = pathlib.Path(__file__).resolve().parents[1] / "spike" / "wbc"
_SPEC = importlib.util.spec_from_file_location("mb_mailbox_test", WBC / "mb_mailbox.py")
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover - a missing file, not a branch
    raise RuntimeError("could not load spike/wbc/mb_mailbox.py")
mb_mailbox = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mb_mailbox)


def pose(root_x: float = 0.0) -> list[float]:
    """A 36-float pose whose only interesting value is the root x."""
    return [root_x] + [0.0] * 35


class Mailbox(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = pathlib.Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()


class Pose36Ring(Mailbox):
    def test_missing_file_reads_none(self) -> None:
        self.assertIsNone(mb_mailbox.read_pose36(self.dir))

    def test_ring_caps_at_32_oldest_first(self) -> None:
        ring: list = []
        for i in range(40):
            ring = mb_mailbox.write_pose36(
                self.dir, pose(float(i)), seq=i, t_mono_ns=1000 + i, prev=ring
            )
        read = mb_mailbox.read_pose36(self.dir)
        self.assertIsNotNone(read)
        self.assertEqual(len(read["poses"]), 32)
        self.assertEqual(len(ring), 32)
        self.assertEqual(read["poses"][0][0], 8.0)
        self.assertEqual(read["poses"][-1][0], 39.0)
        self.assertEqual(read["poses"], ring)

    def test_read_after_two_writes_has_the_second_seq(self) -> None:
        mb_mailbox.write_pose36(self.dir, pose(1.0), seq=1, t_mono_ns=10, prev=None)
        mb_mailbox.write_pose36(self.dir, pose(2.0), seq=2, t_mono_ns=20, prev=None)
        read = mb_mailbox.read_pose36(self.dir)
        self.assertEqual(read["seq"], 2)
        self.assertEqual(read["t_mono_ns"], 20)
        self.assertEqual(read["hz"], 50)
        self.assertEqual([row[0] for row in read["poses"]], [1.0, 2.0])

    def test_prev_ring_is_used_instead_of_the_file(self) -> None:
        mb_mailbox.write_pose36(self.dir, pose(1.0), seq=1, t_mono_ns=10, prev=None)
        ring = mb_mailbox.write_pose36(self.dir, pose(2.0), seq=2, t_mono_ns=20, prev=[])
        self.assertEqual(ring, [pose(2.0)])
        self.assertEqual([row[0] for row in mb_mailbox.read_pose36(self.dir)["poses"]], [2.0])

    def test_prev_defaults_to_none_and_reads_the_file(self) -> None:
        # The 50 Hz call site passes four arguments; the default must rebuild the ring
        # from the file it is about to replace.
        mb_mailbox.write_pose36(self.dir, pose(1.0), seq=1, t_mono_ns=10)
        mb_mailbox.write_pose36(self.dir, pose(2.0), seq=2, t_mono_ns=20)
        read = mb_mailbox.read_pose36(self.dir)
        self.assertEqual(read["seq"], 2)
        self.assertEqual([row[0] for row in read["poses"]], [1.0, 2.0])

    def test_payload_keys_are_the_context_contract(self) -> None:
        mb_mailbox.write_pose36(self.dir, pose(1.0), seq=7, t_mono_ns=70, prev=None)
        raw = json.loads((self.dir / mb_mailbox.POSE36_NAME).read_bytes())
        self.assertEqual(set(raw), {"seq", "t_mono_ns", "hz", "poses"})
        self.assertEqual(len(raw["poses"][0]), 36)

    def test_empty_and_torn_files_read_none(self) -> None:
        path = self.dir / mb_mailbox.POSE36_NAME
        path.write_bytes(b"")
        self.assertIsNone(mb_mailbox.read_pose36(self.dir))
        path.write_bytes(b'{"seq": 3, "t_mono_ns": 30, "hz": 50, "poses": [[0.0, 1.0')
        self.assertIsNone(mb_mailbox.read_pose36(self.dir))

    def test_wrong_shape_reads_none(self) -> None:
        path = self.dir / mb_mailbox.POSE36_NAME
        short_row = {"seq": 3, "t_mono_ns": 30, "hz": 50, "poses": [[0.0]]}
        path.write_bytes(json.dumps(short_row).encode())
        self.assertIsNone(mb_mailbox.read_pose36(self.dir))
        path.write_bytes(json.dumps([1, 2, 3]).encode())
        self.assertIsNone(mb_mailbox.read_pose36(self.dir))

    def test_empty_ring_reads_none(self) -> None:
        # The writer never emits `{"poses": []}`; accepting it would hand a caller a
        # valid-looking context whose `ring[-1]` raises IndexError.
        payload = {"seq": 3, "t_mono_ns": 30, "hz": 50, "poses": []}
        (self.dir / mb_mailbox.POSE36_NAME).write_bytes(json.dumps(payload).encode())
        self.assertIsNone(mb_mailbox.read_pose36(self.dir))

    def test_wrong_length_pose_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            mb_mailbox.write_pose36(self.dir, [0.0] * 35, seq=1, t_mono_ns=10, prev=None)


class Play(Mailbox):
    def test_play_is_latest_wins(self) -> None:
        mb_mailbox.write_play(self.dir, seq=1)
        mb_mailbox.write_play(self.dir, seq=2)
        self.assertEqual(mb_mailbox.read_play(self.dir), {"seq": 2, "path": "mb_clip.npz"})

    def test_play_payload_has_no_xyz_or_clip_bytes(self) -> None:
        mb_mailbox.write_play(self.dir, seq=4, path="mb_clip.npz")
        raw = json.loads((self.dir / mb_mailbox.PLAY_NAME).read_bytes())
        self.assertEqual(set(raw), {"seq", "path"})
        self.assertIsInstance(raw["seq"], int)
        self.assertIsInstance(raw["path"], str)

    def test_missing_play_reads_none(self) -> None:
        self.assertIsNone(mb_mailbox.read_play(self.dir))

    def test_torn_play_reads_none(self) -> None:
        (self.dir / mb_mailbox.PLAY_NAME).write_bytes(b'{"seq": 1, "path"')
        self.assertIsNone(mb_mailbox.read_play(self.dir))


class Status(Mailbox):
    def test_status_round_trip(self) -> None:
        mb_mailbox.write_status(self.dir, seq_ack=3, status="playing", frame=12, n=48)
        read = mb_mailbox.read_status(self.dir)
        self.assertEqual(
            read, {"seq_ack": 3, "status": "playing", "frame": 12, "n": 48}
        )

    def test_status_is_latest_wins(self) -> None:
        mb_mailbox.write_status(self.dir, seq_ack=3, status="playing", frame=12, n=48)
        mb_mailbox.write_status(self.dir, seq_ack=3, status="done", frame=48, n=48)
        self.assertEqual(mb_mailbox.read_status(self.dir)["status"], "done")
        self.assertEqual(mb_mailbox.read_status(self.dir)["frame"], 48)

    def test_unknown_status_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            mb_mailbox.write_status(self.dir, seq_ack=3, status="paused", frame=0, n=48)

    def test_missing_and_torn_status_read_none(self) -> None:
        self.assertIsNone(mb_mailbox.read_status(self.dir))
        (self.dir / mb_mailbox.STATUS_NAME).write_bytes(b"\x00\x00garbage")
        self.assertIsNone(mb_mailbox.read_status(self.dir))


class ClearStale(Mailbox):
    """A new body process starts from an empty mailbox, whatever the last run left.

    `--out` is `session_root/body`: persistent, not one of the per-session directories, so
    a finished run's files are still there at the next start. A leftover `mb_play.json` is
    indistinguishable from a fresh deposit - nothing has loaded a seq yet - and used to
    play the previous session's clip on the first tick with no adapter running.
    """

    def _write_a_finished_runs_mailbox(self) -> None:
        mb_mailbox.write_pose36(self.dir, pose(0.5), seq=99, t_mono_ns=10, prev=None)
        mb_mailbox.write_play(self.dir, seq=99)
        (self.dir / mb_mailbox.CLIP_NAME).write_bytes(b"not really a zip")
        mb_mailbox.write_status(self.dir, seq_ack=99, status="done", frame=4, n=5)

    def test_a_finished_runs_files_are_all_removed(self) -> None:
        self._write_a_finished_runs_mailbox()
        self.assertEqual(mb_mailbox.clear_stale(self.dir), [])
        self.assertIsNone(mb_mailbox.read_play(self.dir))
        self.assertIsNone(mb_mailbox.read_pose36(self.dir))
        self.assertIsNone(mb_mailbox.read_status(self.dir))
        self.assertFalse((self.dir / mb_mailbox.CLIP_NAME).exists())

    def test_the_status_file_goes_too(self) -> None:
        # A stale `done`/`seq_ack` is the other half of the trap: it reads as this
        # process's own first acknowledgement of a request it never received.
        mb_mailbox.write_status(self.dir, seq_ack=4, status="done", frame=9, n=10)
        mb_mailbox.clear_stale(self.dir)
        self.assertFalse((self.dir / mb_mailbox.STATUS_NAME).exists())

    def test_missing_files_are_not_a_failure(self) -> None:
        self.assertEqual(mb_mailbox.clear_stale(self.dir), [])

    def test_a_missing_directory_is_not_a_failure(self) -> None:
        # The body is started before the session directory is guaranteed to exist.
        self.assertEqual(mb_mailbox.clear_stale(self.dir / "absent"), [])

    def test_a_file_that_cannot_be_removed_is_reported_not_raised(self) -> None:
        # The body has to start either way; a directory in place of the clip is the
        # shortest stand-in for an unlink that fails.
        (self.dir / mb_mailbox.CLIP_NAME).mkdir()
        mb_mailbox.write_play(self.dir, seq=1)
        self.assertEqual(mb_mailbox.clear_stale(self.dir), [mb_mailbox.CLIP_NAME])
        self.assertFalse((self.dir / mb_mailbox.PLAY_NAME).exists(), "the rest still go")

    def test_files_that_are_not_the_mailbox_are_left_alone(self) -> None:
        # `--out` is also where the rendered frame, the state file, the log and the
        # command socket live: this clears four names, not the directory.
        keep = ("body_state.json", "body_latest.jpg", "body_view.log", "control.sock")
        for name in keep:
            (self.dir / name).write_bytes(b"keep me")
        mb_mailbox.clear_stale(self.dir)
        for name in keep:
            self.assertTrue((self.dir / name).exists(), name)


class Atomic(Mailbox):
    def test_write_leaves_no_temp_file_behind(self) -> None:
        mb_mailbox.write_pose36(self.dir, pose(1.0), seq=1, t_mono_ns=10, prev=None)
        self.assertEqual([p.name for p in self.dir.iterdir()], [mb_mailbox.POSE36_NAME])

    def test_temp_name_carries_the_pid(self) -> None:
        # Observed, not inferred: a shared temp name also writes the file and cleans up
        # after itself, and only the recorded source path shows whether two writers can
        # collide. Two writers sharing one tmp is the FileNotFoundError storm.
        replaced: list[pathlib.Path] = []
        real_replace = os.replace

        def record(src: pathlib.Path, dst: pathlib.Path) -> None:
            replaced.append(pathlib.Path(src))
            real_replace(src, dst)

        with unittest.mock.patch.object(os, "replace", side_effect=record):
            mb_mailbox.write_pose36(self.dir, pose(1.0), seq=1, t_mono_ns=10, prev=None)
        self.assertEqual(
            replaced, [self.dir / f"{mb_mailbox.POSE36_NAME}.{os.getpid()}.tmp"]
        )

    def test_another_writers_temp_file_does_not_block(self) -> None:
        # The temp name carries the pid: a shared one let each writer's replace()
        # find the other's tmp already renamed away.
        (self.dir / f"{mb_mailbox.POSE36_NAME}.999999.tmp").write_bytes(b"stale")
        mb_mailbox.write_pose36(self.dir, pose(1.0), seq=1, t_mono_ns=10, prev=None)
        self.assertEqual(mb_mailbox.read_pose36(self.dir)["seq"], 1)


if __name__ == "__main__":
    unittest.main()
