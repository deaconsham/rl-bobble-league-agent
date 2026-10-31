"""
fake godot for the mock game tests. opens the shm file, zeroes it, writes
protocol v2, then answers reset and step requests with simple fake physics.
no real engine, no graphics. runs in-process on a thread so a test can tell
it to hang or die.

it mirrors the bits of GameManager/DataManager/SharedMemory the python side
leans on: the learner is team 1, obs is 14 floats in the learner frame, a goal
sets done=1 and scorer, and the step/reset flags are cleared when the turn is
done. it does NOT copy the real jolt physics; positions just drift toward the
ball so an episode ends in a handful of turns.
"""
import os
import struct
import threading
import time

import numpy as np

# same layout as bobble_model/shared_memory_link.py (protocol v2, 120 bytes)
OBS_OFFSET = 0
ACT_OFFSET = 56
REWARD_OFFSET = 80
DONE_OFFSET = 84
STEP_REQ_OFFSET = 85
RESET_REQ_OFFSET = 86
LEARNER_TEAM_OFFSET = 87
SCORER_OFFSET = 88
PLAY_MODE_OFFSET = 89
OPP_VALID_OFFSET = 90
OPP_ACT_OFFSET = 92
VERSION_OFFSET = 116
TOTAL_SIZE = 120

PROTOCOL_VERSION = 2
OBS_DIM = 14
ACT_DIM = 6


class Mock_godot:
    """
    Docstring for Mock_godot:

    A tiny stand-in for the Godot game, talking protocol v2 over a file-backed
    mmap. Start it (creates the file), then attach Shared_memory_link(path=...,
    create=False) in the test. Call stop() when done.

    Args:
        path: backing file path (the test picks a temp file).
        learner_team: team the obs are framed for (1 like real AI_VS_AI).
        play_mode: 0 AI_VS_AI, 1 HUMAN_VS_AI, 2 HUMAN_VS_HUMAN.
        goal_after: score a goal once a turn count hits this (ends the episode).
        scorer: who scores when a goal lands (1 learner, 2 opponent).
        seed: seeds the fake physics so tests are repeatable.
        protocol_version: write a wrong number here to test old-build refusal.
    """

    def __init__(self, path, learner_team=1, play_mode=0, goal_after=3,
                 scorer=1, seed=0, protocol_version=PROTOCOL_VERSION):
        self.path = path
        self.learner_team = learner_team
        self.play_mode = play_mode
        self.goal_after = goal_after
        self.scorer = scorer
        self.protocol_version = protocol_version
        self.rng = np.random.default_rng(seed)

        self._hang = False
        self._stop = False
        self._turn = 0
        self._ball = np.zeros(2, dtype=np.float32)
        self._friendly = self.rng.uniform(-5, 5, size=(3, 2)).astype(np.float32)
        self._enemy = self.rng.uniform(-5, 5, size=(3, 2)).astype(np.float32)

        # create + zero the file, write the version (like SharedMemory boots)
        import mmap
        with open(self.path, "wb") as f:
            f.write(bytes(TOTAL_SIZE))
        self._file = open(self.path, "r+b")
        self.buf = mmap.mmap(self._file.fileno(), TOTAL_SIZE, access=mmap.ACCESS_WRITE)
        self.buf[VERSION_OFFSET] = self.protocol_version

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    # ---- controls the test uses ----

    def set_hang(self, hang):
        """when hung the loop stops clearing flags, so python times out."""
        self._hang = hang

    def die(self):
        """stop answering and release the file, like the process crashed."""
        self._stop = True
        self._thread.join(timeout=2.0)
        self._close_file()

    def stop(self):
        self._stop = True
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._close_file()

    def _close_file(self):
        if getattr(self, "buf", None) is not None:
            try:
                self.buf.close()
            except Exception:
                pass
            self.buf = None
        if getattr(self, "_file", None) is not None:
            try:
                self._file.close()
            except Exception:
                pass
            self._file = None

    # ---- fake engine loop ----

    def _write_obs(self):
        obs = np.concatenate([self._ball, self._friendly.ravel(), self._enemy.ravel()])
        obs = obs.astype(np.float32)
        self.buf[OBS_OFFSET:OBS_OFFSET + OBS_DIM * 4] = obs.tobytes()

    def _read_action(self):
        raw = self.buf[ACT_OFFSET:ACT_OFFSET + ACT_DIM * 4]
        return np.frombuffer(raw, dtype=np.float32).copy().reshape(3, 2)

    def _advance(self, action):
        """move friendly players a bit along their action, drift the ball."""
        self._friendly = self._friendly + 0.5 * action
        self._ball = self._ball + 0.2 * action.mean(axis=0)
        self._ball = np.clip(self._ball, -18.0, 18.0).astype(np.float32)
        self._friendly = np.clip(self._friendly, -10.0, 10.0).astype(np.float32)

    def _run(self):
        while not self._stop:
            if self._hang:
                time.sleep(0.005)
                continue

            # a reset request wins over a step, same as GameManager
            if self.buf[RESET_REQ_OFFSET] == 1:
                self._turn = 0
                self._ball = np.zeros(2, dtype=np.float32)
                self._write_obs()
                self.buf[REWARD_OFFSET:REWARD_OFFSET + 4] = struct.pack("<f", 0.0)
                self.buf[DONE_OFFSET] = 0
                self.buf[SCORER_OFFSET] = 0
                self.buf[LEARNER_TEAM_OFFSET] = self.learner_team
                self.buf[PLAY_MODE_OFFSET] = self.play_mode
                self.buf[RESET_REQ_OFFSET] = 0
                continue

            if self.buf[STEP_REQ_OFFSET] == 1:
                action = self._read_action()
                self._advance(action)
                self._turn += 1

                done = self._turn >= self.goal_after
                reward = 0.0
                scorer = 0
                if done:
                    scorer = self.scorer
                    reward = 1.0 if scorer == 1 else -1.0
                    # a goal sends everyone back to kickoff, like the real game
                    self._ball = np.zeros(2, dtype=np.float32)

                self._write_obs()
                self.buf[REWARD_OFFSET:REWARD_OFFSET + 4] = struct.pack("<f", reward)
                self.buf[DONE_OFFSET] = 1 if done else 0
                self.buf[SCORER_OFFSET] = scorer
                self.buf[LEARNER_TEAM_OFFSET] = self.learner_team
                self.buf[PLAY_MODE_OFFSET] = self.play_mode
                self.buf[STEP_REQ_OFFSET] = 0
                continue

            time.sleep(0.0005)
