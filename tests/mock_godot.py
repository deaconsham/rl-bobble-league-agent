"""
fake godot for the mock game tests. opens the shm file, zeroes it, writes
protocol v4, then answers reset and step requests with simple fake physics.
no real engine, no graphics. runs in-process on a thread so a test can tell
it to hang or die.

it mirrors the bits of GameManager/DataManager/SharedMemory the python side
leans on: the learner is team 1, obs is 14 floats in the learner frame, a goal
sets done=1 and scorer, the step/reset flags are cleared when the turn is done,
and each step reports a fake turn length in game seconds. it does NOT copy the
real jolt physics; positions just drift toward the ball so an episode ends in a
handful of turns.
"""
import os
import struct
import threading
import time

import numpy as np

# same layout as bobble_model/shared_memory_link.py (protocol v4, 124 bytes)
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
TURN_SECONDS_OFFSET = 120
TOTAL_SIZE = 124

PROTOCOL_VERSION = 4
OBS_DIM = 14
ACT_DIM = 6


class Mock_godot:
    """
    Docstring for Mock_godot:

    A tiny stand-in for the Godot game, talking protocol v4 over a file-backed
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
        action = np.frombuffer(raw, dtype=np.float32).copy().reshape(3, 2)
        # protocol v3: [angle, power] per player -> (x, z) * power, like Godot decodes.
        # learner frame so FrameZ is +1; angle a*pi from -z, power 0.2+0.8*(p+1)/2.
        a = np.clip(action[:, 0], -1.0, 1.0)
        p = np.clip(action[:, 1], -1.0, 1.0)
        theta = a * np.pi
        dirs = np.stack([np.sin(theta), -np.cos(theta)], axis=-1)
        power = 0.2 + 0.8 * (p + 1.0) / 2.0
        return (dirs * power[:, None]).astype(np.float32)

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
                # fake turn length in game seconds: a small spread so the
                # turn-length histogram has something to show (real Godot sends simTime)
                turn_seconds = float(2.0 + 0.5 * (self._turn % 4))
                self.buf[TURN_SECONDS_OFFSET:TURN_SECONDS_OFFSET + 4] = struct.pack("<f", turn_seconds)
                self.buf[STEP_REQ_OFFSET] = 0
                continue

            time.sleep(0.0005)


class Mock_launcher:
    """
    Docstring for Mock_launcher class:

    Stand-in for launch_games.Game_launcher used by the Vector_env mock tests.
    Instead of starting real Godots it spins up Mock_godot threads, one per game,
    each on its own shm file under a temp dir. Exposes the same surface Vector_env
    leans on: shm_paths, launch_one, launch_all, wait_for_handshake, stop_one,
    stop_all, plus the Shared_memory_link bridges the tests attach.

    Args:
        n_games: number of games.
        tmp_dir: folder for the per-game shm files.
        goal_after: turns before each mock game scores (ends an episode).
        hang_games: set of indices whose mock starts hung (to test the timeout).
    """

    def __init__(self, n_games, tmp_dir, goal_after=3, hang_games=None):
        import os as _os
        self.n_games = n_games
        self.tmp_dir = str(tmp_dir)
        self.goal_after = goal_after
        self.hang_games = set(hang_games or [])
        self.shm_paths = [_os.path.join(self.tmp_dir, f"mmap_{i}.dat") for i in range(n_games)]
        self.games = [None] * n_games
        self.pids = [None] * n_games
        self.launch_counts = [0] * n_games

    def launch_one(self, index):
        """start (or restart) the mock game at this index on its own shm file."""
        if self.games[index] is not None:
            self.games[index].stop()
        game = Mock_godot(self.shm_paths[index], goal_after=self.goal_after,
                          seed=index)
        if index in self.hang_games:
            game.set_hang(True)
        self.games[index] = game
        self.launch_counts[index] += 1
        self.pids[index] = 1000 + index
        return self.pids[index]

    def launch_all(self):
        for i in range(self.n_games):
            self.launch_one(i)
        return self.pids

    def wait_for_handshake(self, index, timeout=10.0):
        from shared_memory_link import Shared_memory_link
        import time as _time
        deadline = _time.time() + timeout
        while _time.time() < deadline:
            try:
                return Shared_memory_link(path=self.shm_paths[index], create=False)
            except (RuntimeError, ValueError, FileNotFoundError):
                _time.sleep(0.02)
        raise TimeoutError(f"mock game {index} never handshook")

    def kill_game(self, index):
        """simulate a crash: the mock stops answering and releases its file."""
        if self.games[index] is not None:
            self.games[index].die()

    def stop_one(self, index):
        if self.games[index] is not None:
            self.games[index].stop()
            self.games[index] = None
        self.pids[index] = None

    def stop_all(self):
        for i in range(self.n_games):
            self.stop_one(i)
