import mmap
import os
import struct
import tempfile
import time

import numpy as np


DEFAULT_FILE_NAME = "bobble_league_mmap.dat"
PATH_ENV_VAR = "BOBBLE_SHM_PATH"


def resolve_path():
    """Same rule as SharedMemory.cs: env var wins, else <temp dir>/bobble_league_mmap.dat."""
    return os.environ.get(PATH_ENV_VAR) or os.path.join(tempfile.gettempdir(), DEFAULT_FILE_NAME)


class Shared_memory_link:
    """
    Docstring for Shared_memory_link:

    File-backed memory-map bridge between Godot and Python.
    Works on Windows, macOS and Linux (both sides map the same file).

    Protocol v2 layout (120 bytes). Must match bobble_fuck/Scripts/SharedMemory.cs.
    - 0-55    Observation (14 f32), learner's frame       Godot -> Python
    - 56-79   Learner action (6 f32), learner's frame     Python -> Godot
    - 80-83   Reward (f32), goal reward from learner POV  Godot -> Python
    - 84      Done (u8, 1 = goal scored)                  Godot -> Python
    - 85      Step request (u8)                           Python sets, Godot clears
    - 86      Reset request (u8)                          Python sets, Godot clears
    - 87      Learner team (u8, 1 or 2)                   Godot -> Python
    - 88      Scorer (u8, 0 none, 1 learner, 2 opponent)  Godot -> Python
    - 89      Play mode (u8, see PLAY_MODES)              Godot -> Python
    - 90      Opponent action valid (u8)                  Python -> Godot
    - 92-115  Opponent action (6 f32), opponent's frame   Python -> Godot
    - 116     Protocol version (u8)                       Godot -> Python

    Start Godot FIRST: it zeroes the buffer and writes the version when it boots.
    """

    PROTOCOL_VERSION = 2

    OBS_OFFSET = 0
    OBS_SIZE = 14 * 4
    ACT_OFFSET = 56
    ACT_SIZE = 6 * 4
    REWARD_OFFSET = 80
    REWARD_SIZE = 4
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

    OBS_DIM = 14
    ACT_DIM = 6

    PLAY_MODES = {0: "AI_VS_AI", 1: "HUMAN_VS_AI", 2: "HUMAN_VS_HUMAN"}

    def __init__(self, path=None, create=False, poll_interval=0.001, check_version=True):
        """
        Docstring for __init__:

        Args:
            path: Backing file path (default: resolve_path(), must match Godot side)
            create: If True, create/zero the file (tests only). If False, Godot must have made it.
            poll_interval: Time in seconds between polls when waiting (default 1ms)
            check_version: Refuse to attach to a buffer from an older Godot build.
        """
        self.path = path or resolve_path()
        self.poll_interval = poll_interval

        if create:
            with open(self.path, "wb") as f:
                f.write(bytes(self.TOTAL_SIZE))
        elif not os.path.exists(self.path):
            raise FileNotFoundError(
                f"No shared memory file at '{self.path}'. Start the Godot game first."
            )

        self._file = open(self.path, "r+b")
        if os.fstat(self._file.fileno()).st_size < self.TOTAL_SIZE:
            self._file.truncate(self.TOTAL_SIZE)
        self.buf = mmap.mmap(self._file.fileno(), self.TOTAL_SIZE, access=mmap.ACCESS_WRITE)

        if create:
            self.buf[self.VERSION_OFFSET] = self.PROTOCOL_VERSION
        elif check_version and self.buf[self.VERSION_OFFSET] != self.PROTOCOL_VERSION:
            found = self.buf[self.VERSION_OFFSET]
            self.close()
            raise RuntimeError(
                f"Shared memory at '{self.path}' is protocol v{found}, expected "
                f"v{self.PROTOCOL_VERSION}. Rebuild and restart the Godot game."
            )

    # ---- python -> godot ----

    def write_action(self, action):
        """
        Args:
            action: 6 float32 values [p1_x, p1_z, p2_x, p2_z, p3_x, p3_z], learner's frame
        """
        data = np.asarray(action, dtype=np.float32).reshape(self.ACT_DIM).tobytes()
        self.buf[self.ACT_OFFSET:self.ACT_OFFSET + self.ACT_SIZE] = data

    def write_opponent_action(self, action):
        """
        Args:
            action: 6 float32 values in the OPPONENT's own frame, or None so the
                    opponent stands still (and is ignored in HUMAN_VS_AI anyway).
        """
        if action is None:
            self.buf[self.OPP_VALID_OFFSET] = 0
            return
        data = np.asarray(action, dtype=np.float32).reshape(self.ACT_DIM).tobytes()
        self.buf[self.OPP_ACT_OFFSET:self.OPP_ACT_OFFSET + self.ACT_SIZE] = data
        self.buf[self.OPP_VALID_OFFSET] = 1

    def request_step(self):
        """Tell Godot to play a turn. Call after write_action / write_opponent_action."""
        self.buf[self.STEP_REQ_OFFSET] = 1

    def request_reset(self):
        """Signal Godot to reset the environment."""
        self.buf[self.RESET_REQ_OFFSET] = 1

    # ---- godot -> python ----

    def read_observation(self):
        """Returns: numpy array of 14 float32 values (world units, learner's frame)."""
        obs_bytes = self.buf[self.OBS_OFFSET:self.OBS_OFFSET + self.OBS_SIZE]
        return np.frombuffer(obs_bytes, dtype=np.float32).copy()

    def read_reward(self):
        reward_bytes = self.buf[self.REWARD_OFFSET:self.REWARD_OFFSET + self.REWARD_SIZE]
        return struct.unpack('<f', reward_bytes)[0]

    def read_done(self):
        return self.buf[self.DONE_OFFSET] == 1

    def read_learner_team(self):
        return self.buf[self.LEARNER_TEAM_OFFSET]

    def read_scorer(self):
        """0 = nobody scored this turn, 1 = learner scored, 2 = opponent scored."""
        return self.buf[self.SCORER_OFFSET]

    def read_play_mode(self):
        return self.PLAY_MODES.get(self.buf[self.PLAY_MODE_OFFSET], "UNKNOWN")

    # ---- waiting ----

    def _wait_flag_clear(self, offset, timeout, what):
        start = time.time()
        while self.buf[offset] == 1:
            if timeout is not None and time.time() - start > timeout:
                raise TimeoutError(f"Godot didn't {what} within {timeout}s - is the game running?")
            time.sleep(self.poll_interval)

    def wait_for_step(self, timeout=120.0):
        """Wait until Godot clears the step flag. timeout=None waits forever (human play)."""
        self._wait_flag_clear(self.STEP_REQ_OFFSET, timeout, "finish the turn")

    def wait_for_reset(self, timeout=120.0):
        """Wait for Godot to complete the reset. timeout=None waits forever."""
        self._wait_flag_clear(self.RESET_REQ_OFFSET, timeout, "reset")

    def close(self):
        """Unmap and close. The file stays on disk so Godot keeps working."""
        if getattr(self, "buf", None) is not None:
            self.buf.close()
            self.buf = None
        if getattr(self, "_file", None) is not None:
            self._file.close()
            self._file = None
