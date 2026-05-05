import numpy as np
import struct
import time
from multiprocessing import shared_memory


class Shared_memory_link:
    """
    Docstring for Shared_memory_link:

    Shared memory bridge between Godot and Python.
    
    Memory Layout (87 bytes total):
    - Bytes 0-55: Observation (14 floats) - ball pos + 3 friendly + 3 enemy positions
    - Bytes 56-79: Action (6 floats) - power/angle pairs for 3 friendly players
    - Bytes 80-83: Reward (1 float)
    - Byte 84: Done flag (0=running, 1=episode over)
    - Byte 85: Step request flag (Python writes, Godot clears)
    - Byte 86: Reset request flag (Python writes, Godot clears)
    """
    
    # Memory offsets
    OBS_OFFSET = 0
    OBS_SIZE = 14 * 4
    ACT_OFFSET = 56
    ACT_SIZE = 6 * 4
    REWARD_OFFSET = 80
    REWARD_SIZE = 4
    DONE_OFFSET = 84
    STEP_REQ_OFFSET = 85
    RESET_REQ_OFFSET = 86
    TOTAL_SIZE = 87

    OBS_DIM = 14
    ACT_DIM = 6

    def __init__(self, name="BobbleFuckState", create=False, poll_interval=0.001):
        """
        Docstring for __init__:

        Args:
            name: Name of the shared memory segment (must match Godot side)
            create: If True, create the shared memory. If False, attach to existing.
            poll_interval: Time in seconds between polls when waiting (default 1ms)
        """
        self.name = name
        self.poll_interval = poll_interval

        if create:
            self.shm = shared_memory.SharedMemory(name=self.name, create=True, size=self.TOTAL_SIZE)
            self.shm.buf[:self.TOTAL_SIZE] = bytearray(self.TOTAL_SIZE)  # zero it out
        else:
            self.shm = shared_memory.SharedMemory(name=self.name, create=False)

    def write_action(self, action):
        """
        Docstring for write_action:

        Args:
            action: numpy array of 6 float32 values [f1_x, f1_y, f2_x, f2_y, f3_x, f3_y]
        """
        # convert action to bytes and write to shared mem
        action_bytes = np.array(action, dtype=np.float32).tobytes()
        self.shm.buf[self.ACT_OFFSET:self.ACT_OFFSET + self.ACT_SIZE] = action_bytes

    def read_observation(self):
        """
        Docstring for read_observation:

        Returns:
            numpy array of 14 float32 values
        """
        obs_bytes = bytes(self.shm.buf[self.OBS_OFFSET:self.OBS_OFFSET + self.OBS_SIZE])
        return np.frombuffer(obs_bytes, dtype=np.float32).copy()

    def read_reward(self):
        """
        Docstring for read_reward:

        Returns:
            float: reward value
        """
        reward_bytes = bytes(self.shm.buf[self.REWARD_OFFSET:self.REWARD_OFFSET + self.REWARD_SIZE])
        return struct.unpack('f', reward_bytes)[0]

    def read_done(self):
        """
        Docstring for read_done:

        Returns:
            bool: True if episode is over
        """
        return self.shm.buf[self.DONE_OFFSET] == 1

    def request_step(self):
        """
        Docstring for request_step:

        Tell Godot we want it to do a physics step.
        """
        self.shm.buf[self.STEP_REQ_OFFSET] = 1

    def wait_for_step(self, timeout=10.0):
        """
        Docstring for wait_for_step:

        Wait until Godot clears the step flag.

        Args:
            timeout: Maximum time to wait in seconds
        """
        start = time.time()
        while self.shm.buf[self.STEP_REQ_OFFSET] == 1:
            if time.time() - start > timeout:
                raise TimeoutError(f"Godot didn't respond within {timeout}s - is it running?")
            time.sleep(self.poll_interval)

    def request_reset(self):
        """
        Docstring for request_reset:

        Signal Godot to reset the environment.
        """
        self.shm.buf[self.RESET_REQ_OFFSET] = 1

    def wait_for_reset(self, timeout=10.0):
        """
        Docstring for wait_for_reset:

        Wait for Godot to complete the reset.

        Args:
            timeout: Maximum time to wait in seconds
        """
        start = time.time()
        while self.shm.buf[self.RESET_REQ_OFFSET] == 1:
            if time.time() - start > timeout:
                raise TimeoutError(f"Godot didn't reset within {timeout}s - check if game is running")
            time.sleep(self.poll_interval)

    def close(self):
        """
        Docstring for close:

        Clean up shared memory resources.
        """
        self.shm.close()
        try:
            self.shm.unlink()
        except FileNotFoundError:
            pass
