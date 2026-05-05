import numpy as np
import gymnasium as gym
from shared_memory_link import Shared_memory_link


class Bobble_fuck_env(gym.Env):
    """
    Docstring for Bobble_fuck_env:
    
    Gymnasium environment for Bobble League reinforcement learning.
    Communicates with Godot via shared memory bridge.
    
    Observation Space:
        - Ball position (x, y): 2 values
        - 3 Friendly player positions (x, y): 6 values
        - 3 Enemy player positions (x, y): 6 values
        Total: 14-dimensional observation vector
    
    Action Space:
        - 3 Friendly player actions (x, y): 6 values
        Total: 6-dimensional continuous action vector
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(self):
        """
        Docstring for __init__:
        
        Initialize the Bobble League environment.
        Creates shared memory bridge to communicate with Godot.
        """
        super().__init__()
        
        # Observation space: 14-dim vector
        # [ball_x, ball_y, f1_x, f1_y, f2_x, f2_y, f3_x, f3_y, e1_x, e1_y, e2_x, e2_y, e3_x, e3_y]
        self.observation_space = gym.spaces.Box(
            low=np.array([-1.0] * 14, dtype=np.float32),
            high=np.array([1.0] * 14, dtype=np.float32),
            dtype=np.float32
        )
        
        # Action space: 6-dim vector
        # [f1_x, f1_y, f2_x, f2_y, f3_x, f3_y]
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0] * 6, dtype=np.float32),
            high=np.array([1.0] * 6, dtype=np.float32),
            dtype=np.float32
        )
        
        # Episode tracking
        self.ep_len = 0
        self.max_ep_len = 1000
        
        # Shared memory bridge (attach to Godot-created shared memory)
        self.bridge = Shared_memory_link(create=False)
    
    def _normalize_obs(self, obs):
        """
        Docstring for _normalize_obs:
        
        Normalizes the raw Godot observations so neural networks process them better.
        Assumes field limits of X = [-50, 50] and Y = [-15, 15].
        """
        
        norm_factors = np.array([50.0, 15.0] * 7, dtype=np.float32)
        normalized_obs = np.clip(obs / norm_factors, -1.0, 1.0)
        return normalized_obs

    def reset(self):
        """
        Docstring for reset:
        
        Reset the environment to initial state.
        Signals Godot to reset the game and waits for initial observation.
        """
        self.ep_len = 0
        
        # Request Godot to reset the game
        self.bridge.request_reset()
        self.bridge.wait_for_reset()
        
        # Read initial observation from Godot and normalize it
        raw_obs = self.bridge.read_observation()
        
        return self._normalize_obs(raw_obs)
    
    def step(self, action):
        """
        Docstring for step:
        
        Execute one step in the environment.
        Sends action to Godot via shared memory and waits for the result.
        
        Args:
            action: Action to take (6-dim numpy array)
            
        Returns:
            observation: Current observation after action (14-dim numpy array)
            reward: Reward received (float)
            done: Whether episode ended (bool)
            info: Additional information (dict)
        """
        # Write action to shared memory
        self.bridge.write_action(action)
        
        # Signal Godot to step and wait for completion
        self.bridge.request_step()
        self.bridge.wait_for_step()
        
        self.ep_len += 1
        
        # Read results from Godot
        raw_obs = self.bridge.read_observation()
        reward = self.bridge.read_reward()
        done = self.bridge.read_done() or (self.ep_len >= self.max_ep_len)
        info = {}
        
        return self._normalize_obs(raw_obs), reward, done, info
    
    def render(self, mode='human'):
        """
        Docstring for render:
        
        Render the environment (handled by Godot).
        
        Args:
            mode: Rendering mode
        """
        # Rendering handled by Godot
        pass
    
    def close(self):
        """
        Docstring for close:
        
        Clean up environment resources.
        Closes the shared memory bridge.
        """
        self.bridge.close()


# Helper function to create environment (used by training loop)
def bobble_fuck_env_helper():
    """
    Docstring for bobble_fuck_env_helper:
    
    Factory function to create bobble_fuck_env instance.
    Following OpenAI Spinning Up convention of using env_fn.
    
    Returns:
        env: bobble_fuck_env instance
    """
    return Bobble_fuck_env()