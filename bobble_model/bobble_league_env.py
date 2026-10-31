from dataclasses import dataclass

import numpy as np
import gymnasium as gym

from shared_memory_link import Shared_memory_link


# ---- field geometry (world units, from Scenes/scene.tscn: Arena mesh scaled 0.5) ----
# x runs across the pitch, z runs goal to goal.
FIELD_HALF_X = 10.0   # side walls at x = +-10
FIELD_HALF_Z = 18.5   # back of the goal nets at z = +-18.5
GOAL_Z = 16.5         # goal trigger areas at z = +-16.5

# In the learner's frame (see DataManager.FrameZ) the learner ALWAYS attacks -z.
OPP_GOAL = np.array([0.0, -GOAL_Z], dtype=np.float32)
OWN_GOAL = np.array([0.0, GOAL_Z], dtype=np.float32)

OBS_SCALE = np.array([FIELD_HALF_X, FIELD_HALF_Z] * 7, dtype=np.float32)


def normalize_obs(raw_obs):
    """World units -> roughly [-1, 1]."""
    return np.clip(np.asarray(raw_obs, dtype=np.float32) / OBS_SCALE, -1.0, 1.0)


def split_obs(raw_obs):
    """Returns (ball[2], friendly[3,2], enemy[3,2])."""
    o = np.asarray(raw_obs, dtype=np.float32).reshape(7, 2)
    return o[0], o[1:4], o[4:7]


def opponent_view(raw_obs):
    """
    Learner-frame obs -> the same moment seen from the opponent's frame.
    The two frames differ by a z flip, and friendly/enemy swap.
    """
    ball, friendly, enemy = split_obs(raw_obs)
    flip = np.array([1.0, -1.0], dtype=np.float32)
    return np.concatenate([ball * flip, (enemy * flip).ravel(), (friendly * flip).ravel()])


@dataclass
class RewardConfig:
    """
    Per-turn shaping added on top of Godot's goal reward (+1 / -1).
    Positions only change between turns, so "velocity" here means
    displacement over one turn. Distances are divided by FIELD_HALF_Z.
    Set a weight to 0 to switch that term off.
    """
    goal: float = 1.0               # scales Godot's +-1
    ball_to_goal: float = 0.5       # ball moved closer to the opponent goal
    players_to_ball: float = 0.05   # friendly players closed in on the ball (mean)
    defence: float = 0.02           # penalty when no friendly is goal-side of the ball
    time_penalty: float = 0.0       # flat cost per turn, pushes for faster goals


class Bobble_league_env(gym.Env):
    """
    Gymnasium environment for Bobble League reinforcement learning.
    Communicates with Godot via the shared memory bridge.

    Observation (14, normalized to [-1, 1], learner's frame):
        [ball_x, ball_z, f1_x, f1_z, f2_x, f2_z, f3_x, f3_z, e1_x, e1_z, e2_x, e2_z, e3_x, e3_z]
    Action (6, [-1, 1]): launch direction (x, z) for the 3 friendly players.
        Player.Launch() normalizes it, so only the direction matters
        (all zeros = that player does not move).

    Args:
        opponent: object with act(raw_obs) -> 6 actions in its own frame, and an
            optional reset(). It gets opponent_view(...) of the raw obs. None = the
            opponent team stands still. Ignored by Godot in HUMAN_VS_AI.
        reward_config: RewardConfig for shaping.
        max_ep_len: turns before the episode is truncated.
        step_timeout: seconds to wait for Godot per turn; None = forever (human play).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, opponent=None, reward_config=None, max_ep_len=1000, step_timeout=120.0, bridge=None):
        super().__init__()

        self.observation_space = gym.spaces.Box(-1.0, 1.0, shape=(14,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(6,), dtype=np.float32)

        self.opponent = opponent
        self.reward_config = reward_config or RewardConfig()
        self.max_ep_len = max_ep_len
        self.step_timeout = step_timeout

        self.ep_len = 0
        self._raw_obs = None
        self.play_mode = None
        self.learner_team = None

        # Shared memory bridge (attach to the file Godot created)
        self.bridge = bridge or Shared_memory_link(create=False)

    # ---- gym API ----

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.ep_len = 0

        self.bridge.request_reset()
        self.bridge.wait_for_reset(self.step_timeout)

        self._raw_obs = self.bridge.read_observation()
        self.play_mode = self.bridge.read_play_mode()
        self.learner_team = self.bridge.read_learner_team()

        if self.opponent is not None and hasattr(self.opponent, "reset"):
            self.opponent.reset()

        info = {"play_mode": self.play_mode, "learner_team": self.learner_team}
        return normalize_obs(self._raw_obs), info

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32).reshape(6), -1.0, 1.0)
        self.bridge.write_action(action)

        if self.opponent is not None and self.play_mode == "AI_VS_AI":
            opp_action = np.clip(np.asarray(self.opponent.act(opponent_view(self._raw_obs)),
                                            dtype=np.float32).reshape(6), -1.0, 1.0)
            self.bridge.write_opponent_action(opp_action)
        else:
            self.bridge.write_opponent_action(None)

        self.bridge.request_step()
        self.bridge.wait_for_step(self.step_timeout)
        self.ep_len += 1

        prev_raw = self._raw_obs
        raw_obs = self.bridge.read_observation()
        goal_reward = self.bridge.read_reward()
        terminated = self.bridge.read_done()
        scorer = self.bridge.read_scorer()
        self._raw_obs = raw_obs

        # On a goal Godot has already moved everyone back to kickoff,
        # so the position terms would be meaningless for this turn.
        shaping = 0.0 if terminated else self._shaping(prev_raw, raw_obs)
        reward = self.reward_config.goal * goal_reward + shaping

        truncated = (not terminated) and self.ep_len >= self.max_ep_len
        info = {"goal_reward": goal_reward, "shaping": shaping, "scorer": scorer}
        return normalize_obs(raw_obs), float(reward), bool(terminated), bool(truncated), info

    # ---- reward shaping ----

    def _shaping(self, prev_raw, raw):
        cfg = self.reward_config
        p_ball, p_friendly, _ = split_obs(prev_raw)
        n_ball, n_friendly, _ = split_obs(raw)
        r = 0.0

        if cfg.ball_to_goal:
            progress = np.linalg.norm(p_ball - OPP_GOAL) - np.linalg.norm(n_ball - OPP_GOAL)
            r += cfg.ball_to_goal * progress / FIELD_HALF_Z

        if cfg.players_to_ball:
            before = np.linalg.norm(p_friendly - p_ball, axis=1)
            after = np.linalg.norm(n_friendly - n_ball, axis=1)
            r += cfg.players_to_ball * float(np.mean(before - after)) / FIELD_HALF_Z

        if cfg.defence:
            # learner attacks -z, so "goal-side" (between ball and own goal) = larger z
            if not np.any(n_friendly[:, 1] > n_ball[1]):
                r -= cfg.defence

        r -= cfg.time_penalty
        return float(r)

    def render(self):
        pass  # Godot renders

    def close(self):
        self.bridge.close()


# Helper function to create environment (used by training loop)
def bobble_league_env_helper(**kwargs):
    """Spinning Up style env_fn."""
    return Bobble_league_env(**kwargs)
