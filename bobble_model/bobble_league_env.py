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


class Vector_env:
    """
    Docstring for Vector_env class:

    Steps N Bobble League games at once over N shared-memory bridges, so one
    network call per turn serves them all. Each game keeps its own bridge, its own
    opponent, its own raw obs and its own turn counter, exactly like the single
    env. Done games reset on their own without touching the others.

    A game whose Godot process dies, or whose step does not come back in time, is
    restarted on its own: the caller is told so it can cut that game's path
    (truncated, bootstrap from the critic), the one Godot is relaunched, the
    handshake is awaited, and the other games carry on. Every restart is logged.

    Args:
        n_games: number of games.
        bridges: list of N Shared_memory_link (tests inject these); None means
            build them from the launcher's shm paths after launching.
        launcher: a Game_launcher (or stand-in with launch_one / wait_for_handshake /
            stop_all) used to start games and relaunch a crashed one. May be None
            when bridges are injected and no restart is needed.
        opponents: list of N opponents (one each, sharing one snapshot list in the
            training loop). None entries stand still.
        reward_config: RewardConfig for shaping (shared across games).
        max_ep_len: turns before an episode is truncated.
        step_timeout: seconds to wait per game per turn before calling it a hang.
        auto_launch: when True and a launcher is given, launch all games and attach
            bridges on construction.
        restart_tries: how many relaunch attempts before giving up on a game.
    """

    def __init__(self, n_games, bridges=None, launcher=None, opponents=None,
                 reward_config=None, max_ep_len=1000, step_timeout=120.0,
                 auto_launch=False, restart_tries=3):
        self.n_games = n_games
        self.launcher = launcher
        self.reward_config = reward_config or RewardConfig()
        self.max_ep_len = max_ep_len
        self.step_timeout = step_timeout
        self.restart_tries = restart_tries

        self.observation_space = gym.spaces.Box(-1.0, 1.0, shape=(14,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(6,), dtype=np.float32)

        if opponents is None:
            opponents = [None] * n_games
        if len(opponents) != n_games:
            raise ValueError(f"need {n_games} opponents, got {len(opponents)}")
        self.opponents = list(opponents)

        if auto_launch:
            if launcher is None:
                raise ValueError("auto_launch needs a launcher")
            launcher.launch_all()
            bridges = [launcher.wait_for_handshake(i) for i in range(n_games)]

        if bridges is None:
            raise ValueError("Vector_env needs bridges or auto_launch with a launcher")
        if len(bridges) != n_games:
            raise ValueError(f"need {n_games} bridges, got {len(bridges)}")
        self.bridges = list(bridges)

        self.ep_len = [0] * n_games
        self._raw_obs = [None] * n_games
        self.play_mode = [None] * n_games
        self.learner_team = [None] * n_games
        self.restart_counts = [0] * n_games

    # ---- vector api ----

    def reset(self):
        """
        Docstring for reset:

        Reset every game. Returns a stacked (N, 14) normalized obs and a list of
        per-game info dicts.
        """
        obs = np.zeros((self.n_games, 14), dtype=np.float32)
        infos = []
        for i in range(self.n_games):
            obs[i], info = self._reset_one(i)
            infos.append(info)
        return obs, infos

    def _reset_one(self, i):
        """Docstring for _reset_one: reset a single game, like the single env does."""
        self.ep_len[i] = 0
        self.bridges[i].request_reset()
        self.bridges[i].wait_for_reset(self.step_timeout)
        self._raw_obs[i] = self.bridges[i].read_observation()
        self.play_mode[i] = self.bridges[i].read_play_mode()
        self.learner_team[i] = self.bridges[i].read_learner_team()
        opp = self.opponents[i]
        if opp is not None and hasattr(opp, "reset"):
            opp.reset()
        info = {"play_mode": self.play_mode[i], "learner_team": self.learner_team[i]}
        return normalize_obs(self._raw_obs[i]), info

    def step(self, actions):
        """
        Docstring for step:

        Play one turn on all N games with a batched (N, 6) action array. Writes
        every learner action and opponent action, sets every step flag, then waits
        for each game in turn. A game that hangs or whose process died is restarted
        and reported with info['restarted'] = True so the caller can cut its path
        (truncated) and reset it.

        Returns stacked (obs, rewards, terminated, truncated, infos), each length N.
        """
        actions = np.asarray(actions, dtype=np.float32).reshape(self.n_games, 6)

        # 1) write all actions and set all step flags, so the games run in parallel
        for i in range(self.n_games):
            self._write_turn(i, actions[i])

        # 2) collect each game's result, restarting any that hangs or has died
        obs = np.zeros((self.n_games, 14), dtype=np.float32)
        rewards = np.zeros(self.n_games, dtype=np.float32)
        terminated = np.zeros(self.n_games, dtype=bool)
        truncated = np.zeros(self.n_games, dtype=bool)
        infos = []
        for i in range(self.n_games):
            o, r, term, trunc, info = self._collect_turn(i)
            obs[i], rewards[i], terminated[i], truncated[i] = o, r, term, trunc
            infos.append(info)
        return obs, rewards, terminated, truncated, infos

    def _write_turn(self, i, action):
        """Docstring for _write_turn: write one game's action + opponent action, set its step flag."""
        try:
            action = np.clip(action, -1.0, 1.0)
            self.bridges[i].write_action(action)
            opp = self.opponents[i]
            if opp is not None and self.play_mode[i] == "AI_VS_AI":
                opp_action = np.clip(np.asarray(opp.act(opponent_view(self._raw_obs[i])),
                                                dtype=np.float32).reshape(6), -1.0, 1.0)
                self.bridges[i].write_opponent_action(opp_action)
            else:
                self.bridges[i].write_opponent_action(None)
            self.bridges[i].request_step()
        except (ValueError, OSError):
            # a dead process can leave the mmap unwritable; the collect step restarts it
            pass

    def _collect_turn(self, i):
        """
        Docstring for _collect_turn:

        Wait for one game's turn and read its result. On a timeout (or a dead
        process) restart that one game and return a truncated turn so the caller
        bootstraps the path from the critic.
        """
        try:
            self.bridges[i].wait_for_step(self.step_timeout)
        except TimeoutError:
            return self._restart_and_truncate(i, reason="step timed out")

        self.ep_len[i] += 1
        prev_raw = self._raw_obs[i]
        raw_obs = self.bridges[i].read_observation()
        goal_reward = self.bridges[i].read_reward()
        terminated = self.bridges[i].read_done()
        scorer = self.bridges[i].read_scorer()
        self._raw_obs[i] = raw_obs

        shaping = 0.0 if terminated else self._shaping(prev_raw, raw_obs)
        reward = self.reward_config.goal * goal_reward + shaping
        truncated = (not terminated) and self.ep_len[i] >= self.max_ep_len
        info = {"goal_reward": goal_reward, "shaping": shaping, "scorer": scorer,
                "restarted": False}

        if terminated or truncated:
            obs, _ = self._reset_one(i)
            return obs, float(reward), bool(terminated), bool(truncated), info
        return normalize_obs(raw_obs), float(reward), False, False, info

    def _restart_and_truncate(self, i, reason):
        """
        Docstring for _restart_and_truncate:

        One game died or hung: cut its path (truncated), relaunch only that Godot,
        wait for the handshake, reset it, and carry on. The returned obs is the
        fresh kickoff; terminated=False and truncated=True tell the loop to
        bootstrap the value of the cut path from the critic.
        """
        self.restart_counts[i] += 1
        print(f"Vector_env: game {i} {reason}; restart #{self.restart_counts[i]}", flush=True)

        if self.launcher is None:
            raise RuntimeError(f"game {i} {reason} and no launcher to restart it")

        last_err = None
        for attempt in range(self.restart_tries):
            try:
                try:
                    self.bridges[i].close()
                except Exception:
                    pass
                self.launcher.stop_one(i)
                self.launcher.launch_one(i)
                self.bridges[i] = self.launcher.wait_for_handshake(i, timeout=self.step_timeout)
                obs, _ = self._reset_one(i)
                info = {"goal_reward": 0.0, "shaping": 0.0, "scorer": 0, "restarted": True}
                # truncated so the training loop bootstraps the cut path from the critic
                return obs, 0.0, False, True, info
            except (TimeoutError, RuntimeError, OSError) as e:
                last_err = e
                print(f"Vector_env: game {i} relaunch attempt {attempt + 1} failed: {e}", flush=True)
        raise RuntimeError(f"game {i} could not be restarted after {self.restart_tries} tries: {last_err}")

    # ---- reward shaping (same as the single env) ----

    def _shaping(self, prev_raw, raw):
        return Bobble_league_env._shaping(self, prev_raw, raw)

    def close(self):
        """Docstring for close: unmap every bridge and stop the games we launched."""
        for b in self.bridges:
            try:
                b.close()
            except Exception:
                pass
        if self.launcher is not None:
            self.launcher.stop_all()
