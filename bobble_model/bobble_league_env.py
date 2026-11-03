from dataclasses import dataclass
import time

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
    """
    Docstring for normalize_obs:

    World units -> roughly [-1, 1].
    """
    return np.clip(np.asarray(raw_obs, dtype=np.float32) / OBS_SCALE, -1.0, 1.0)


def split_obs(raw_obs):
    """
    Docstring for split_obs:

    Returns (ball[2], friendly[3,2], enemy[3,2]).
    """
    o = np.asarray(raw_obs, dtype=np.float32).reshape(7, 2)
    return o[0], o[1:4], o[4:7]


# ---- action shape (angle + power), protocol v3 ----
# each player gets 2 numbers: aim angle a and power p, both in [-1, 1], in the
# team's mirrored frame (learner always attacks -z, own goal at +z).
#   angle: theta = a * pi, measured from -z (the enemy goal). the -1/+1 seam
#          therefore points straight back at own goal (+z), the move a player
#          should need least.
#   power: 0.2 + 0.8 * (p + 1) / 2, so a shot is 20% to 100% strength.
# godot decodes the same way (ApplyTeamAction + Player.Launch(powerScale)) and
# applies FrameZ to z, so both teams launch in the same world direction.
MIN_POWER = 0.2
ACTION_PAIRS = 3  # angle + power per player, 3 players


def wrap_angle(a):
    """
    Docstring for wrap_angle:

    Wrap an angle number into [-1, 1) with a = (a + 1) % 2 - 1.

    The angle axis is circular: a and a+2 both mean the same aim (theta differs
    by 2*pi). Clipping a sample past the edge to +-1 instead pins it at the seam,
    which points straight back at own goal - the worst move - so a policy whose
    mean sits near an edge wastes every out-of-range sample on an own-goal shot.
    Wrapping keeps the real aim: 1.1 -> -0.9 (just past the enemy-goal side comes
    back round the near side), -1.1 -> 0.9, and large multiples fold in too. Only
    the ANGLE slots wrap; power still clips (it is a magnitude, not circular).

    Note a == +1 maps to -1, but both ends of the seam are the same direction
    (own goal), so that is the same aim, not a jump.
    """
    a = np.asarray(a, dtype=np.float32)
    return ((a + 1.0) % 2.0 - 1.0).astype(np.float32)


def wrap_action(action):
    """
    Docstring for wrap_action:

    Wrap the angle slots (0, 2, 4) of a 6-number action and clip the power
    slots (1, 3, 5) to [-1, 1]. Returns a new array; the input is not changed.

    This is the single clip/wrap applied to a decoded action before it reaches
    Godot. PPO stores the PRE-wrap sampled action (its log-probs are under the
    unwrapped Gaussian), so wrapping here only changes what is played, never what
    is learned from.
    """
    action = np.asarray(action, dtype=np.float32).reshape(-1).copy()
    action[0::2] = wrap_angle(action[0::2])              # angle slots wrap
    action[1::2] = np.clip(action[1::2], -1.0, 1.0)      # power slots clip
    return action


def decode_power(p):
    """
    Docstring for decode_power:

    Power number p in [-1, 1] -> scale in [0.2, 1.0].
    """
    p = np.clip(np.asarray(p, dtype=np.float32), -1.0, 1.0)
    return MIN_POWER + (1.0 - MIN_POWER) * (p + 1.0) / 2.0


def decode_angle(a):
    """
    Docstring for decode_angle:

    Angle number a -> (x, z) unit direction in the mirrored frame.

    a is WRAPPED into [-1, 1) (the angle axis is circular), then theta = a * pi
    from -z, so a=0 aims at the enemy goal (-z) and the seam (a=+-1) aims straight
    back at own goal (+z). Matches Godot's Atan2(x, z). Wrapping (not clipping)
    keeps an out-of-range sample's real aim instead of pinning it at own goal.

    """
    a = wrap_angle(a)
    theta = a * np.pi
    x = np.sin(theta)
    z = -np.cos(theta)
    return np.stack([x, z], axis=-1).astype(np.float32)


def decode_action(action):
    """
    Docstring for decode_action:

    6-number action [a1, p1, a2, p2, a3, p3] -> (3, 2) launch vectors.

    Each player's vector is its unit aim direction scaled by its power, in the
    team's mirrored frame. Length is the power, so Player.Launch can scale by it.

    """
    action = np.asarray(action, dtype=np.float32).reshape(ACTION_PAIRS, 2)
    angles = action[:, 0]
    powers = action[:, 1]
    dirs = decode_angle(angles)                 # (3, 2) unit
    scale = decode_power(powers)[:, None]       # (3, 1)
    return (dirs * scale).astype(np.float32)


def encode_action(angles, powers):
    """
    Docstring for encode_action:

    Inverse helper for scripted opponents: pack angle/power numbers into the
    6-number wire action [a1, p1, a2, p2, a3, p3].
    """
    angles = np.asarray(angles, dtype=np.float32).reshape(ACTION_PAIRS)
    powers = np.asarray(powers, dtype=np.float32).reshape(ACTION_PAIRS)
    out = np.empty(6, dtype=np.float32)
    out[0::2] = angles
    out[1::2] = powers
    return out


def vector_to_action(vectors):
    """
    Docstring for vector_to_action:

    (3, 2) mirrored-frame launch vectors -> 6-number wire action.

    Used by the chaser and human input, which think in (x, z): the direction
    sets the angle (a = atan2(x, -z) / pi, seam at own goal) and the vector
    length sets the power (clamped into [0.2, 1], then back to [-1, 1]).
    """
    vectors = np.asarray(vectors, dtype=np.float32).reshape(ACTION_PAIRS, 2)
    x = vectors[:, 0]
    z = vectors[:, 1]
    theta = np.arctan2(x, -z)                   # inverse of decode_angle
    angles = (theta / np.pi).astype(np.float32)
    length = np.linalg.norm(vectors, axis=1)
    scale = np.clip(length, MIN_POWER, 1.0)
    powers = (2.0 * (scale - MIN_POWER) / (1.0 - MIN_POWER) - 1.0).astype(np.float32)
    return encode_action(angles, powers)


def opponent_view(raw_obs):
    """
    Docstring for opponent_view:

    Learner-frame obs -> the same moment seen from the opponent's frame.
    The two frames differ by a z flip, and friendly/enemy swap.

    """
    ball, friendly, enemy = split_obs(raw_obs)
    flip = np.array([1.0, -1.0], dtype=np.float32)
    return np.concatenate([ball * flip, (enemy * flip).ravel(), (friendly * flip).ravel()])


@dataclass
class RewardConfig:
    """
    Docstring for RewardConfig:

    Per-turn shaping added on top of Godot's goal reward (+1 / -1).

    Phase 12, step 3: the position terms are now EXACT potential-based shaping.
    Each weight scales a term of a state potential phi(s) (see `potential` below),
    and the per-turn shaping is F = gamma * phi(s') - phi(s), with phi = 0 at a
    goal. In this form the shaping provably leaves the best policy unchanged (Ng,
    Harada & Russell 1999), so it can nudge the AI toward the ball and goal early
    without ever teaching "push the ball" instead of "score".

    - ball_to_goal: potential rises as the ball nears the opponent goal.
    - players_to_ball: potential rises as the friendly players close on the ball.
    - defence: potential term, 0 when a friendly is goal-side of the ball and
      -1 (times the weight) when none is. Turned into a potential so it too is
      policy-invariant (set to 0 to drop it).
    - time_penalty: a flat per-turn cost. This is NOT a potential (it is a real
      cost that should change the policy toward faster goals), so it is added
      outside the telescoping shaping, unchanged.

    Distances are divided by FIELD_HALF_Z. Set a weight to 0 to switch that term
    off.

    """
    goal: float = 1.0               # scales Godot's +-1
    ball_to_goal: float = 0.5       # potential: ball nearer the opponent goal
    players_to_ball: float = 0.05   # potential: friendly players nearer the ball (mean)
    defence: float = 0.02           # potential: friendly goal-side of the ball
    time_penalty: float = 0.0       # flat cost per turn (not a potential)

    def scaled(self, frac):
        """
        Docstring for scaled:

        Return a copy with the per-turn SHAPING weights multiplied by `frac`,
        leaving `goal` untouched (Phase 6 shaping fade, R39). frac=1 is the full
        shaping, frac=0 turns every shaping term off so only the goal reward is
        left. The goal reward never fades: the point is to end up playing for
        goals, not "ball moved forward".
        """
        f = float(frac)
        return RewardConfig(
            goal=self.goal,
            ball_to_goal=self.ball_to_goal * f,
            players_to_ball=self.players_to_ball * f,
            defence=self.defence * f,
            time_penalty=self.time_penalty * f,
        )


def potential(raw_obs, cfg):
    """
    Docstring for potential:

    State potential phi(s) for the potential-based shaping (Phase 12, step 3).

    phi is built from the learner-frame raw obs and the RewardConfig weights, so
    the per-turn shaping F = gamma*phi(s') - phi(s) telescopes exactly. Each term
    is HIGHER when the state is better for the learner, so a positive F means the
    state improved:

      - ball_to_goal: -w * d(ball, OPP_GOAL)/FIELD_HALF_Z  (nearer the enemy goal
        is a smaller distance, so a less-negative, higher potential).
      - players_to_ball: -w * mean(d(player, ball))/FIELD_HALF_Z.
      - defence: 0 when at least one friendly is goal-side of the ball (z larger
        than the ball, since the learner attacks -z and defends +z), else -w.

    time_penalty is NOT in phi (it is a real per-turn cost, added outside the
    telescoping shaping). Takes a raw (un-normalized) obs, same frame as _shaping.

    """
    ball, friendly, _ = split_obs(raw_obs)
    phi = 0.0
    if cfg.ball_to_goal:
        phi += cfg.ball_to_goal * (-np.linalg.norm(ball - OPP_GOAL) / FIELD_HALF_Z)
    if cfg.players_to_ball:
        phi += cfg.players_to_ball * (-float(np.mean(np.linalg.norm(friendly - ball, axis=1))) / FIELD_HALF_Z)
    if cfg.defence:
        goal_side = np.any(friendly[:, 1] > ball[1])
        phi += 0.0 if goal_side else -cfg.defence
    return float(phi)


def legacy_shaping(prev_raw, raw, cfg):
    """
    Docstring for legacy_shaping:

    The Phase 11 (pre-Phase-12) shaping, kept as the DEFAULT when
    `--potential-shaping` is off (Phase 12 evaluated the exact potential form and
    it did not beat Phase 11 in T5, so it is no longer the default).

    This is NOT exact potential-based shaping: it is the un-discounted distance
    deltas `phi(s') - phi(s)` (no gamma), a flat per-state defence penalty, and
    the flat time_penalty. It takes raw (un-normalized) obs, the learner's frame.

    """
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


def shaping_fade_fraction(epoch, total_epochs, fade_fraction=0.5):
    """
    Docstring for shaping_fade_fraction:

    The shaping multiplier for a given epoch (Phase 6, R39). It falls linearly
    from 1.0 at epoch 0 to 0.0 at the halfway epoch (`fade_fraction` of the run),
    then stays at 0 for the rest of training. So the shaping is on early, when the
    AI needs the nudge toward the ball and goal, and gone by the midpoint, when it
    should be playing for goals alone.

    epoch is 0-based. With total_epochs=100 and the default fade_fraction=0.5 the
    multiplier is 1.0 at epoch 0, 0.5 at epoch 25, and 0.0 at epoch 50 and after.
    A fade_fraction of 0 (or total_epochs <= 1) means no shaping from the start.
    """
    fade_end = max(0.0, float(fade_fraction) * (int(total_epochs) - 1))
    if fade_end <= 0.0:
        return 0.0
    frac = 1.0 - (float(epoch) / fade_end)
    return max(0.0, min(1.0, frac))


class Bobble_league_env(gym.Env):
    """
    Docstring for Bobble_league_env:

    Gymnasium environment for Bobble League reinforcement learning.
    Communicates with Godot via the shared memory bridge.

    Observation (14, normalized to [-1, 1], learner's frame):
        [ball_x, ball_z, f1_x, f1_z, f2_x, f2_z, f3_x, f3_z, e1_x, e1_z, e2_x, e2_z, e3_x, e3_z]
    Action (6, [-1, 1]): angle + power per player [a1, p1, a2, p2, a3, p3] in the
        team's mirrored frame. Godot decodes it (decode_action above): angle
        a -> a*pi from -z (seam at own goal), power p -> 0.2 + 0.8*(p+1)/2.
        The env passes the 6 numbers through; Godot does the decode so both teams
        launch in the same world direction.

    Args:
        opponent: object with act(raw_obs) -> 6 actions in its own frame, and an
            optional reset(). It gets opponent_view(...) of the raw obs. None = the
            opponent team stands still. Ignored by Godot in HUMAN_VS_AI.
        reward_config: RewardConfig for shaping.
        max_ep_len: turns before the episode is truncated.
        step_timeout: seconds to wait for Godot per turn; None = forever (human play).

    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, opponent=None, reward_config=None, max_ep_len=1000, step_timeout=120.0,
                 bridge=None, random_start_prob=0.0, gamma=0.99, potential_shaping=False):
        super().__init__()

        self.observation_space = gym.spaces.Box(-1.0, 1.0, shape=(14,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(6,), dtype=np.float32)

        self.opponent = opponent
        self.reward_config = reward_config or RewardConfig()
        self.max_ep_len = max_ep_len
        self.step_timeout = step_timeout
        # discount used by the potential-based shaping F = gamma*phi(s') - phi(s)
        # (Phase 12, step 3); must match the training loop's gamma so the shaping
        # is policy-invariant (Ng, Harada & Russell 1999).
        self.gamma = float(gamma)
        # Phase 12, step 1: fraction of env resets that start from a random legal
        # layout instead of kickoff, so the AI also sees odd boards (defending near
        # its own goal, a scramble). 0.0 keeps every reset at kickoff.
        self.random_start_prob = float(random_start_prob)
        # Phase 12, step 3: when True, use exact potential-based shaping
        # (gamma*phi(s')-phi(s), phi=0 at a goal, OOB skip). Default False keeps
        # the Phase 11 shaping, so a plain launch reproduces Phase 11 (the
        # potential form was a T5 miss). See legacy_shaping / _shaping.
        self.potential_shaping = bool(potential_shaping)

        self.ep_len = 0
        self._raw_obs = None
        self.play_mode = None
        self.learner_team = None
        # True on the turn right after an out-of-bounds / random reset, where the
        # positions jump, so shaping (a difference between turns) is skipped there.
        self._skip_shaping_next = False

        # Shared memory bridge (attach to the file Godot created)
        self.bridge = bridge or Shared_memory_link(create=False)

    # ---- gym API ----

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.ep_len = 0

        # Phase 12, step 1: scatter this reset to a random legal layout with
        # probability random_start_prob (Godot does the placement when byte 91 is
        # set), otherwise a normal kickoff.
        random_start = np.random.random() < self.random_start_prob
        self.bridge.request_reset(random_start=random_start)
        self.bridge.wait_for_reset(self.step_timeout)

        self._raw_obs = self.bridge.read_observation()
        self.play_mode = self.bridge.read_play_mode()
        self.learner_team = self.bridge.read_learner_team()

        if self.opponent is not None and hasattr(self.opponent, "reset"):
            self.opponent.reset()

        info = {"play_mode": self.play_mode, "learner_team": self.learner_team}
        return normalize_obs(self._raw_obs), info

    def step(self, action):
        # wrap the angle slots and clip the power slots before Godot sees them.
        # the training loop has already stored the raw sampled action (its
        # log-probs are under the unwrapped Gaussian), so this changes only what
        # is played, not what is learned from (Phase 10, step 3).
        action = wrap_action(np.asarray(action, dtype=np.float32).reshape(6))
        self.bridge.write_action(action)

        if self.opponent is not None and self.play_mode == "AI_VS_AI":
            opp_action = wrap_action(np.asarray(self.opponent.act(opponent_view(self._raw_obs)),
                                                dtype=np.float32).reshape(6))
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
        turn_seconds = self.bridge.read_turn_seconds()
        oob_reset = self.bridge.read_oob_reset()
        self._raw_obs = raw_obs

        # potential-based shaping: phi is 0 at a goal, and the turn right after an
        # out-of-bounds kickoff is skipped (the positions jumped, so a difference
        # of potentials is meaningless there).
        shaping = self._shaping(prev_raw, raw_obs, terminated=terminated, skip=oob_reset)
        reward = self.reward_config.goal * goal_reward + shaping

        truncated = (not terminated) and self.ep_len >= self.max_ep_len
        info = {"goal_reward": goal_reward, "shaping": shaping, "scorer": scorer,
                "turn_seconds": float(turn_seconds), "oob_reset": bool(oob_reset)}
        return normalize_obs(raw_obs), float(reward), bool(terminated), bool(truncated), info

    # ---- reward shaping ----

    def _shaping(self, prev_raw, raw, terminated=False, skip=False):
        """
        Docstring for _shaping:

        Per-turn shaping. Two forms, chosen by `self.potential_shaping`.

        OFF (default, the Phase 11 form): the un-discounted distance deltas
        `phi(s') - phi(s)` with a flat defence penalty, minus the flat
        time_penalty. `terminated`/`skip` are ignored. A plain training launch
        uses this, so it reproduces the Phase 11 recipe (Phase 12 evaluated the
        potential form as a T5 miss, so it is no longer the default).

        ON (`--potential-shaping`, Phase 12, step 3): exact potential-based
        shaping F = gamma * phi(s') - phi(s), with phi = 0 at a goal
        (`terminated`), plus the flat time_penalty (a real cost, outside the
        telescoping sum). `skip` is True on the turn right after an out-of-bounds
        kickoff, where the positions jump, so no shaping is applied there.

        """
        cfg = self.reward_config
        if not getattr(self, 'potential_shaping', False):
            return legacy_shaping(prev_raw, raw, cfg)
        if skip:
            return 0.0
        phi_prev = potential(prev_raw, cfg)
        # phi is 0 at a goal; otherwise it is the next state's potential.
        phi_next = 0.0 if terminated else potential(raw, cfg)
        r = self.gamma * phi_next - phi_prev
        r -= cfg.time_penalty
        return float(r)

    def render(self):
        pass  # Godot renders

    def close(self):
        self.bridge.close()


# Helper function to create environment (used by training loop)
def bobble_league_env_helper(**kwargs):
    """
    Docstring for bobble_league_env_helper:

    Spinning Up style env_fn.
    """
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
                 auto_launch=False, restart_tries=3, random_start_prob=0.0, gamma=0.99,
                 potential_shaping=False):
        self.n_games = n_games
        self.launcher = launcher
        self.reward_config = reward_config or RewardConfig()
        self.max_ep_len = max_ep_len
        self.step_timeout = step_timeout
        self.restart_tries = restart_tries
        # Phase 12: potential-based shaping discount and the per-reset random-start
        # probability (same meaning as the single env).
        self.gamma = float(gamma)
        self.random_start_prob = float(random_start_prob)
        # Phase 12, step 3: exact potential-based shaping when True; default False
        # keeps the Phase 11 shaping (see legacy_shaping / _shaping).
        self.potential_shaping = bool(potential_shaping)

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
        random_start = np.random.random() < self.random_start_prob
        self.bridges[i].request_reset(random_start=random_start)
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
        every learner action and opponent action and sets every step flag, so the
        games run in parallel, then collects each game AS SOON AS IT IS READY
        rather than waiting on them in a fixed index order: a fast game no longer
        sits idle behind a slow one. A game that hangs or whose process died is
        restarted and reported with info['restarted'] = True so the caller can cut
        its path (truncated) and reset it.

        Each game still produces exactly one result per call, written into its own
        slot i, so a game's path stays in order and no turn is lost or reordered.

        Returns stacked (obs, rewards, terminated, truncated, infos), each length N.
        """
        actions = np.asarray(actions, dtype=np.float32).reshape(self.n_games, 6)

        # 1) write all actions and set all step flags, so the games run in parallel
        for i in range(self.n_games):
            self._write_turn(i, actions[i])

        # 2) collect each game the moment its turn is done, in whatever order they
        # finish. results go into the game's own slot, so slot order is preserved.
        obs = np.zeros((self.n_games, 14), dtype=np.float32)
        rewards = np.zeros(self.n_games, dtype=np.float32)
        terminated = np.zeros(self.n_games, dtype=bool)
        truncated = np.zeros(self.n_games, dtype=bool)
        infos = [None] * self.n_games

        pending = set(range(self.n_games))
        deadlines = {i: time.time() + self.step_timeout for i in pending}
        while pending:
            progressed = False
            for i in list(pending):
                ready = False
                try:
                    ready = self.bridges[i].step_ready()
                except (ValueError, OSError):
                    # a dead process can leave the mmap unreadable; treat as a hang
                    # so the deadline below restarts it
                    ready = False
                if ready:
                    o, r, term, trunc, info = self._read_turn(i)
                    obs[i], rewards[i], terminated[i], truncated[i] = o, r, term, trunc
                    infos[i] = info
                    pending.discard(i)
                    progressed = True
                elif time.time() > deadlines[i]:
                    o, r, term, trunc, info = self._restart_and_truncate(i, reason="step timed out")
                    obs[i], rewards[i], terminated[i], truncated[i] = o, r, term, trunc
                    infos[i] = info
                    pending.discard(i)
                    progressed = True
            if pending and not progressed:
                # nothing was ready this sweep: sleep a beat so we are not spinning
                time.sleep(0.001)
        return obs, rewards, terminated, truncated, infos

    def _write_turn(self, i, action):
        """Docstring for _write_turn: write one game's action + opponent action, set its step flag.

        Angle slots are wrapped and power slots clipped before Godot sees them
        (Phase 10, step 3). The training loop stored the raw sampled action, so
        wrapping here changes only what is played, not what is learned from."""
        try:
            action = wrap_action(action)
            self.bridges[i].write_action(action)
            opp = self.opponents[i]
            if opp is not None and self.play_mode[i] == "AI_VS_AI":
                opp_action = wrap_action(np.asarray(opp.act(opponent_view(self._raw_obs[i])),
                                                    dtype=np.float32).reshape(6))
                self.bridges[i].write_opponent_action(opp_action)
            else:
                self.bridges[i].write_opponent_action(None)
            self.bridges[i].request_step()
        except (ValueError, OSError):
            # a dead process can leave the mmap unwritable; the collect step restarts it
            pass

    def _read_turn(self, i):
        """
        Docstring for _read_turn:

        Read one game's finished turn and process it. The caller (step) has already
        checked the game is ready via step_ready(), so this never waits; it just
        reads the result, computes the reward and shaping, resets the game on a
        goal, and returns the per-game (obs, reward, terminated, truncated, info).

        When the game's opponent is the live learner (live vs live), also compute
        Team B's reward from ITS mirrored frame and expose the opponent-frame
        transition in info, so the loop can store the opponent side as training
        data. info always carries 'turn_seconds' (how long the turn took to settle).
        """
        self.ep_len[i] += 1
        prev_raw = self._raw_obs[i]
        raw_obs = self.bridges[i].read_observation()
        goal_reward = self.bridges[i].read_reward()
        terminated = self.bridges[i].read_done()
        scorer = self.bridges[i].read_scorer()
        turn_seconds = self.bridges[i].read_turn_seconds()
        oob_reset = self.bridges[i].read_oob_reset()
        self._raw_obs[i] = raw_obs

        # potential-based shaping: phi is 0 at a goal, and a turn that hit an
        # out-of-bounds kickoff is skipped (the positions jumped).
        shaping = self._shaping(prev_raw, raw_obs, terminated=terminated, skip=oob_reset)
        reward = self.reward_config.goal * goal_reward + shaping
        truncated = (not terminated) and self.ep_len[i] >= self.max_ep_len
        info = {"goal_reward": goal_reward, "shaping": shaping, "scorer": scorer,
                "turn_seconds": float(turn_seconds), "restarted": False,
                "oob_reset": bool(oob_reset)}

        # live vs live: store Team B too, from its mirrored view. Team B's goal
        # reward is the negated learner reward, its shaping is computed in the
        # opponent frame (opponent_view flips z and swaps teams).
        self._maybe_add_opponent_turn(i, info, prev_raw, raw_obs, goal_reward,
                                      terminated, oob_reset)

        if terminated or truncated:
            self._maybe_record_pool_result(i, scorer if terminated else 0)
            obs, _ = self._reset_one(i)
            return obs, float(reward), bool(terminated), bool(truncated), info
        return normalize_obs(raw_obs), float(reward), False, False, info

    def _maybe_record_pool_result(self, i, scorer):
        """
        Docstring for _maybe_record_pool_result:

        When game i's opponent is a self-play pool playing a FROZEN copy, fold this
        game's outcome into that copy's win rate (for PFSP) before the game resets
        and the pool draws a new copy. scorer is 1 = learner won, 2 = opponent won,
        0 = draw/timeout. Only frozen-copy games count: the live and chaser slots
        have no snapshot index.
        """
        opp = self.opponents[i]
        idx = getattr(opp, "last_index", None)
        if idx is None or getattr(opp, "record_result", None) is None:
            return
        learner_won = 1.0 if scorer == 1 else 0.0 if scorer == 2 else 0.5
        opp.record_result(idx, learner_won)

    def _is_live_opponent(self, i):
        """
        Docstring for _is_live_opponent:

        True when game i's opponent is currently the live learner (SelfPlayPool live slot).
        """
        opp = self.opponents[i]
        return opp is not None and getattr(opp, "is_live_current", None) is not None and opp.is_live_current()

    def _maybe_add_opponent_turn(self, i, info, prev_raw, raw_obs, goal_reward, terminated, oob_reset=False):
        """
        Docstring for _maybe_add_opponent_turn:

        Fill info['opponent'] with Team B's transition for a live-vs-live game, so
        the training loop can push it into the opponent-side buffer. The acting
        obs/action/value/logp come from the LiveOpponent (recorded at act time,
        already in the opponent frame); here we add the reward Team B earned.

        """
        if not self._is_live_opponent(i):
            return
        opp = self.opponents[i]
        live = getattr(opp, "live", None)
        last = getattr(live, "last", None) if live is not None else None
        if last is None:
            return
        opp_goal_reward = -goal_reward  # a learner goal is Team B's loss, and vice versa
        # Team B's shaping is the same potential-based form in its mirrored frame,
        # with the same terminal (phi=0) and out-of-bounds skip handling.
        opp_shaping = self._shaping(opponent_view(prev_raw), opponent_view(raw_obs),
                                    terminated=terminated, skip=oob_reset)
        opp_reward = self.reward_config.goal * opp_goal_reward + opp_shaping
        info["opponent"] = {
            "obs": last["obs"],
            "act": last["act"],
            "val": last["val"],
            "logp": last["logp"],
            "reward": float(opp_reward),
        }

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
                info = {"goal_reward": 0.0, "shaping": 0.0, "scorer": 0,
                        "turn_seconds": 0.0, "restarted": True}
                # truncated so the training loop bootstraps the cut path from the critic
                return obs, 0.0, False, True, info
            except (TimeoutError, RuntimeError, OSError) as e:
                last_err = e
                print(f"Vector_env: game {i} relaunch attempt {attempt + 1} failed: {e}", flush=True)
        raise RuntimeError(f"game {i} could not be restarted after {self.restart_tries} tries: {last_err}")

    # ---- reward shaping (same as the single env) ----

    def _shaping(self, prev_raw, raw, terminated=False, skip=False):
        return Bobble_league_env._shaping(self, prev_raw, raw,
                                          terminated=terminated, skip=skip)

    def close(self):
        """Docstring for close: unmap every bridge and stop the games we launched."""
        for b in self.bridges:
            try:
                b.close()
            except Exception:
                pass
        if self.launcher is not None:
            self.launcher.stop_all()
