"""
Opponents for AI_VS_AI training and evaluation.

Every opponent gets RAW (world-unit) observations in ITS OWN frame
(bobble_league_env.opponent_view) and returns 6 actions in that same frame.
Godot turns them back into world directions, so any policy trained as the
learner can also play the opponent side.
"""
import random

import numpy as np
import torch

from bobble_league_env import encode_action, normalize_obs, split_obs, vector_to_action
from neural_network import Actor_critic_neural_network
from shared_memory_link import Shared_memory_link


def pfsp_weights(win_rates, power=2.0, eps=1e-6):
    """
    Docstring for pfsp_weights:

    Prioritized fictitious self-play weighting (PFSP). Given the learner's win
    rate against each pool copy, return a probability over the copies that favours
    the ones it loses to more often, using weight `(1 - winrate) ** power`
    (power 2 from AlphaStar). A copy the learner always beats (winrate 1) gets
    weight ~0; one it always loses to (winrate 0) gets weight 1.

    `eps` keeps a tiny floor so a copy is never picked with probability exactly 0
    (it can still be sampled rarely), and so an all-beaten pool does not divide by
    zero but falls back to uniform. Returns a numpy array that sums to 1; an empty
    input returns an empty array.
    """
    wr = np.asarray(win_rates, dtype=np.float64)
    if wr.size == 0:
        return np.asarray([], dtype=np.float64)
    weights = np.clip(1.0 - wr, 0.0, 1.0) ** power + eps
    total = weights.sum()
    if total <= 0:
        return np.full(wr.shape, 1.0 / wr.size)
    return weights / total


class StillOpponent:
    """Never moves."""
    name = "still"

    def act(self, raw_obs):
        return np.zeros(6, dtype=np.float32)


class RandomOpponent:
    """Random launch directions."""
    name = "random"

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def act(self, raw_obs):
        return self.rng.uniform(-1.0, 1.0, size=6).astype(np.float32)


class ChaserOpponent:
    """
    Simple scripted baseline: each player launches at the ball, aiming a bit
    behind it (seen from the goal it attacks) so contact pushes the ball goal-ward.

    Kickoff (ball within `kickoff_radius` of the centre spot) is softened, because
    three straight-line chasers scored from kickoff in one turn almost every time:
    only the `kickoff_movers` players nearest the ball move, with wider aim noise,
    and the rest stay home.
    """
    name = "chaser"

    def __init__(self, behind=0.8, noise=0.1, kickoff_radius=0.5,
                 kickoff_movers=1, kickoff_noise=0.6, seed=None):
        self.behind = behind
        self.noise = noise
        self.kickoff_radius = kickoff_radius
        self.kickoff_movers = kickoff_movers
        self.kickoff_noise = kickoff_noise
        self.rng = np.random.default_rng(seed)

    def is_kickoff(self, ball):
        return float(np.linalg.norm(ball)) < self.kickoff_radius

    def act(self, raw_obs):
        ball, friendly, _ = split_obs(raw_obs)
        kickoff = self.is_kickoff(ball)
        # attacks -z in its own frame, so "behind the ball" is +z
        target = ball + np.array([0.0, self.behind], dtype=np.float32)
        direction = target - friendly
        dist = np.linalg.norm(direction, axis=1, keepdims=True) + 1e-6
        noise = self.kickoff_noise if kickoff else self.noise
        direction = direction / dist + self.rng.normal(0.0, noise, size=direction.shape)
        if kickoff:
            movers = np.argsort(dist.ravel())[:self.kickoff_movers]
            stay = np.ones(len(friendly), dtype=bool)
            stay[movers] = False
            # v3 has no zero-power "don't move", so aim non-movers back at own
            # goal (+z); vector_to_action floors them at 0.2 power, a harmless
            # nudge away from the ball instead of a weak shot at the enemy goal.
            direction[stay] = np.array([0.0, 1.0], dtype=np.float32)
        # protocol v3: aim vectors -> angle + power (vector_to_action).
        return vector_to_action(direction)


class DefenderOpponent:
    """
    Docstring for DefenderOpponent class:

    Held-out test opponent (never enters the training pool). Each player sits
    between the ball and its OWN goal and taps the ball away: it aims at a point
    just goal-side of the ball (its own goal is +z in its frame) so a weak nudge
    clears the ball rather than carrying it forward. A low fixed power keeps it
    defensive, not an attacker in disguise. Used to measure whether the learner
    can break down a parked defence, which beating its own copies does not show.
    """
    name = "defender"

    def __init__(self, power=0.35, noise=0.05, seed=None):
        self.power = power
        self.noise = noise
        self.rng = np.random.default_rng(seed)

    def act(self, raw_obs):
        ball, friendly, _ = split_obs(raw_obs)
        # own goal is +z, so "goal-side of the ball" is a point a little beyond
        # the ball toward +z; clearing from there pushes the ball to -z (away).
        target = ball + np.array([0.0, -0.6], dtype=np.float32)
        direction = target - friendly
        dist = np.linalg.norm(direction, axis=1, keepdims=True) + 1e-6
        unit = direction / dist + self.rng.normal(0.0, self.noise, size=direction.shape)
        unit = unit / (np.linalg.norm(unit, axis=1, keepdims=True) + 1e-6)
        # fixed low power so it clears rather than shoots
        vectors = unit * self.power
        return vector_to_action(vectors)


class ReplayOpponent:
    """
    Docstring for ReplayOpponent class:

    Held-out test opponent (never enters the training pool). Replays a fixed
    recorded move every turn, cycling through the recorded frames. This stands in
    for "a replay of the winning move": a single counter the learner should not
    keep losing to once it is strong. Pass a list of 6-number actions; it repeats
    them in order and wraps around.
    """
    name = "replay"

    def __init__(self, actions=None):
        if actions is None or len(actions) == 0:
            # default: all three players shoot straight at the enemy goal, full power
            actions = [encode_action([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])]
        self.actions = [np.asarray(a, dtype=np.float32).reshape(6) for a in actions]
        self.i = 0

    def reset(self):
        self.i = 0

    def act(self, raw_obs):
        a = self.actions[self.i % len(self.actions)]
        self.i += 1
        return a.astype(np.float32)


class PolicyOpponent:
    """A frozen actor-critic network (CPU).

    `std_scale` shrinks the sampling spread at play time: with deterministic off,
    actions are sampled from the policy mean with its std multiplied by
    `std_scale`. `evaluate.py --human` uses 0.3 so one fixed counter does not
    always work, while the play stays close to the learned mean. It is ignored
    when `deterministic` is True (the mean is used).

    `abs_std` overrides the spread with an ABSOLUTE value (same for every action
    slot) instead of scaling the net's own std. Scoring runs use abs_std=0.1 (the
    small-sigma scoring in Phase 10, step 7) so the Elo number does not move with
    whatever spread the net currently carries. abs_std takes precedence over
    std_scale and is likewise ignored when deterministic is True."""

    def __init__(self, state_dict=None, deterministic=False, hidden=(256, 256),
                 name="policy", std_scale=1.0, abs_std=None, use_layernorm=False):
        self.net = Actor_critic_neural_network(obs_dim=14, act_dim=6, hidden=list(hidden),
                                               use_layernorm=use_layernorm)
        self.net.eval()
        self._hidden = tuple(hidden)
        self.deterministic = deterministic
        self.std_scale = float(std_scale)
        self.abs_std = None if abs_std is None else float(abs_std)
        self.name = name
        if state_dict is not None:
            self.load(state_dict)

    def load(self, state_dict):
        # A LayerNorm checkpoint carries an extra parametrised layer inside each
        # hidden block (Linear -> LayerNorm -> activation), so 'pi_net.1.weight'
        # exists only when LayerNorm is on. Rebuild the net to match the save's
        # architecture before loading, so a scoring/eval PolicyOpponent handles a
        # LayerNorm snapshot (Phase 11 step 5) without the caller passing a flag.
        wants_ln = "pi_net.1.weight" in state_dict
        if self.net.use_layernorm != wants_ln:
            hidden = getattr(self, "_hidden", (256, 256))
            self.net = Actor_critic_neural_network(obs_dim=14, act_dim=6,
                                                   hidden=list(hidden),
                                                   use_layernorm=wants_ln)
            self.net.eval()
        self.net.load_state_dict(state_dict)

    def act(self, raw_obs):
        obs = normalize_obs(raw_obs)
        if self.deterministic or (self.abs_std is None and self.std_scale == 1.0):
            return self.net.act(obs, deterministic=self.deterministic).numpy()
        # sample with a shrunk / fixed spread: mean + std * noise
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32)
            mu = self.net.pi_net(obs_t)
            if self.abs_std is not None:
                std = torch.full_like(mu, self.abs_std)
            else:
                std = self.net.log_std.exp() * self.std_scale
            action = mu + std * torch.randn_like(std)
        return action.numpy().astype(np.float32)


class LiveOpponent:
    """
    The CURRENT learner playing the opponent side (live vs live).

    Unlike PolicyOpponent it holds a REFERENCE to the live actor-critic, not a
    frozen copy, so its turns are on-policy and can be stored as training data.
    It SAMPLES (never the deterministic mean) in its own mirrored frame and
    records the (obs, action, value, log-prob) of its last turn so the training
    loop can push the opponent side into a buffer with the right value and logp.

    'obs' here is the normalized opponent-frame observation the net acted on, so
    it matches exactly what the loop stores for Team B.
    """
    name = "live"

    def __init__(self, ac):
        self.ac = ac
        self.last = None  # dict(obs, act, val, logp) from the most recent act()

    def act(self, raw_obs):
        obs = normalize_obs(raw_obs)
        a, v, logp = self.ac.step(obs)
        a_np = a.detach().cpu().numpy().reshape(6)
        self.last = dict(
            obs=np.asarray(obs, dtype=np.float32),
            act=a_np.astype(np.float32),
            val=float(v.detach().cpu().reshape(()).item()),
            logp=float(logp.detach().cpu().reshape(()).item()),
        )
        return a_np


class SelfPlayPool:
    """
    Self-play opponent. Keeps the last `max_size` snapshots of the learner.
    Each episode (reset) it picks one of three kinds of opponent so the learner
    does not overfit to one:
      - the scripted chaser (~`scripted_prob`, default 0.15),
      - the LIVE learner itself (~`live_prob`, default 0.30) once a live net is
        attached, so both teams can be trained on (Phase 5),
      - otherwise a past frozen snapshot (~0.55).
    Until the first snapshot exists it falls back to chaser / live only.

    Each snapshot carries a win rate (the learner's win rate against it, as the
    training loop records results). It is kept so a resumed run restores the pool
    exactly and so later phases can pick harder copies more often.
    """

    def __init__(self, max_size=10, scripted_prob=0.15, live_prob=0.30,
                 hidden=(256, 256), pfsp=False, seed=None, use_layernorm=False):
        self.max_size = max_size
        self.scripted_prob = scripted_prob
        self.live_prob = live_prob
        self.pfsp = pfsp
        self.hidden = tuple(hidden)
        self.use_layernorm = use_layernorm
        self.snapshots = []
        self.win_rates = []  # learner win rate vs each snapshot, same index as snapshots
        self.rng = random.Random(seed)
        self.scripted = ChaserOpponent(seed=seed)
        self.policy = PolicyOpponent(hidden=hidden, name="self", use_layernorm=use_layernorm)
        self.live = None  # LiveOpponent, set by attach_live() when a live net exists
        self.current = self.scripted
        self.last_index = None  # index of the frozen copy drawn this episode (PFSP)
        self._siblings = [self]  # pools sharing this snapshot list (set by make_siblings)

    @property
    def name(self):
        return self.current.name

    def attach_live(self, ac):
        """Give the pool the live actor-critic so it can play the learner side."""
        self.live = LiveOpponent(ac)
        for sib in self._siblings:
            if sib is not self:
                sib.live = LiveOpponent(ac)

    def make_siblings(self, n):
        """
        Build `n` pools that SHARE this pool's snapshot list, win rates and RNG-free
        settings, but each keep their OWN `current` opponent and policy buffer. With
        N parallel games this lets every game pick (and remember) its own opponent
        for the episode, while a snapshot added once is seen by all (shared list).
        Returns the list of n pools; element 0 is this pool.
        """
        siblings = [self]
        for i in range(1, n):
            sib = SelfPlayPool(self.max_size, self.scripted_prob, self.live_prob,
                               self.hidden, pfsp=self.pfsp, seed=None,
                               use_layernorm=self.use_layernorm)
            # share the mutable snapshot + win-rate LISTS so an add() is seen by all
            sib.snapshots = self.snapshots
            sib.win_rates = self.win_rates
            sib.live = self.live
            siblings.append(sib)
        self._siblings = siblings
        for sib in siblings:
            sib._siblings = siblings
        return siblings

    def is_live_current(self):
        """True when this game's current opponent is the live learner (store its data)."""
        return self.current is self.live and self.live is not None

    def pick_snapshot_index(self):
        """
        Docstring for pick_snapshot_index:

        Choose which frozen copy to play this episode. With `pfsp` on, copies the
        learner loses to more often are picked more often (weight `(1-winrate)^2`,
        from `pfsp_weights`); with it off, uniform. Draws from the pool's own RNG
        so a resumed run repeats the same stream.
        """
        n = len(self.snapshots)
        if self.pfsp and n > 0:
            weights = pfsp_weights(self.win_rates)
            # draw an index from the categorical weights using the pool's RNG
            r = self.rng.random()
            acc = 0.0
            for i, w in enumerate(weights):
                acc += float(w)
                if r <= acc:
                    return i
            return n - 1
        return self.rng.randrange(n)

    def record_result(self, index, learner_won, alpha=0.1):
        """
        Docstring for record_result:

        Fold one game's outcome into the win rate for snapshot `index` with an
        exponential moving average (so recent games weigh more). `learner_won` is
        1.0 for a learner win, 0.0 for a loss, 0.5 for a draw. PFSP reads these
        rates to pick harder copies more often.
        """
        if index is None or not (0 <= index < len(self.win_rates)):
            return
        old = self.win_rates[index]
        self.win_rates[index] = (1.0 - alpha) * old + alpha * float(learner_won)

    def add(self, state_dict):
        cpu_copy = {k: v.detach().cpu().clone() for k, v in state_dict.items()}
        self.snapshots.append(cpu_copy)
        self.win_rates.append(0.5)  # unknown until games are played
        if len(self.snapshots) > self.max_size:
            self.snapshots.pop(0)
            self.win_rates.pop(0)

    def reset(self):
        self.last_index = None
        r = self.rng.random()
        if r < self.scripted_prob or not self.snapshots and self.live is None:
            self.current = self.scripted
        elif self.live is not None and r < self.scripted_prob + self.live_prob:
            self.current = self.live
        elif self.snapshots:
            idx = self.pick_snapshot_index()
            self.last_index = idx
            self.policy.load(self.snapshots[idx])
            self.current = self.policy
        else:
            # no frozen copies yet: fall back to the live learner, else the chaser
            self.current = self.live if self.live is not None else self.scripted

    def act(self, raw_obs):
        return self.current.act(raw_obs)

    def state(self):
        """
        Docstring for state:

        Everything needed to rebuild the pool after a resume: the snapshot weights,
        their win rates, the size/prob settings and the RNG state. The live slot is
        NOT saved (it is just a reference to the training net, re-attached on resume).
        """
        return {
            "snapshots": [{k: v.detach().cpu().clone() for k, v in s.items()}
                          for s in self.snapshots],
            "win_rates": list(self.win_rates),
            "max_size": self.max_size,
            "scripted_prob": self.scripted_prob,
            "live_prob": self.live_prob,
            "pfsp": self.pfsp,
            "hidden": self.hidden,
            "use_layernorm": self.use_layernorm,
            "rng_state": self.rng.getstate(),
        }

    def load_state(self, state):
        """Docstring for load_state: restore the pool from a dict made by state()."""
        self.max_size = state.get("max_size", self.max_size)
        self.scripted_prob = state.get("scripted_prob", self.scripted_prob)
        self.live_prob = state.get("live_prob", self.live_prob)
        self.pfsp = state.get("pfsp", self.pfsp)
        self.use_layernorm = state.get("use_layernorm", self.use_layernorm)
        self.snapshots = [dict(s) for s in state.get("snapshots", [])]
        self.win_rates = list(state.get("win_rates", [0.5] * len(self.snapshots)))
        # rebuild the frozen-copy player if the saved architecture differs, so a
        # LayerNorm snapshot loads into a LayerNorm net (and vice versa) on resume.
        if getattr(self.policy, "net", None) is None or \
                self.policy.net.use_layernorm != self.use_layernorm:
            self.policy = PolicyOpponent(hidden=self.hidden, name="self",
                                         use_layernorm=self.use_layernorm)
        if "rng_state" in state:
            self.rng.setstate(state["rng_state"])


def load_state_dict(path, map_location="cpu"):
    """Loads a checkpoint saved by training_loop and refuses a pre-Phase-2 one.

    The action shape changed to angle+power (protocol v3) without changing any
    tensor shapes, so an old checkpoint would load silently and play garbage.
    Only a full dict carrying 'action_format' == PROTOCOL_VERSION is accepted."""
    # weights_only=False matches training_loop.load_checkpoint: a full save
    # carries numpy RNG/pool objects, and it is the user's own training output.
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise ValueError(
            f"'{path}' is a pre-Phase-2 checkpoint (no action_format tag). The "
            f"action shape is now angle+power (protocol v{Shared_memory_link.PROTOCOL_VERSION}); "
            f"old checkpoints no longer apply, train a fresh run."
        )
    found = ckpt.get("action_format")
    if found != Shared_memory_link.PROTOCOL_VERSION:
        raise ValueError(
            f"'{path}' is action_format {found!r}, expected "
            f"{Shared_memory_link.PROTOCOL_VERSION} (angle+power). Old checkpoints "
            f"no longer apply, train a fresh run."
        )
    return ckpt["model"]


def make_opponent(spec, deterministic=False):
    """
    'none' | 'still' | 'random' | 'chaser' | 'self' | path/to/checkpoint.pt
    ('self' returns a SelfPlayPool; the training loop feeds it snapshots.)
    """
    if spec in (None, "", "none"):
        return None
    if spec == "still":
        return StillOpponent()
    if spec == "random":
        return RandomOpponent()
    if spec == "chaser":
        return ChaserOpponent()
    if spec == "defender":
        return DefenderOpponent()
    if spec == "replay":
        return ReplayOpponent()
    if spec == "self":
        return SelfPlayPool()
    return PolicyOpponent(load_state_dict(spec), deterministic=deterministic, name=str(spec))


# ---- held-out test opponents (Phase 6, R36) ----
# A fixed set used ONLY to measure real strength. These NEVER enter the training
# pool: SelfPlayPool only ever plays its chaser slot, its live slot, or a frozen
# snapshot of the learner, so nothing here can be sampled during training. They
# are built on demand by make_test_opponent and read by league_eval.py.
TEST_ONLY_OPPONENTS = ("random", "chaser", "defender", "replay")


def make_test_opponent(name, seed=None):
    """
    Docstring for make_test_opponent:

    Build one held-out test opponent by name (see TEST_ONLY_OPPONENTS). These are
    anchors for strength measurement and are never added to the self-play pool.
    'chaser' here is the harder, non-softened chaser (no kickoff easing) so the
    test set is a real test, distinct from the pool's training chaser.
    """
    if name == "random":
        return RandomOpponent(seed=seed)
    if name == "chaser":
        # a harder chaser than the pool's: no kickoff softening
        return ChaserOpponent(kickoff_movers=3, kickoff_noise=0.1,
                              kickoff_radius=0.0, seed=seed)
    if name == "defender":
        return DefenderOpponent(seed=seed)
    if name == "replay":
        return ReplayOpponent()
    raise ValueError(f"unknown test opponent '{name}'; expected one of {TEST_ONLY_OPPONENTS}")
