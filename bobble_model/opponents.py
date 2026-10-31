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

from bobble_league_env import normalize_obs, split_obs, vector_to_action
from neural_network import Actor_critic_neural_network
from shared_memory_link import Shared_memory_link


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


class PolicyOpponent:
    """A frozen actor-critic network (CPU)."""

    def __init__(self, state_dict=None, deterministic=False, hidden=(256, 256), name="policy"):
        self.net = Actor_critic_neural_network(obs_dim=14, act_dim=6, hidden=list(hidden))
        self.net.eval()
        self.deterministic = deterministic
        self.name = name
        if state_dict is not None:
            self.load(state_dict)

    def load(self, state_dict):
        self.net.load_state_dict(state_dict)

    def act(self, raw_obs):
        return self.net.act(normalize_obs(raw_obs), deterministic=self.deterministic).numpy()


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
                 hidden=(256, 256), seed=None):
        self.max_size = max_size
        self.scripted_prob = scripted_prob
        self.live_prob = live_prob
        self.hidden = tuple(hidden)
        self.snapshots = []
        self.win_rates = []  # learner win rate vs each snapshot, same index as snapshots
        self.rng = random.Random(seed)
        self.scripted = ChaserOpponent(seed=seed)
        self.policy = PolicyOpponent(hidden=hidden, name="self")
        self.live = None  # LiveOpponent, set by attach_live() when a live net exists
        self.current = self.scripted
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
                               self.hidden, seed=None)
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

    def add(self, state_dict):
        cpu_copy = {k: v.detach().cpu().clone() for k, v in state_dict.items()}
        self.snapshots.append(cpu_copy)
        self.win_rates.append(0.5)  # unknown until games are played
        if len(self.snapshots) > self.max_size:
            self.snapshots.pop(0)
            self.win_rates.pop(0)

    def reset(self):
        r = self.rng.random()
        if r < self.scripted_prob or not self.snapshots and self.live is None:
            self.current = self.scripted
        elif self.live is not None and r < self.scripted_prob + self.live_prob:
            self.current = self.live
        elif self.snapshots:
            self.policy.load(self.rng.choice(self.snapshots))
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
            "hidden": self.hidden,
            "rng_state": self.rng.getstate(),
        }

    def load_state(self, state):
        """Docstring for load_state: restore the pool from a dict made by state()."""
        self.max_size = state.get("max_size", self.max_size)
        self.scripted_prob = state.get("scripted_prob", self.scripted_prob)
        self.live_prob = state.get("live_prob", self.live_prob)
        self.snapshots = [dict(s) for s in state.get("snapshots", [])]
        self.win_rates = list(state.get("win_rates", [0.5] * len(self.snapshots)))
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
    if spec == "self":
        return SelfPlayPool()
    return PolicyOpponent(load_state_dict(spec), deterministic=deterministic, name=str(spec))
