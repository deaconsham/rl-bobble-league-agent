"""
Opponents for AI_VS_AI training and evaluation.

Every opponent gets RAW (world-unit) observations in ITS OWN frame
(bobble_league_env.opponent_view) and returns 6 actions in that same frame.
Godot turns them back into world directions, so any policy trained as the
learner can also play the opponent side.
"""
import copy
import random

import numpy as np
import torch

from bobble_league_env import normalize_obs, split_obs
from neural_network import Actor_critic_neural_network


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
            direction[stay] = 0.0
        return np.clip(direction, -1.0, 1.0).astype(np.float32).ravel()


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


class SelfPlayPool:
    """
    Self-play opponent. Keeps the last `max_size` snapshots of the learner.
    Each episode (reset) it picks a past snapshot, or with `scripted_prob`
    the scripted chaser, so the learner does not overfit to one opponent.
    Until the first snapshot exists it always uses the chaser.

    Each snapshot carries a win rate (the learner's win rate against it, as the
    training loop records results). It is kept so a resumed run restores the pool
    exactly and so later phases can pick harder copies more often.
    """

    def __init__(self, max_size=10, scripted_prob=0.2, hidden=(256, 256), seed=None):
        self.max_size = max_size
        self.scripted_prob = scripted_prob
        self.hidden = tuple(hidden)
        self.snapshots = []
        self.win_rates = []  # learner win rate vs each snapshot, same index as snapshots
        self.rng = random.Random(seed)
        self.scripted = ChaserOpponent(seed=seed)
        self.policy = PolicyOpponent(hidden=hidden, name="self")
        self.current = self.scripted

    @property
    def name(self):
        return self.current.name

    def add(self, state_dict):
        cpu_copy = {k: v.detach().cpu().clone() for k, v in state_dict.items()}
        self.snapshots.append(cpu_copy)
        self.win_rates.append(0.5)  # unknown until games are played
        if len(self.snapshots) > self.max_size:
            self.snapshots.pop(0)
            self.win_rates.pop(0)

    def reset(self):
        if not self.snapshots or self.rng.random() < self.scripted_prob:
            self.current = self.scripted
        else:
            self.policy.load(self.rng.choice(self.snapshots))
            self.current = self.policy

    def act(self, raw_obs):
        return self.current.act(raw_obs)

    def state(self):
        """
        Docstring for state:

        Everything needed to rebuild the pool after a resume: the snapshot weights,
        their win rates, the size/prob settings and the RNG state.
        """
        return {
            "snapshots": [{k: v.detach().cpu().clone() for k, v in s.items()}
                          for s in self.snapshots],
            "win_rates": list(self.win_rates),
            "max_size": self.max_size,
            "scripted_prob": self.scripted_prob,
            "hidden": self.hidden,
            "rng_state": self.rng.getstate(),
        }

    def load_state(self, state):
        """Docstring for load_state: restore the pool from a dict made by state()."""
        self.max_size = state.get("max_size", self.max_size)
        self.scripted_prob = state.get("scripted_prob", self.scripted_prob)
        self.snapshots = [dict(s) for s in state.get("snapshots", [])]
        self.win_rates = list(state.get("win_rates", [0.5] * len(self.snapshots)))
        if "rng_state" in state:
            self.rng.setstate(state["rng_state"])


def load_state_dict(path, map_location="cpu"):
    """Loads a checkpoint saved by training_loop (plain state_dict or {'model': ...})."""
    ckpt = torch.load(path, map_location=map_location, weights_only=True)
    if isinstance(ckpt, dict) and "model" in ckpt:
        return ckpt["model"]
    return copy.deepcopy(ckpt)


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
