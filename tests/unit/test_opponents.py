"""
T1 (Phase 5): the self-play opponent mix.

Checks that SelfPlayPool.reset() draws the chaser / live / frozen slots in roughly
the set proportions over 10,000 draws, that the live slot only appears once a live
net is attached, that sibling pools share one snapshot list, and that the chaser
still produces in-range angle+power actions.
"""
import numpy as np

from neural_network import Actor_critic_neural_network
from opponents import (
    ChaserOpponent,
    LiveOpponent,
    PolicyOpponent,
    SelfPlayPool,
)


def _fresh_net():
    net = Actor_critic_neural_network(obs_dim=14, act_dim=6, hidden=[32, 32])
    net.eval()
    return net


def _fill_snapshots(pool, n=3):
    net = _fresh_net()
    for _ in range(n):
        pool.add(net.state_dict())


def test_mix_matches_the_set_weights_over_10k_draws():
    pool = SelfPlayPool(scripted_prob=0.15, live_prob=0.30, hidden=(32, 32), seed=0)
    pool.attach_live(_fresh_net())
    _fill_snapshots(pool, 3)

    counts = {"chaser": 0, "live": 0, "self": 0}
    for _ in range(10000):
        pool.reset()
        counts[pool.name] += 1

    chaser = counts["chaser"] / 10000
    live = counts["live"] / 10000
    frozen = counts["self"] / 10000
    # targets: 0.15 / 0.30 / 0.55, allow +-0.03 sampling slack
    assert abs(chaser - 0.15) < 0.03, counts
    assert abs(live - 0.30) < 0.03, counts
    assert abs(frozen - 0.55) < 0.03, counts
    assert chaser + live + frozen == 1.0


def test_live_slot_absent_until_a_net_is_attached():
    pool = SelfPlayPool(scripted_prob=0.15, live_prob=0.30, hidden=(32, 32), seed=1)
    _fill_snapshots(pool, 3)  # snapshots exist but no live net yet

    names = set()
    for _ in range(2000):
        pool.reset()
        names.add(pool.name)
    # with no live net it is only chaser or a frozen copy, never "live"
    assert "live" not in names
    assert names <= {"chaser", "self"}


def test_is_live_current_tracks_the_live_slot():
    pool = SelfPlayPool(scripted_prob=0.0, live_prob=1.0, hidden=(32, 32), seed=2)
    pool.attach_live(_fresh_net())
    _fill_snapshots(pool, 2)
    pool.reset()
    # live_prob 1.0 with a live net -> always the live learner
    assert pool.name == "live"
    assert pool.is_live_current() is True

    chaser_only = SelfPlayPool(scripted_prob=1.0, live_prob=0.0, hidden=(32, 32), seed=3)
    chaser_only.attach_live(_fresh_net())
    chaser_only.reset()
    assert chaser_only.name == "chaser"
    assert chaser_only.is_live_current() is False


def test_siblings_share_one_snapshot_list_and_live_net():
    pool = SelfPlayPool(scripted_prob=0.15, live_prob=0.30, hidden=(32, 32), seed=4)
    pool.attach_live(_fresh_net())
    siblings = pool.make_siblings(6)
    assert len(siblings) == 6 and siblings[0] is pool

    # a snapshot added to any sibling is seen by all (shared list object)
    _fill_snapshots(siblings[2], 1)
    for sib in siblings:
        assert len(sib.snapshots) == 1
    _fill_snapshots(pool, 1)
    for sib in siblings:
        assert len(sib.snapshots) == 2

    # every sibling has a live slot, and each keeps its OWN current opponent
    for sib in siblings:
        assert sib.live is not None
    siblings[0].current = siblings[0].scripted
    siblings[1].current = siblings[1].live
    assert siblings[0].name == "chaser"
    assert siblings[1].name == "live"


def test_live_opponent_samples_and_records_its_turn():
    net = _fresh_net()
    live = LiveOpponent(net)
    raw = np.zeros(14, dtype=np.float32)
    a = live.act(raw)
    assert a.shape == (6,)
    assert live.last is not None
    assert live.last["obs"].shape == (14,)
    assert live.last["act"].shape == (6,)
    assert np.isfinite(live.last["val"])
    assert np.isfinite(live.last["logp"])


def test_chaser_output_shape_and_range():
    chaser = ChaserOpponent(seed=0)
    raw = np.array([1.0, -2.0, 3, 4, -3, -4, 0, 5, 2, -6, -2, -6, 0, -8], dtype=np.float32)
    action = chaser.act(raw)
    assert action.shape == (6,)
    angles = action[0::2]
    powers = action[1::2]
    assert np.all(angles >= -1.0) and np.all(angles <= 1.0)
    assert np.all(powers >= -1.0) and np.all(powers <= 1.0)


def test_mix_without_frozen_copies_uses_live_or_chaser():
    # before the first snapshot exists, draws are only chaser or live
    pool = SelfPlayPool(scripted_prob=0.15, live_prob=0.30, hidden=(32, 32), seed=5)
    pool.attach_live(_fresh_net())
    names = set()
    for _ in range(2000):
        pool.reset()
        names.add(pool.name)
    assert names <= {"chaser", "live"}
    assert "self" not in names  # no frozen copies yet
