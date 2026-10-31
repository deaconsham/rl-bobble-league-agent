"""
T1 (Phase 5): the both-teams buffer.

The opponent (Team B) side of a live-vs-live game goes into an Opponent_buffer,
which must keep separate paths apart (GAE never bleeds across an episode end) and
join into the same training batch as the learner (Team A) buffers without mixing
their advantages until the single whole-batch normalization.
"""
import numpy as np
import torch

from training_loop import (
    Opponent_buffer,
    PPO_buffer,
    join_and_normalize,
)


def test_opponent_buffer_keeps_two_paths_apart():
    # two short paths; GAE on path 2 must not see path 1's rewards/values.
    buf = Opponent_buffer(gamma=0.99, lam=0.97)
    obs = np.zeros(14, dtype=np.float32)
    act = np.zeros(6, dtype=np.float32)

    # path 1: three turns then a terminal goal (bootstrap 0)
    for rew, val in [(0.0, 0.5), (0.0, 0.4), (1.0, 0.3)]:
        buf.store(obs, act, rew, val, 0.0)
    buf.finish_path(0.0)
    len_after_p1 = len(buf)
    assert len_after_p1 == 3

    # path 2: two turns, cut by the epoch end (bootstrap from a critic value)
    for rew, val in [(0.0, 0.2), (-1.0, 0.1)]:
        buf.store(obs, act, rew, val, 0.0)
    buf.finish_path(0.25)
    assert len(buf) == 5

    data = buf.get()
    assert data['obs'].shape == (5, 14)
    assert data['act'].shape == (5, 6)
    assert data['ret'].shape == (5,)
    assert data['adv'].shape == (5,)
    # returns of path 1 depend only on path-1 rewards (terminal, no bootstrap):
    # r2=1 -> ret3=1; ret2 = 0 + .99*1 = .99; ret1 = 0 + .99*.99 = .9801
    assert np.allclose(data['ret'][:3].numpy(), [0.9801, 0.99, 1.0], atol=1e-5)
    # get() clears the buffer
    assert len(buf) == 0


def test_opponent_buffer_gae_matches_ppo_buffer_for_one_path():
    # the same rewards/values through PPO_buffer and Opponent_buffer give the
    # same advantages and returns (both use discount_cumsum with a last_val).
    rews = [0.1, -0.2, 0.3, 1.0]
    vals = [0.5, 0.4, 0.6, 0.2]
    obs = np.zeros(14, dtype=np.float32)
    act = np.zeros(6, dtype=np.float32)

    ppo = PPO_buffer(14, 6, len(rews), gamma=0.99, lam=0.97)
    for r, v in zip(rews, vals):
        ppo.store(obs, act, r, v, 0.0)
    ppo.finish_path(0.0)
    ppo_data = ppo.get(normalize_adv=False)

    opp = Opponent_buffer(gamma=0.99, lam=0.97)
    for r, v in zip(rews, vals):
        opp.store(obs, act, r, v, 0.0)
    opp.finish_path(0.0)
    opp_data = opp.get()

    assert np.allclose(ppo_data['adv'].numpy(), opp_data['adv'].numpy(), atol=1e-6)
    assert np.allclose(ppo_data['ret'].numpy(), opp_data['ret'].numpy(), atol=1e-6)


def test_join_folds_both_sides_into_one_batch():
    # two learner buffers (Team A) of 4 turns each + one opponent buffer of 3 turns.
    obs = np.zeros(14, dtype=np.float32)
    act = np.zeros(6, dtype=np.float32)

    a_buffers = []
    for _ in range(2):
        b = PPO_buffer(14, 6, 4, gamma=0.99, lam=0.97)
        for i in range(4):
            b.store(obs, act, float(i % 2), 0.3, 0.0)
        b.finish_path(0.0)
        a_buffers.append(b)

    opp = Opponent_buffer(gamma=0.99, lam=0.97)
    for i in range(3):
        opp.store(obs, act, float(-(i % 2)), 0.2, 0.0)
    opp.finish_path(0.0)
    opp_data = opp.get()

    joined = join_and_normalize(a_buffers, extra_data=[opp_data])
    # 2 * 4 learner turns + 3 opponent turns
    assert joined['obs'].shape[0] == 11
    # advantages are normalized over the WHOLE 11-row batch. join uses torch's
    # std (unbiased, ddof=1), so the population std of the result is sqrt((n-1)/n).
    adv = joined['adv'].numpy()
    n = adv.shape[0]
    assert abs(adv.mean()) < 1e-5
    assert abs(adv.std() - np.sqrt((n - 1) / n)) < 1e-2


def test_join_ignores_empty_opponent_data():
    # a non-live epoch has no opponent turns: an empty data dict is dropped.
    obs = np.zeros(14, dtype=np.float32)
    act = np.zeros(6, dtype=np.float32)
    b = PPO_buffer(14, 6, 4, gamma=0.99, lam=0.97)
    for i in range(4):
        b.store(obs, act, 1.0, 0.3, 0.0)
    b.finish_path(0.0)

    empty = Opponent_buffer().get()  # nothing stored
    joined = join_and_normalize([b], extra_data=[empty])
    assert joined['obs'].shape[0] == 4
