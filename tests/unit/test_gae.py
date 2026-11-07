"""
T1: GAE and discounted returns. checks discount_cumsum against hand numbers,
and that finish_path uses 0 bootstrap on a real terminal but the critic's
value on a truncated path.
"""
import numpy as np
import pytest

from training_loop import PPO_buffer


def test_discount_cumsum_matches_hand_maths():
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    d = 0.9
    # by hand, from the back:
    #   t2 = 3
    #   t1 = 2 + 0.9*3 = 4.7
    #   t0 = 1 + 0.9*4.7 = 5.23
    out = PPO_buffer.discount_cumsum(x, d)
    assert np.allclose(out, [5.23, 4.7, 3.0])


def test_terminal_path_uses_zero_bootstrap():
    buf = PPO_buffer(obs_dim=2, act_dim=2, size=2, gamma=1.0, lam=1.0)
    # two steps, reward 1 each, critic says value 0.5 each
    buf.store([0, 0], [0, 0], rew=1.0, val=0.5, logp=0.0)
    buf.store([0, 0], [0, 0], rew=1.0, val=0.5, logp=0.0)
    buf.finish_path(last_val=0.0)  # real terminal
    # returns with gamma=1, no bootstrap: [1+1, 1] = [2, 1]
    assert np.allclose(buf.ret_buf, [2.0, 1.0])
    # with lam=gamma=1 and vals 0.5: deltas = [1+0.5-0.5, 1+0-0.5] = [1.0, 0.5]
    # adv (cumsum, discount 1) = [1.5, 0.5]
    assert np.allclose(buf.adv_buf, [1.5, 0.5])


def test_truncated_path_bootstraps_from_critic():
    buf = PPO_buffer(obs_dim=2, act_dim=2, size=1, gamma=1.0, lam=1.0)
    buf.store([0, 0], [0, 0], rew=1.0, val=0.0, logp=0.0)
    buf.finish_path(last_val=4.0)  # cut off, critic says 4.0 ahead
    # return = reward + bootstrap = 1 + 4 = 5
    assert np.allclose(buf.ret_buf, [5.0])
    # delta = 1 + 1.0*4.0 - 0.0 = 5.0
    assert np.allclose(buf.adv_buf, [5.0])


def test_get_normalizes_advantages(monkeypatch):
    buf = PPO_buffer(obs_dim=2, act_dim=2, size=3, gamma=0.99, lam=0.97)
    for _ in range(3):
        buf.store([0, 0], [0, 0], rew=1.0, val=0.0, logp=0.0)
    buf.finish_path(last_val=0.0)
    data = buf.get(device='cpu')
    adv = data['adv'].numpy()
    assert abs(adv.mean()) < 1e-5
    assert adv.std() == pytest.approx(1.0, abs=1e-3)
