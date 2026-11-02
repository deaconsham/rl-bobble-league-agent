"""
T1 for Phase 12 (better data and rewards). No Godot.

- input normalization: the running obs stats update from batches, normalize to
  ~unit scale, survive save/load inside the model state_dict, and a pre-Phase-12
  save with no obs-norm buffers loads with fresh identity stats (load_model_state)
  instead of crashing;
- potential-based shaping: covered in test_rewards.py (telescoping sum, phi=0 at a
  goal, skip after out-of-bounds);
- random starts: the env draws the random-start flag at the chosen rate and the
  flag reaches Godot (checked against the mock), and the env reads the mock's
  out-of-bounds flag and skips that turn's shaping;
- reward scaling: the running std of the discounted return scales rewards and
  survives save/load.
"""
import numpy as np
import pytest
import torch
from torch.optim import Adam

from neural_network import (
    Actor_critic_neural_network, Input_norm, RewardScaler, load_model_state,
)
from training_loop import save_checkpoint, load_checkpoint
from shared_memory_link import Shared_memory_link
from bobble_league_env import Bobble_league_env
from mock_godot import Mock_godot


# ---- input normalization (step 2) ----

def test_input_norm_updates_and_normalizes():
    """the running per-dim stats track a batch, and forward() brings a sample from
    that distribution back to ~unit scale."""
    rms = Input_norm(dim=14)
    rng = np.random.default_rng(0)
    data = rng.normal(5.0, 3.0, size=(4000, 14)).astype(np.float32)
    rms.update(torch.as_tensor(data))
    # mean/var track the batch per dimension
    assert np.allclose(rms.mean.numpy(), data.mean(axis=0), atol=1e-3)
    assert np.allclose(rms.var.numpy(), data.var(axis=0), atol=1e-1)
    # a fresh sample from the same distribution normalizes to ~0 mean, ~1 std
    sample = torch.as_tensor(rng.normal(5.0, 3.0, size=(2000, 14)).astype(np.float32))
    out = rms(sample)
    assert abs(float(out.mean())) < 0.1
    assert abs(float(out.std()) - 1.0) < 0.1


def test_fresh_input_norm_is_identity():
    rms = Input_norm(dim=14)
    x = torch.randn(8, 14)
    assert torch.allclose(rms(x), x, atol=1e-3)   # mean 0, var 1


def test_input_norm_clips_outliers():
    rms = Input_norm(dim=4, clip=5.0)
    x = torch.full((1, 4), 1000.0)   # far outside the identity stats
    assert torch.all(rms(x) <= 5.0 + 1e-5)
    assert torch.all(rms(x) >= -5.0 - 1e-5)


def test_input_norm_stats_travel_in_state_dict():
    """the obs-norm buffers are part of the model state_dict, so a checkpoint
    carries them and every consumer that loads the weights gets them."""
    ac = Actor_critic_neural_network(input_norm=True)
    keys = set(ac.state_dict().keys())
    assert {"obs_norm.mean", "obs_norm.var", "obs_norm.count"} <= keys


def test_input_norm_stats_survive_save_load(tmp_path):
    ac = Actor_critic_neural_network(input_norm=True)
    ac.obs_norm.update(torch.as_tensor(
        np.random.default_rng(1).normal(2.0, 4.0, size=(3000, 14)).astype(np.float32)))
    pi_opt = Adam(list(ac.pi_net.parameters()) + [ac.log_std], lr=1e-3)
    vf_opt = Adam(ac.v_net.parameters(), lr=1e-3)
    path = str(tmp_path / "in.pt")
    save_checkpoint(path, ac, pi_opt, vf_opt, epoch=1, lr_step=1)

    ac2 = Actor_critic_neural_network(input_norm=True)
    load_model_state(ac2, load_checkpoint(path)["model"])
    assert torch.allclose(ac2.obs_norm.mean, ac.obs_norm.mean, atol=1e-6)
    assert torch.allclose(ac2.obs_norm.var, ac.obs_norm.var, atol=1e-6)
    # a net that was fit and one loaded from it score an obs identically
    x = torch.randn(5, 14)
    with torch.no_grad():
        assert torch.allclose(ac.obs_norm(x), ac2.obs_norm(x), atol=1e-6)


def test_old_save_without_input_norm_loads_with_fresh_stats(tmp_path):
    """a pre-Phase-12 save has no obs_norm.* buffers; load_model_state keeps the
    fresh identity stats instead of crashing (the plan's compatibility case)."""
    # build a state_dict with the obs_norm keys stripped, like an old save
    old = Actor_critic_neural_network(input_norm=True)
    sd = {k: v for k, v in old.state_dict().items() if not k.startswith("obs_norm.")}
    fresh = Actor_critic_neural_network(input_norm=True)
    load_model_state(fresh, sd)   # must not raise
    # the normalizer is left as the identity
    x = torch.randn(6, 14)
    assert torch.allclose(fresh.obs_norm(x), x, atol=1e-3)


def test_load_model_state_rejects_a_real_mismatch(tmp_path):
    """a genuine shape mismatch (LayerNorm vs not) is still re-raised, not hidden
    by the obs-norm tolerance."""
    plain = Actor_critic_neural_network(use_layernorm=False, input_norm=True)
    ln = Actor_critic_neural_network(use_layernorm=True, input_norm=True)
    with pytest.raises(RuntimeError):
        load_model_state(ln, plain.state_dict())


def test_input_norm_frozen_at_eval():
    """eval never calls update(), so repeated forward passes do not move the
    stats (the plan's 'frozen at eval time')."""
    ac = Actor_critic_neural_network(input_norm=True)
    ac.eval()
    before = ac.obs_norm.mean.clone()
    with torch.no_grad():
        for _ in range(5):
            ac(torch.randn(16, 14))
    assert torch.allclose(ac.obs_norm.mean, before)


# ---- random starts (step 1) ----

def test_random_start_flag_reaches_godot(tmp_path):
    """with random_start_prob=1.0 every reset asks the mock for a scattered
    layout, so its random-start counter climbs once per reset."""
    path = str(tmp_path / "shm.dat")
    game = Mock_godot(path, goal_after=99)
    bridge = Shared_memory_link(path=path, create=False)
    env = Bobble_league_env(bridge=bridge, step_timeout=5.0, random_start_prob=1.0)
    try:
        for _ in range(3):
            env.reset()
        assert game.random_start_count == 3
    finally:
        env.close()
        game.stop()


def test_no_random_start_when_prob_zero(tmp_path):
    path = str(tmp_path / "shm.dat")
    game = Mock_godot(path, goal_after=99)
    bridge = Shared_memory_link(path=path, create=False)
    env = Bobble_league_env(bridge=bridge, step_timeout=5.0, random_start_prob=0.0)
    try:
        for _ in range(3):
            env.reset()
        assert game.random_start_count == 0
    finally:
        env.close()
        game.stop()


class _OobMock(Mock_godot):
    """a mock that reports an out-of-bounds kickoff on its first step, to check the
    env reads byte 117 and skips that turn's shaping."""
    def __init__(self, *args, **kwargs):
        self._oob_done = False
        super().__init__(*args, **kwargs)

    def _run(self):
        import struct as _struct
        import time as _time
        while not self._stop:
            if self._hang:
                _time.sleep(0.005)
                continue
            if self.buf[86] == 1:   # RESET_REQ
                self._turn = 0
                self._ball = np.zeros(2, dtype=np.float32)
                self._write_obs()
                self.buf[80:84] = _struct.pack("<f", 0.0)
                self.buf[84] = 0
                self.buf[88] = 0
                self.buf[117] = 0
                self.buf[87] = self.learner_team
                self.buf[89] = self.play_mode
                self.buf[86] = 0
                continue
            if self.buf[85] == 1:   # STEP_REQ
                self._turn += 1
                # jump the ball (as an OOB kickoff would) and flag it on turn 1
                oob = not self._oob_done
                if oob:
                    self._oob_done = True
                    self._ball = np.zeros(2, dtype=np.float32)
                self._write_obs()
                self.buf[80:84] = _struct.pack("<f", 0.0)
                self.buf[84] = 0
                self.buf[88] = 0
                self.buf[117] = 1 if oob else 0
                self.buf[87] = self.learner_team
                self.buf[89] = self.play_mode
                self.buf[120:124] = _struct.pack("<f", 1.0)
                self.buf[85] = 0
                continue
            _time.sleep(0.0005)


def test_env_skips_shaping_after_out_of_bounds(tmp_path):
    from bobble_league_env import RewardConfig
    path = str(tmp_path / "shm.dat")
    game = _OobMock(path, goal_after=99)
    bridge = Shared_memory_link(path=path, create=False)
    cfg = RewardConfig(ball_to_goal=0.5, players_to_ball=0.05, defence=0.02)
    # the OOB shaping-skip is part of the potential form (step 3); turn it on
    env = Bobble_league_env(bridge=bridge, reward_config=cfg, step_timeout=5.0,
                            potential_shaping=True)
    try:
        env.reset()
        # move the ball far so a non-skipped turn WOULD produce shaping
        env._raw_obs = np.concatenate([[0, 10], np.zeros(12)]).astype(np.float32)
        _, r, _, _, info = env.step(np.zeros(6, dtype=np.float32))
        assert info["oob_reset"] is True
        assert info["shaping"] == 0.0     # skipped
    finally:
        env.close()
        game.stop()


# ---- reward scaling (step 4) ----

def test_reward_scaler_scales_toward_unit():
    """after folding a stream in, the scaled rewards have ~unit-scale discounted
    return, and a positive reward stays positive (no mean subtraction)."""
    rs = RewardScaler(n_games=1, gamma=0.99)
    rng = np.random.default_rng(0)
    scaled_all = []
    for _ in range(500):
        r = rng.normal(2.0, 1.0, size=1)
        scaled_all.append(rs.scale(r, np.array([False]))[0])
    # a fixed positive reward stays positive after scaling
    out = rs.scale(np.array([3.0]), np.array([False]))
    assert out[0] > 0
    # the running std has grown past 1 (discounted returns of a biased stream),
    # so later rewards are scaled DOWN from their raw size
    assert np.sqrt(rs.ret_rms.var) > 1.0


def test_reward_scaler_resets_running_return_on_done():
    rs = RewardScaler(n_games=2, gamma=0.9)
    rs.scale(np.array([1.0, 1.0]), np.array([False, False]))
    rs.scale(np.array([1.0, 1.0]), np.array([True, False]))   # game 0 ended
    # game 0's running return was reset to 0, game 1's kept accumulating
    assert rs._returns[0] == pytest.approx(0.0)
    assert rs._returns[1] > rs._returns[0]


def test_reward_scaler_state_round_trip():
    rs = RewardScaler(n_games=4, gamma=0.97)
    rng = np.random.default_rng(2)
    for _ in range(200):
        rs.scale(rng.normal(0.0, 2.0, size=4), np.zeros(4, dtype=bool))
    state = rs.state()
    rs2 = RewardScaler(n_games=4, gamma=0.97)
    rs2.load_state(state)
    assert rs2.ret_rms.mean == pytest.approx(rs.ret_rms.mean, rel=1e-9)
    assert rs2.ret_rms.var == pytest.approx(rs.ret_rms.var, rel=1e-9)


def test_reward_scaler_fresh_is_near_identity():
    """an empty/old save loads as std ~1 (var 1) BEFORE any scaling, so a resumed
    run picks up the identity and re-learns the scale from there."""
    rs = RewardScaler(n_games=1, gamma=0.99)
    rs.load_state(None)   # no-op: leaves the fresh identity stats
    assert rs.ret_rms.var == pytest.approx(1.0)
    # scaling still produces a finite, positive value for a positive reward
    out = rs.scale(np.array([1.0]), np.array([False]))
    assert np.isfinite(out[0]) and out[0] > 0
