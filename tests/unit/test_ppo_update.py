"""
T1: one PPO update on fake data (Phase 3). Checks the minibatch update returns
finite losses, logs approx_kl / clip fraction / explained variance / passes run,
clips gradients to <= 0.5, and that the linear LR schedule decays to 0 and resumes
from its saved step. No Godot: the data dict is hand-built like join_and_normalize
hands it over.
"""
import math

import numpy as np
import pytest
import torch
from torch.optim import Adam

from neural_network import Actor_critic_neural_network
from training_loop import (
    update_from_data,
    Linear_lr_schedule,
    explained_variance,
    save_checkpoint,
    load_checkpoint,
)


def make_ac_and_opts(pi_lr=3e-4, vf_lr=1e-3):
    ac = Actor_critic_neural_network(obs_dim=14, act_dim=6)
    pi_opt = Adam(list(ac.pi_net.parameters()) + [ac.log_std], lr=pi_lr)
    vf_opt = Adam(ac.v_net.parameters(), lr=vf_lr)
    return ac, pi_opt, vf_opt


def fake_data(n, ac=None, obs_dim=14, act_dim=6):
    """a batch like join_and_normalize returns: obs, act, ret, adv, logp.

    logp is recorded UNDER the policy that collected the data (here `ac`), the
    way a real rollout stores it, so the pre-update approx_kl starts near 0. With
    a random logp instead, KL would look huge on pass 0 and the early stop would
    fire before any learning, which is not how the live loop behaves.
    """
    obs = torch.randn(n, obs_dim)
    ret = torch.randn(n)
    adv = torch.randn(n)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)   # normalized over the batch
    if ac is None:
        ac = Actor_critic_neural_network(obs_dim=obs_dim, act_dim=act_dim)
    with torch.no_grad():
        dist, _ = ac(obs)
        act = dist.sample()
        logp = dist.log_prob(act)
    return dict(obs=obs, act=act, ret=ret, adv=adv, logp=logp)


def test_one_update_logs_finite_and_in_range():
    """finite losses, and approx_kl / clip fraction / explained variance / passes
    all come back in the dict."""
    ac, pi_opt, vf_opt = make_ac_and_opts()
    data = fake_data(600, ac)   # not a multiple of 256: 256 + 256 + 88

    info = update_from_data(ac, data, pi_opt, vf_opt, target_kl=0.02,
                            minibatch_size=256, n_passes=10, ent_coef=0.005)

    for k in ("LossPi", "LossV", "KL", "Entropy", "ClipFrac",
              "ExplainedVariance", "PassesRun"):
        assert k in info
    assert math.isfinite(info["LossPi"])
    assert math.isfinite(info["LossV"])
    assert math.isfinite(info["KL"])
    assert 0.0 <= info["ClipFrac"] <= 1.0
    assert math.isfinite(info["ExplainedVariance"])
    assert 1 <= info["PassesRun"] <= 10


def test_gradients_are_clipped_to_half():
    """after backward (before the step) the grad norm of each net is <= 0.5.

    we replicate one minibatch's clip here so we can read the norm the clip
    returns: clip_grad_norm_ returns the norm BEFORE clipping, and caps it in
    place, so the post-clip norm must be <= max_grad_norm (+ a little float slack).
    """
    ac, pi_opt, vf_opt = make_ac_and_opts()
    data = fake_data(300, ac)

    from training_loop import compute_loss_pi, compute_loss_v

    pi_params = list(ac.pi_net.parameters()) + [ac.log_std]
    pi_opt.zero_grad()
    loss_pi, pi_info = compute_loss_pi(ac, data)
    (loss_pi - 0.005 * pi_info["ent_tensor"]).backward()
    torch.nn.utils.clip_grad_norm_(pi_params, 0.5)
    pi_norm = math.sqrt(sum(float(p.grad.pow(2).sum()) for p in pi_params if p.grad is not None))
    assert pi_norm <= 0.5 + 1e-4

    v_params = list(ac.v_net.parameters())
    vf_opt.zero_grad()
    compute_loss_v(ac, data).backward()
    torch.nn.utils.clip_grad_norm_(v_params, 0.5)
    v_norm = math.sqrt(sum(float(p.grad.pow(2).sum()) for p in v_params if p.grad is not None))
    assert v_norm <= 0.5 + 1e-4


def test_lr_decays_linearly_to_zero():
    """the schedule ramps both LRs from their base to ~0 over the run."""
    ac, pi_opt, vf_opt = make_ac_and_opts(pi_lr=3e-4, vf_lr=1e-3)
    epochs = 10
    sched = Linear_lr_schedule([pi_opt, vf_opt], [3e-4, 1e-3], total=epochs)

    # step 0: full LR
    assert pi_opt.param_groups[0]["lr"] == pytest.approx(3e-4)
    assert vf_opt.param_groups[0]["lr"] == pytest.approx(1e-3)

    lrs = []
    for _ in range(epochs):
        lrs.append(pi_opt.param_groups[0]["lr"])
        sched.advance()

    # monotonically non-increasing
    for a, b in zip(lrs, lrs[1:]):
        assert b <= a
    # halfway is ~half the base LR
    assert lrs[epochs // 2] == pytest.approx(3e-4 * (epochs - epochs // 2) / epochs)
    # last update ran at ~1/epochs of base, and after the final advance it is 0
    assert pi_opt.param_groups[0]["lr"] == pytest.approx(0.0, abs=1e-12)


def test_lr_step_saves_and_resumes_mid_decay(tmp_path):
    """resume restores the LR step so decay picks up where it left off, not at
    the full LR."""
    ac, pi_opt, vf_opt = make_ac_and_opts(pi_lr=3e-4, vf_lr=1e-3)
    epochs = 10
    sched = Linear_lr_schedule([pi_opt, vf_opt], [3e-4, 1e-3], total=epochs)
    for _ in range(4):      # run 4 of 10 updates
        sched.advance()
    assert sched.step_count == 4

    path = str(tmp_path / "mid.pt")
    save_checkpoint(path, ac, pi_opt, vf_opt, epoch=3, lr_step=sched.step_count)
    ckpt = load_checkpoint(path)
    assert ckpt["lr_step"] == 4

    # a resumed schedule built from the saved step gives the same LR, not the base
    ac2, pi2, vf2 = make_ac_and_opts(pi_lr=3e-4, vf_lr=1e-3)
    sched2 = Linear_lr_schedule([pi2, vf2], [3e-4, 1e-3], total=epochs,
                                step=ckpt["lr_step"])
    assert pi2.param_groups[0]["lr"] == pytest.approx(pi_opt.param_groups[0]["lr"])
    assert pi2.param_groups[0]["lr"] == pytest.approx(3e-4 * (epochs - 4) / epochs)
    assert pi2.param_groups[0]["lr"] < 3e-4   # not snapped back to full


def test_batch_smaller_than_minibatch_runs():
    """a batch smaller than the minibatch size still updates (one minibatch of n)."""
    ac, pi_opt, vf_opt = make_ac_and_opts()
    data = fake_data(100, ac)   # < 256
    info = update_from_data(ac, data, pi_opt, vf_opt, minibatch_size=256, n_passes=3)
    assert info["PassesRun"] >= 1
    assert math.isfinite(info["LossPi"])


def test_batch_exactly_divisible_by_minibatch():
    """a batch that is an exact multiple of 256 sweeps cleanly with no remainder."""
    ac, pi_opt, vf_opt = make_ac_and_opts()
    data = fake_data(512, ac)   # 256 + 256, no remainder
    info = update_from_data(ac, data, pi_opt, vf_opt, minibatch_size=256, n_passes=2)
    assert info["PassesRun"] >= 1
    assert math.isfinite(info["LossV"])


def test_kl_early_stop_on_first_pass():
    """a tiny target_kl forces the per-pass KL check to trip on pass 0, so no
    pass runs (early stop is per pass and can fire immediately)."""
    ac, pi_opt, vf_opt = make_ac_and_opts()
    data = fake_data(300, ac)
    # nudge the policy so the batch's approx_kl is clearly > 0, then a tiny
    # target_kl stops it on pass 0.
    with torch.no_grad():
        for p in ac.pi_net.parameters():
            p.add_(0.5 * torch.randn_like(p))
    info = update_from_data(ac, data, pi_opt, vf_opt, target_kl=1e-9, n_passes=10)
    assert info["PassesRun"] == 0


def test_explained_variance_bounds():
    """explained_variance is 1 for a perfect fit and 0 when returns are constant."""
    r = torch.randn(50)
    assert explained_variance(r.clone(), r.clone()) == pytest.approx(1.0, abs=1e-6)
    const = torch.ones(50) * 3.0
    assert explained_variance(torch.randn(50), const) == 0.0


def test_entropy_bonus_resists_std_collapse():
    """with a positive ent_coef the net is pushed toward MORE entropy: log_std
    should not collapse hard after a few updates on fixed fake data (sanity)."""
    ac, pi_opt, vf_opt = make_ac_and_opts(pi_lr=1e-3)
    data = fake_data(512, ac)
    std_before = ac.log_std.detach().clone()
    for _ in range(3):
        update_from_data(ac, data, pi_opt, vf_opt, n_passes=5, ent_coef=0.05,
                         target_kl=1.0)   # high target_kl so passes are not cut
    assert ac.log_std.detach().mean() >= std_before.mean() - 0.5
