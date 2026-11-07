"""
T1: value normalization, critic passes, the KL teacher and LayerNorm. No Godot.

- value normalization: the running mean/std matches numpy over the same stream,
  normalize/denormalize round trips, the stats survive save/load in a checkpoint,
  and a save WITHOUT the stats loads with fresh identity stats instead of crashing;
- separate critic passes: the critic can run more passes than the policy, and the
  reported policy/critic pass counts come back;
- mean-only KL to a teacher: the KL term is >= 0, is 0 when the teacher IS the
  learner, and training against a teacher never pulls log_std (the spread) up;
- KL fade: the weight the loop applies falls to 0 at the fade point and stays 0;
- LayerNorm: it changes the saved weight shape, so a non-LayerNorm save refuses to
  load into a LayerNorm net, and a LayerNorm net round trips through a checkpoint.
"""
import math

import numpy as np
import pytest
import torch
from torch.optim import Adam

from neural_network import (
    Actor_critic_neural_network, INIT_LOG_STD, Running_mean_std,
)
from training_loop import (
    update_from_data, kl_to_teacher, compute_loss_v,
    save_checkpoint, load_checkpoint,
)


def make_ac_and_opts(use_layernorm=False, pi_lr=3e-4, vf_lr=1e-3):
    ac = Actor_critic_neural_network(obs_dim=14, act_dim=6, use_layernorm=use_layernorm)
    pi_opt = Adam(list(ac.pi_net.parameters()) + [ac.log_std], lr=pi_lr)
    vf_opt = Adam(ac.v_net.parameters(), lr=vf_lr)
    return ac, pi_opt, vf_opt


def fake_data(n, ac, obs_dim=14, act_dim=6):
    """
    Docstring for fake_data:

    a batch like join_and_normalize returns: obs, act, ret, adv, logp, with
    logp recorded under `ac` so the pre-update KL starts near 0.
    """
    obs = torch.randn(n, obs_dim)
    ret = torch.randn(n) * 5.0 + 2.0   # off unit scale, so value-norm has work to do
    adv = torch.randn(n)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    with torch.no_grad():
        dist, _ = ac(obs)
        act = dist.sample()
        logp = dist.log_prob(act)
    return dict(obs=obs, act=act, ret=ret, adv=adv, logp=logp)


# ---- value normalization ----

def test_running_mean_std_matches_numpy():
    rms = Running_mean_std()
    stream = []
    rng = np.random.default_rng(0)
    for _ in range(20):
        batch = rng.normal(3.0, 2.0, size=rng.integers(50, 200))
        rms.update(batch)
        stream.append(batch)
    allx = np.concatenate(stream)
    assert rms.mean == pytest.approx(float(allx.mean()), rel=1e-6)
    assert rms.var == pytest.approx(float(allx.var()), rel=1e-4)


def test_normalize_denormalize_round_trip():
    rms = Running_mean_std()
    rms.update(np.random.default_rng(1).normal(10.0, 4.0, size=5000))
    x = torch.randn(100) * 3.0 + 1.0
    back = rms.denormalize(rms.normalize(x))
    assert torch.allclose(back, x, atol=1e-4)
    # torch and numpy paths agree
    xn = x.numpy()
    assert np.allclose(rms.normalize(xn), rms.normalize(x).numpy(), atol=1e-5)


def test_fresh_normalizer_is_identity():
    rms = Running_mean_std()
    x = torch.randn(10)
    assert torch.allclose(rms.normalize(x), x, atol=1e-3)  # mean 0, std ~1


def test_value_norm_stats_survive_save_load(tmp_path):
    ac, pi_opt, vf_opt = make_ac_and_opts()
    rms = Running_mean_std()
    rms.update(np.random.default_rng(2).normal(7.0, 3.0, size=4000))
    path = str(tmp_path / 'vn.pt')
    save_checkpoint(path, ac, pi_opt, vf_opt, epoch=1, lr_step=1, value_norm=rms)

    ckpt = load_checkpoint(path)
    assert 'value_norm' in ckpt
    rms2 = Running_mean_std()
    rms2.load_state(ckpt['value_norm'])
    assert rms2.mean == pytest.approx(rms.mean, rel=1e-9)
    assert rms2.var == pytest.approx(rms.var, rel=1e-9)
    assert rms2.count == pytest.approx(rms.count, rel=1e-9)


def test_save_without_value_norm_loads_with_fresh_stats(tmp_path):
    """
    Docstring for test_save_without_value_norm_loads_with_fresh_stats:

    a pre-Phase-11 save has no value_norm key; a fresh normalizer loading its
    (absent) state stays the identity instead of crashing.
    """
    ac, pi_opt, vf_opt = make_ac_and_opts()
    path = str(tmp_path / 'old.pt')
    save_checkpoint(path, ac, pi_opt, vf_opt, epoch=0, lr_step=0)  # no value_norm
    ckpt = load_checkpoint(path)
    assert 'value_norm' not in ckpt
    rms = Running_mean_std()
    rms.load_state(ckpt.get('value_norm'))   # None -> no-op
    x = torch.randn(10)
    assert torch.allclose(rms.normalize(x), x, atol=1e-3)


def test_value_norm_updates_from_batch_in_update():
    """
    Docstring for test_value_norm_updates_from_batch_in_update:

    update_from_data folds the batch returns into the normalizer, so after the
    update its mean/std track the (off-unit-scale) returns.
    """
    ac, pi_opt, vf_opt = make_ac_and_opts()
    rms = Running_mean_std()
    data = fake_data(600, ac)
    update_from_data(ac, data, pi_opt, vf_opt, n_passes=2, n_passes_v=2,
                     value_norm=rms, target_kl=1.0)
    ret = data['ret'].numpy()
    assert rms.mean == pytest.approx(float(ret.mean()), rel=1e-3)
    assert math.sqrt(rms.var) == pytest.approx(float(ret.std()), rel=1e-2)


def test_compute_loss_v_uses_normalized_target():
    """
    Docstring for test_compute_loss_v_uses_normalized_target:

    with a normalizer whose stats match the returns, the critic (output ~0 at
    init) sees a near-zero-mean target, so the loss is far smaller than the raw
    MSE against returns centred at +2.
    """
    ac, _, _ = make_ac_and_opts()
    data = fake_data(400, ac)
    rms = Running_mean_std()
    rms.update(data['ret'])
    raw = compute_loss_v(ac, data, value_norm=None).item()
    normed = compute_loss_v(ac, data, value_norm=rms).item()
    assert normed < raw


# ---- separate critic passes ----

def test_critic_runs_more_passes_than_policy():
    ac, pi_opt, vf_opt = make_ac_and_opts()
    data = fake_data(300, ac)
    info = update_from_data(ac, data, pi_opt, vf_opt, n_passes=2, n_passes_v=6,
                            target_kl=1.0)   # high target_kl so the policy is not early-stopped
    assert info['PassesRun'] == 2
    assert info['CriticPassesRun'] == 6


def test_critic_passes_default_to_policy_passes():
    ac, pi_opt, vf_opt = make_ac_and_opts()
    data = fake_data(300, ac)
    info = update_from_data(ac, data, pi_opt, vf_opt, n_passes=4, n_passes_v=None,
                            target_kl=1.0)
    assert info['CriticPassesRun'] == 4


def test_n_minibatches_splits_the_batch():
    """
    Docstring for test_n_minibatches_splits_the_batch:

    n_minibatches=4 still updates (four large chunks); a batch not divisible by
    4 is handled (ceil).
    """
    ac, pi_opt, vf_opt = make_ac_and_opts()
    data = fake_data(1000, ac)   # 1000 / 4 = 250 per chunk
    info = update_from_data(ac, data, pi_opt, vf_opt, n_passes=2, n_minibatches=4,
                            target_kl=1.0)
    assert info['PassesRun'] == 2
    assert math.isfinite(info['LossPi'])
    data2 = fake_data(999, ac)   # not divisible: 250+250+250+249
    info2 = update_from_data(ac, data2, pi_opt, vf_opt, n_passes=1,
                             n_minibatches=4, target_kl=1.0)
    assert info2['PassesRun'] == 1


# ---- mean-only KL to the teacher ----

def test_kl_to_teacher_zero_when_same_net():
    ac, _, _ = make_ac_and_opts()
    obs = torch.randn(32, 14)
    kl = kl_to_teacher(ac, ac, obs)
    assert float(kl.detach()) == pytest.approx(0.0, abs=1e-6)


def test_kl_to_teacher_positive_when_means_differ():
    ac, _, _ = make_ac_and_opts()
    teacher, _, _ = make_ac_and_opts()
    with torch.no_grad():
        for p in teacher.pi_net.parameters():
            p.add_(0.3 * torch.randn_like(p))
    obs = torch.randn(64, 14)
    assert float(kl_to_teacher(ac, teacher, obs).detach()) > 0.0


def test_kl_term_does_not_pull_log_std_up():
    """
    Docstring for test_kl_term_does_not_pull_log_std_up:

    The aim keeps a tight spread; the mean-only KL to a WIDE-spread teacher
    must not raise log_std (that would undo the tight aim). Train a few updates with a
    strong KL coef toward a teacher whose log_std is wide and check the learner's
    log_std does not climb toward the teacher's.
    """
    ac, pi_opt, vf_opt = make_ac_and_opts(pi_lr=1e-3)
    teacher, _, _ = make_ac_and_opts()
    with torch.no_grad():
        teacher.log_std.fill_(-0.5)   # sigma ~0.6, much wider than the learner's 0.2
        for p in teacher.pi_net.parameters():
            p.add_(0.2 * torch.randn_like(p))
    std_before = float(ac.log_std.detach().mean())
    data = fake_data(512, ac)
    for _ in range(5):
        update_from_data(ac, data, pi_opt, vf_opt, n_passes=4, target_kl=1.0,
                         kl_teacher=teacher, kl_coef=1.0)
    std_after = float(ac.log_std.detach().mean())
    # the learner's log_std stays near its start (not dragged toward teacher's -0.5)
    assert std_after < std_before + 0.1


def test_kl_teacher_term_reported():
    ac, pi_opt, vf_opt = make_ac_and_opts()
    teacher, _, _ = make_ac_and_opts()
    with torch.no_grad():
        for p in teacher.pi_net.parameters():
            p.add_(0.3 * torch.randn_like(p))
    data = fake_data(300, ac)
    info = update_from_data(ac, data, pi_opt, vf_opt, n_passes=3, target_kl=1.0,
                            kl_teacher=teacher, kl_coef=0.5)
    assert info['KLTeacher'] > 0.0
    # with no teacher it is 0
    info2 = update_from_data(ac, data, pi_opt, vf_opt, n_passes=1, target_kl=1.0)
    assert info2['KLTeacher'] == 0.0


# ---- KL fade weight (the maths the loop applies) ----

def kl_weight(epoch, epochs, kl_coef, fade_fraction):
    """
    Docstring for kl_weight:

    the exact expression ppo_train uses for the per-epoch KL-to-teacher weight.
    """
    fade_epochs = max(1, int(epochs * fade_fraction))
    return kl_coef * max(0.0, 1.0 - epoch / fade_epochs)


def test_kl_weight_starts_at_coef_and_fades_to_zero():
    epochs, coef, frac = 100, 1.0, 0.5
    assert kl_weight(0, epochs, coef, frac) == pytest.approx(1.0)        # full at start
    assert kl_weight(25, epochs, coef, frac) == pytest.approx(0.5)       # halfway through fade
    # at the fade point (epoch 50 of a 0.5 fade) and after, the weight is 0
    assert kl_weight(50, epochs, coef, frac) == pytest.approx(0.0)
    assert kl_weight(80, epochs, coef, frac) == pytest.approx(0.0)


def test_kl_weight_never_negative():
    for e in range(0, 200, 7):
        assert kl_weight(e, 100, 1.0, 0.5) >= 0.0


# ---- LayerNorm changes the save shape ----

def test_layernorm_changes_state_dict_keys():
    plain = Actor_critic_neural_network(obs_dim=14, act_dim=6, use_layernorm=False)
    ln = Actor_critic_neural_network(obs_dim=14, act_dim=6, use_layernorm=True)
    plain_keys = set(plain.state_dict().keys())
    ln_keys = set(ln.state_dict().keys())
    assert ln_keys != plain_keys
    # LayerNorm adds weight/bias params the plain net does not have
    assert any('weight' in k or 'bias' in k for k in (ln_keys - plain_keys))


def test_plain_save_refuses_to_load_into_layernorm_net(tmp_path):
    ac, pi_opt, vf_opt = make_ac_and_opts(use_layernorm=False)
    path = str(tmp_path / 'plain.pt')
    save_checkpoint(path, ac, pi_opt, vf_opt, epoch=0, lr_step=0)
    ln = Actor_critic_neural_network(obs_dim=14, act_dim=6, use_layernorm=True)
    with pytest.raises(RuntimeError):
        ln.load_state_dict(load_checkpoint(path)['model'])


def test_layernorm_net_round_trips(tmp_path):
    ac, pi_opt, vf_opt = make_ac_and_opts(use_layernorm=True)
    path = str(tmp_path / 'ln.pt')
    save_checkpoint(path, ac, pi_opt, vf_opt, epoch=2, lr_step=2)
    ln2 = Actor_critic_neural_network(obs_dim=14, act_dim=6, use_layernorm=True)
    ln2.load_state_dict(load_checkpoint(path)['model'])
    for k, v in ac.state_dict().items():
        assert torch.allclose(v, ln2.state_dict()[k])
    # a forward pass works
    with torch.no_grad():
        dist, value = ln2(torch.randn(4, 14))
    assert value.shape == (4,)
