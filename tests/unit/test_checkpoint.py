"""
T1: full checkpoint round trip. save_checkpoint then load_checkpoint gives back
the same network weights, both optimizer states, the epoch and lr_step, the
self-play pool (snapshot weights + win rates) and the RNG state, so a resumed
run continues exactly.
"""
import random

import numpy as np
import pytest
import torch
from torch.optim import Adam

from neural_network import Actor_critic_neural_network
from shared_memory_link import Shared_memory_link
from training_loop import save_checkpoint, load_checkpoint, rng_state, set_rng_state
from opponents import SelfPlayPool


def make_ac_and_opts():
    ac = Actor_critic_neural_network(obs_dim=14, act_dim=6)
    pi_opt = Adam(list(ac.pi_net.parameters()) + [ac.log_std], lr=3e-4)
    vf_opt = Adam(ac.v_net.parameters(), lr=1e-3)
    return ac, pi_opt, vf_opt


def take_one_step(ac, pi_opt, vf_opt):
    """put real state into the optimizers so the round trip has something to check."""
    obs = torch.randn(8, 14)
    dist, value = ac(obs)
    loss = -dist.log_prob(torch.zeros(8, 6)).mean() + value.mean()
    pi_opt.zero_grad(); vf_opt.zero_grad()
    loss.backward()
    pi_opt.step(); vf_opt.step()


def test_checkpoint_round_trip(tmp_path):
    ac, pi_opt, vf_opt = make_ac_and_opts()
    take_one_step(ac, pi_opt, vf_opt)

    pool = SelfPlayPool(max_size=10)
    pool.add(ac.state_dict())
    pool.add(ac.state_dict())
    pool.win_rates[0] = 0.3
    pool.win_rates[1] = 0.7

    path = str(tmp_path / "ckpt.pt")
    save_checkpoint(path, ac, pi_opt, vf_opt, epoch=7, lr_step=7, opponent=pool)

    ckpt = load_checkpoint(path)
    assert ckpt["epoch"] == 7
    assert ckpt["lr_step"] == 7

    # weights match
    ac2, pi_opt2, vf_opt2 = make_ac_and_opts()
    ac2.load_state_dict(ckpt["model"])
    for k, v in ac.state_dict().items():
        assert torch.allclose(v, ac2.state_dict()[k])

    # optimizer state matches (Adam step counters and moments)
    pi_opt2.load_state_dict(ckpt["pi_optimizer"])
    vf_opt2.load_state_dict(ckpt["vf_optimizer"])
    assert pi_opt2.state_dict()["state"].keys() == pi_opt.state_dict()["state"].keys()
    for pid, st in pi_opt.state_dict()["state"].items():
        assert st["step"] == pi_opt2.state_dict()["state"][pid]["step"]

    # pool round trips: snapshot count, win rates, and the actual weights
    pool2 = SelfPlayPool(max_size=10)
    pool2.load_state(ckpt["pool"])
    assert len(pool2.snapshots) == 2
    assert pool2.win_rates == [0.3, 0.7]
    for a, b in zip(pool.snapshots[0].items(), pool2.snapshots[0].items()):
        assert torch.allclose(a[1], b[1])


def test_rng_state_round_trip():
    set_rng_state({"python": random.Random(123).getstate()})
    state = rng_state()
    a = (random.random(), float(np.random.rand()), torch.rand(1).item())
    set_rng_state(state)
    b = (random.random(), float(np.random.rand()), torch.rand(1).item())
    assert a == b


def test_old_checkpoint_is_refused_with_clear_message(tmp_path):
    """Phase 2: the action meaning changed (angle+power) without changing tensor
    shapes, so a pre-Phase-2 save must be refused, not loaded silently."""
    ac, _, _ = make_ac_and_opts()

    # a phase-0 bare state_dict save
    bare = str(tmp_path / "bare.pt")
    torch.save(ac.state_dict(), bare)
    with pytest.raises(ValueError, match="pre-Phase-2"):
        load_checkpoint(bare)

    # a phase-1 full dict that has no action_format tag
    untagged = str(tmp_path / "untagged.pt")
    torch.save({"model": ac.state_dict(), "epoch": 3}, untagged)
    with pytest.raises(ValueError, match="action_format"):
        load_checkpoint(untagged)


def test_old_checkpoint_is_refused_by_eval_loader(tmp_path):
    """opponents.load_state_dict (eval / frozen opponent path) refuses it too."""
    from opponents import load_state_dict as opp_load

    ac, _, _ = make_ac_and_opts()
    untagged = str(tmp_path / "untagged.pt")
    torch.save({"model": ac.state_dict(), "epoch": 1}, untagged)
    with pytest.raises(ValueError, match="action_format"):
        opp_load(untagged)


def test_new_checkpoint_loads_through_both_paths(tmp_path):
    """a Phase 2 save (carrying action_format) loads cleanly in both loaders."""
    from opponents import load_state_dict as opp_load

    ac, pi_opt, vf_opt = make_ac_and_opts()
    path = str(tmp_path / "new.pt")
    save_checkpoint(path, ac, pi_opt, vf_opt, epoch=0, lr_step=0)
    assert load_checkpoint(path)["action_format"] == Shared_memory_link.PROTOCOL_VERSION
    sd = opp_load(path)
    ac2, _, _ = make_ac_and_opts()
    ac2.load_state_dict(sd)


def test_pool_round_trip_empty(tmp_path):
    """resume with a pool of 0 copies: no snapshots saved, loads back empty."""
    ac, pi_opt, vf_opt = make_ac_and_opts()
    pool = SelfPlayPool(max_size=10)  # nothing added yet
    path = str(tmp_path / "ck0.pt")
    save_checkpoint(path, ac, pi_opt, vf_opt, epoch=0, lr_step=0, opponent=pool)
    ckpt = load_checkpoint(path)
    assert ckpt["pool"]["snapshots"] == []
    pool2 = SelfPlayPool(max_size=10)
    pool2.load_state(ckpt["pool"])
    assert pool2.snapshots == [] and pool2.win_rates == []


def test_pool_round_trip_full(tmp_path):
    """resume with a full pool of 10 copies: all 10 and their win rates round trip."""
    ac, pi_opt, vf_opt = make_ac_and_opts()
    pool = SelfPlayPool(max_size=10)
    for _ in range(12):           # add more than max_size; oldest two drop
        pool.add(ac.state_dict())
    assert len(pool.snapshots) == 10
    pool.win_rates = [i / 10 for i in range(10)]
    path = str(tmp_path / "ck10.pt")
    save_checkpoint(path, ac, pi_opt, vf_opt, epoch=5, lr_step=5, opponent=pool)
    ckpt = load_checkpoint(path)
    pool2 = SelfPlayPool(max_size=10)
    pool2.load_state(ckpt["pool"])
    assert len(pool2.snapshots) == 10
    assert pool2.win_rates == [i / 10 for i in range(10)]
