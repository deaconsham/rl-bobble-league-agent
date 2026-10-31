"""
T2: evaluate.py against the mock Godot. builds an agent from a freshly saved
checkpoint (so the test does not depend on a trained file), then plays a few
games the way evaluate.main does: agent.act on the raw learner-frame obs,
env.step, count outcomes.
"""
import os

import numpy as np
import pytest
import torch
from torch.optim import Adam

from shared_memory_link import Shared_memory_link
from bobble_league_env import Bobble_league_env
from neural_network import Actor_critic_neural_network
from opponents import PolicyOpponent, load_state_dict
from evaluate import latest_checkpoint
from training_loop import save_checkpoint
from mock_godot import Mock_godot


def test_latest_checkpoint_picks_highest_epoch(tmp_path):
    folder = tmp_path / "checkpoints"
    folder.mkdir()
    for epoch in (0, 5, 10, 199):
        (folder / f"ppo_epoch_{epoch}.pt").write_bytes(b"x")
    chosen = latest_checkpoint(str(folder))
    assert chosen.endswith("ppo_epoch_199.pt")


def test_latest_checkpoint_raises_when_empty(tmp_path):
    folder = tmp_path / "empty"
    folder.mkdir()
    with pytest.raises(FileNotFoundError):
        latest_checkpoint(str(folder))


def test_evaluate_plays_a_few_games(tmp_path):
    # save a small random network the way training does (tagged action_format),
    # then load it the way evaluate does
    ckpt = str(tmp_path / "ppo_epoch_0.pt")
    net = Actor_critic_neural_network(obs_dim=14, act_dim=6)
    pi_opt = Adam(list(net.pi_net.parameters()) + [net.log_std], lr=3e-4)
    vf_opt = Adam(net.v_net.parameters(), lr=1e-3)
    save_checkpoint(ckpt, net, pi_opt, vf_opt, epoch=0, lr_step=0)

    agent = PolicyOpponent(load_state_dict(ckpt), deterministic=True, name=ckpt)

    path = str(tmp_path / "shm.dat")
    game = Mock_godot(path, goal_after=2, scorer=1)
    bridge = Shared_memory_link(path=path, create=False)
    env = Bobble_league_env(opponent=None, bridge=bridge, step_timeout=5.0, max_ep_len=50)

    results = {"win": 0, "loss": 0, "timeout": 0}
    try:
        obs, info = env.reset()
        for _ in range(3):  # three short games
            terminated = truncated = False
            while not (terminated or truncated):
                action = agent.act(env._raw_obs)
                assert action.shape == (6,)
                obs, r, terminated, truncated, step_info = env.step(action)
            scorer = step_info["scorer"] if terminated else 0
            outcome = "win" if scorer == 1 else "loss" if scorer == 2 else "timeout"
            results[outcome] += 1
            obs, info = env.reset()
    finally:
        env.close()
        game.stop()

    assert sum(results.values()) == 3
    assert results["win"] == 3  # mock always lets the learner score
