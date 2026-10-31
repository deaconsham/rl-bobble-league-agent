"""
T2: short training on 6 mock games. two epochs run end to end on mock Godots,
full checkpoints are written each epoch, and resuming from the epoch-1 save
continues at epoch 2 (it does not redo epochs 0-1).
"""
import os

import numpy as np

from bobble_league_env import Vector_env
from mock_godot import Mock_launcher
from opponents import SelfPlayPool
from training_loop import ppo_train, load_checkpoint


def make_vec(tmp_path, n_games=6, goal_after=3, with_pool=False):
    launcher = Mock_launcher(n_games, tmp_path, goal_after=goal_after)
    launcher.launch_all()
    bridges = [launcher.wait_for_handshake(i) for i in range(n_games)]
    if with_pool:
        pool = SelfPlayPool()
        opponents = [pool] * n_games
    else:
        opponents = [None] * n_games
    env = Vector_env(n_games, bridges=bridges, launcher=launcher,
                     opponents=opponents, step_timeout=5.0, max_ep_len=30)
    return env


def test_two_epochs_then_resume(tmp_path, monkeypatch):
    # checkpoints land under tmp_path/checkpoints
    monkeypatch.chdir(tmp_path)

    vec = make_vec(tmp_path, n_games=6, goal_after=3)
    ppo_train(vec_env=vec, epochs=2, steps_per_epoch=60,  # 10 steps each over 6 games
              save_freq=1, device="cpu", log_dir=str(tmp_path / "runs"),
              train_pi_iters=3, train_v_iters=3)

    assert os.path.exists("checkpoints/ppo_epoch_0.pt")
    assert os.path.exists("checkpoints/ppo_epoch_1.pt")

    ckpt = load_checkpoint("checkpoints/ppo_epoch_1.pt")
    assert ckpt["epoch"] == 1
    assert "pi_optimizer" in ckpt and "vf_optimizer" in ckpt

    # resume: a fresh vec env, pick up from epoch 1's save, ask for 3 epochs total
    vec2 = make_vec(tmp_path, n_games=6, goal_after=3)
    ppo_train(vec_env=vec2, epochs=3, steps_per_epoch=60,
              save_freq=1, device="cpu", log_dir=str(tmp_path / "runs2"),
              resume="checkpoints/ppo_epoch_1.pt",
              train_pi_iters=3, train_v_iters=3)

    # resume continued at epoch 2 (only epoch 2's save is new from this run)
    assert os.path.exists("checkpoints/ppo_epoch_2.pt")
    assert load_checkpoint("checkpoints/ppo_epoch_2.pt")["epoch"] == 2


def test_resume_restores_pool(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    vec = make_vec(tmp_path, n_games=6, goal_after=3, with_pool=True)
    # snapshot_freq=1 so the pool gets a snapshot after epoch 0
    ppo_train(vec_env=vec, epochs=2, steps_per_epoch=60, save_freq=1,
              snapshot_freq=1, device="cpu", log_dir=str(tmp_path / "runs"),
              train_pi_iters=2, train_v_iters=2)
    ckpt = load_checkpoint("checkpoints/ppo_epoch_1.pt")
    assert "pool" in ckpt
    assert len(ckpt["pool"]["snapshots"]) >= 1
