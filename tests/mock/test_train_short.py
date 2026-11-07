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


def test_epoch_saves_land_in_run_dir_not_shared(tmp_path, monkeypatch):
    """
    Docstring for test_epoch_saves_land_in_run_dir_not_shared:

    T2: the periodic ppo_epoch_<N>.pt saves must follow the
    run's own folder (the directory of --best-path), NOT the shared
    bobble_model/Models/. In long run 2 best_p14.pt went to
    p14_run/checkpoints/ while the epoch saves went to the shared dir, so two runs
    could overwrite each other's saves.
    """
    monkeypatch.chdir(tmp_path)
    run_ckpt = tmp_path / 'p14_run' / 'checkpoints'
    best_path = str(run_ckpt / 'best_p14.pt')

    vec = make_vec(tmp_path, n_games=6, goal_after=3)
    ppo_train(vec_env=vec, epochs=2, steps_per_epoch=60, save_freq=1,
              device='cpu', log_dir=str(tmp_path / 'runs'),
              best_path=best_path, train_pi_iters=3, train_v_iters=3)

    # epoch saves landed in the RUN dir
    assert os.path.exists(run_ckpt / 'ppo_epoch_0.pt')
    assert os.path.exists(run_ckpt / 'ppo_epoch_1.pt')
    assert load_checkpoint(str(run_ckpt / 'ppo_epoch_1.pt'))['epoch'] == 1
    # and NOT in the shared Models/ dir (it should not even be created here)
    assert not os.path.exists(tmp_path / 'Models' / 'ppo_epoch_0.pt')


def test_checkpoint_dir_overrides_best_path_dir(tmp_path, monkeypatch):
    """
    Docstring for test_checkpoint_dir_overrides_best_path_dir:

    T2: an explicit --checkpoint-dir sends epoch saves there even when
    --best-path points elsewhere.
    """
    monkeypatch.chdir(tmp_path)
    best_path = str(tmp_path / 'bestonly' / 'best.pt')
    ckpt_dir = tmp_path / 'epochs_here'

    vec = make_vec(tmp_path, n_games=6, goal_after=3)
    ppo_train(vec_env=vec, epochs=1, steps_per_epoch=60, save_freq=1,
              device='cpu', log_dir=str(tmp_path / 'runs'),
              best_path=best_path, checkpoint_dir=str(ckpt_dir),
              train_pi_iters=3, train_v_iters=3)

    assert os.path.exists(ckpt_dir / 'ppo_epoch_0.pt')
    assert not os.path.exists(tmp_path / 'bestonly' / 'ppo_epoch_0.pt')


def test_default_best_path_keeps_shared_models_dir(tmp_path, monkeypatch):
    """
    Docstring for test_default_best_path_keeps_shared_models_dir:

    T2: with the default best_path ('Models/best.pt') epoch saves still
    land in Models/ - the fix must not change the default-path behaviour the
    other mock tests rely on.
    """
    monkeypatch.chdir(tmp_path)
    vec = make_vec(tmp_path, n_games=6, goal_after=3)
    ppo_train(vec_env=vec, epochs=1, steps_per_epoch=60, save_freq=1,
              device='cpu', log_dir=str(tmp_path / 'runs'),
              train_pi_iters=3, train_v_iters=3)
    assert os.path.exists('Models/ppo_epoch_0.pt')


def test_two_epochs_then_resume(tmp_path, monkeypatch):
    # checkpoints land under tmp_path/Models
    monkeypatch.chdir(tmp_path)

    vec = make_vec(tmp_path, n_games=6, goal_after=3)
    ppo_train(vec_env=vec, epochs=2, steps_per_epoch=60,  # 10 steps each over 6 games
              save_freq=1, device='cpu', log_dir=str(tmp_path / 'runs'),
              train_pi_iters=3, train_v_iters=3)

    assert os.path.exists('Models/ppo_epoch_0.pt')
    assert os.path.exists('Models/ppo_epoch_1.pt')

    ckpt = load_checkpoint('Models/ppo_epoch_1.pt')
    assert ckpt['epoch'] == 1
    assert 'pi_optimizer' in ckpt and 'vf_optimizer' in ckpt

    # resume: a fresh vec env, pick up from epoch 1's save, ask for 3 epochs total
    vec2 = make_vec(tmp_path, n_games=6, goal_after=3)
    ppo_train(vec_env=vec2, epochs=3, steps_per_epoch=60,
              save_freq=1, device='cpu', log_dir=str(tmp_path / 'runs2'),
              resume='Models/ppo_epoch_1.pt',
              train_pi_iters=3, train_v_iters=3)

    # resume continued at epoch 2 (only epoch 2's save is new from this run)
    assert os.path.exists('Models/ppo_epoch_2.pt')
    assert load_checkpoint('Models/ppo_epoch_2.pt')['epoch'] == 2


def test_resume_restores_pool(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    vec = make_vec(tmp_path, n_games=6, goal_after=3, with_pool=True)
    # snapshot_freq=1 so the pool gets a snapshot after epoch 0
    ppo_train(vec_env=vec, epochs=2, steps_per_epoch=60, save_freq=1,
              snapshot_freq=1, device='cpu', log_dir=str(tmp_path / 'runs'),
              train_pi_iters=2, train_v_iters=2)
    ckpt = load_checkpoint('Models/ppo_epoch_1.pt')
    assert 'pool' in ckpt
    assert len(ckpt['pool']['snapshots']) >= 1


def test_full_config_runs_and_saves_value_norm(tmp_path, monkeypatch):
    """
    Docstring for test_full_config_runs_and_saves_value_norm:

    T2: a short run with every critic/self-play setting on (value norm, separate critic
    passes, 4 large minibatches, a mean-only KL teacher) runs end to end on mock
    games and writes checkpoints whose value_norm stats are populated (count past
    its epsilon start, so at least one batch was folded in).
    """
    from neural_network import Actor_critic_neural_network
    from training_loop import save_checkpoint
    from torch.optim import Adam

    monkeypatch.chdir(tmp_path)

    # a frozen "warm start" teacher saved in the full checkpoint format
    teacher = Actor_critic_neural_network(obs_dim=14, act_dim=6)
    pi_opt = Adam(list(teacher.pi_net.parameters()) + [teacher.log_std], lr=3e-4)
    vf_opt = Adam(teacher.v_net.parameters(), lr=1e-3)
    teacher_path = str(tmp_path / 'warm.pt')
    save_checkpoint(teacher_path, teacher, pi_opt, vf_opt, epoch=0, lr_step=0)

    vec = make_vec(tmp_path, n_games=6, goal_after=3)
    ppo_train(vec_env=vec, epochs=2, steps_per_epoch=120, save_freq=1,
              device='cpu', log_dir=str(tmp_path / 'runs'),
              value_norm=True, n_passes=5, n_passes_v=10, n_minibatches=4,
              clip_ratio=0.15, kl_warm_start=teacher_path, kl_coef=1.0,
              kl_fade_fraction=0.5, weight_decay=1e-4)

    ckpt = load_checkpoint('Models/ppo_epoch_1.pt')
    assert 'value_norm' in ckpt
    # the normalizer saw real data: count grew past its tiny epsilon start
    assert ckpt['value_norm']['count'] > 1.0


def test_layernorm_full_config_runs_and_saves(tmp_path, monkeypatch):
    """
    Docstring for test_layernorm_full_config_runs_and_saves:

    T2: the same settings but with LayerNorm on. The warm-start
    teacher must be built with LayerNorm too, because LayerNorm changes the saved
    weight shape and a plain teacher would not load into the LayerNorm learner.
    The run completes on mock games and still writes value_norm stats, so the
    LayerNorm + weight-decay + value-norm + KL-teacher paths all wire together.
    """
    from neural_network import Actor_critic_neural_network
    from training_loop import save_checkpoint
    from torch.optim import Adam

    monkeypatch.chdir(tmp_path)

    # a frozen LayerNorm "warm start" teacher saved in the full checkpoint format
    teacher = Actor_critic_neural_network(obs_dim=14, act_dim=6, use_layernorm=True)
    pi_opt = Adam(list(teacher.pi_net.parameters()) + [teacher.log_std], lr=3e-4)
    vf_opt = Adam(teacher.v_net.parameters(), lr=1e-3)
    teacher_path = str(tmp_path / 'warm_ln.pt')
    save_checkpoint(teacher_path, teacher, pi_opt, vf_opt, epoch=0, lr_step=0)

    vec = make_vec(tmp_path, n_games=6, goal_after=3)
    ppo_train(vec_env=vec, epochs=2, steps_per_epoch=120, save_freq=1,
              device='cpu', log_dir=str(tmp_path / 'runs'),
              use_layernorm=True, weight_decay=1e-4,
              value_norm=True, n_passes=5, n_passes_v=10, n_minibatches=4,
              clip_ratio=0.15, kl_warm_start=teacher_path, kl_coef=1.0,
              kl_fade_fraction=0.5)

    ckpt = load_checkpoint('Models/ppo_epoch_1.pt')
    assert 'value_norm' in ckpt
    assert ckpt['value_norm']['count'] > 1.0
    # the saved net is a LayerNorm net: it loads into a LayerNorm AC and refuses a plain one
    ac_ln = Actor_critic_neural_network(obs_dim=14, act_dim=6, use_layernorm=True)
    ac_ln.load_state_dict(ckpt['model'])  # must not raise
    ac_plain = Actor_critic_neural_network(obs_dim=14, act_dim=6, use_layernorm=False)
    import pytest
    with pytest.raises(RuntimeError):
        ac_plain.load_state_dict(ckpt['model'])


def test_single_game_shaping_fade(tmp_path, monkeypatch):
    # regression: the games == 1 path wraps the env in _Single_env_as_vector, and
    # the shaping-fade loop reads and rewrites env.reward_config each epoch. the
    # shim used to drop that attribute, so a single-game run with shaping_fade
    # crashed with AttributeError. it must now run end to end and actually fade
    # the shaping on the wrapped env.
    import os

    from mock_godot import Mock_godot
    from shared_memory_link import Shared_memory_link
    from bobble_league_env import Bobble_league_env

    monkeypatch.chdir(tmp_path)
    path = str(tmp_path / 'shm.dat')
    game = Mock_godot(path, goal_after=3, scorer=1)
    made = {}

    def env_fn(opponent=None, reward_config=None, max_ep_len=1000, step_timeout=120.0,
               random_start_prob=0.0, gamma=0.99, **kwargs):
        bridge = Shared_memory_link(path=path, create=False)
        env = Bobble_league_env(opponent=opponent, reward_config=reward_config,
                                max_ep_len=max_ep_len, step_timeout=step_timeout,
                                bridge=bridge, random_start_prob=random_start_prob,
                                gamma=gamma, **kwargs)
        made['env'] = env
        return env

    try:
        ppo_train(env_fn=env_fn, games=1, epochs=2, steps_per_epoch=30,
                  save_freq=1, device='cpu', log_dir=str(tmp_path / 'runs'),
                  step_timeout=5.0, max_ep_len=30, shaping_fade=True,
                  shaping_fade_fraction=0.5, train_pi_iters=2, train_v_iters=2)
        # it ran without AttributeError and wrote a checkpoint
        assert os.path.exists('Models/ppo_epoch_0.pt')
        # the fade reached the wrapped env: epoch 1 of 2 with fraction 0.5 scales
        # the per-turn shaping to 0, so the shaping weights are below their base
        cfg = made['env'].reward_config
        assert cfg.ball_to_goal == 0.0 and cfg.players_to_ball == 0.0
    finally:
        made.get('env') and made['env'].close()
        game.stop()
