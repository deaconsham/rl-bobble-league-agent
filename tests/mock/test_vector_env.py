"""
T2: Vector_env over 6 mock games. they step together on one batched action, each
keeps its own shm file, and a game that scores resets on its own without
disturbing the others.
"""
import numpy as np

from bobble_league_env import Vector_env
from mock_godot import Mock_launcher


def build(tmp_path, n_games=6, goal_after=3):
    launcher = Mock_launcher(n_games, tmp_path, goal_after=goal_after)
    launcher.launch_all()
    bridges = [launcher.wait_for_handshake(i) for i in range(n_games)]
    env = Vector_env(n_games, bridges=bridges, launcher=launcher,
                     opponents=[None] * n_games, step_timeout=5.0, max_ep_len=50)
    return launcher, env


def test_six_games_step_together(tmp_path):
    launcher, env = build(tmp_path, n_games=6)
    try:
        obs, infos = env.reset()
        assert obs.shape == (6, 14)
        assert len(infos) == 6

        actions = np.zeros((6, 6), dtype=np.float32)
        obs, r, term, trunc, infos = env.step(actions)
        assert obs.shape == (6, 14)
        assert r.shape == (6,)
        assert term.shape == (6,) and trunc.shape == (6,)
        # no goal yet (goal_after=3, this is turn 1)
        assert not term.any()
    finally:
        env.close()


def test_each_game_keeps_its_own_file(tmp_path):
    launcher, env = build(tmp_path, n_games=6)
    try:
        paths = set(launcher.shm_paths)
        assert len(paths) == 6  # all distinct
        for p in launcher.shm_paths:
            import os
            assert os.path.exists(p)
    finally:
        env.close()


def test_done_game_resets_alone(tmp_path):
    # game 0 scores after 1 turn, the rest after 10, so only game 0 terminates first
    launcher = Mock_launcher(6, tmp_path, goal_after=10)
    launcher.launch_all()
    # make game 0 score quickly by swapping its mock
    launcher.stop_one(0)
    from mock_godot import Mock_godot
    launcher.games[0] = Mock_godot(launcher.shm_paths[0], goal_after=1, seed=0)
    bridges = [launcher.wait_for_handshake(i) for i in range(6)]
    env = Vector_env(6, bridges=bridges, launcher=launcher,
                     opponents=[None] * 6, step_timeout=5.0, max_ep_len=50)
    try:
        env.reset()
        actions = np.zeros((6, 6), dtype=np.float32)
        obs, r, term, trunc, infos = env.step(actions)
        # game 0 scored and terminated; it reset (obs back, ep_len reset), others did not
        assert term[0]
        assert not term[1:].any()
        assert env.ep_len[0] == 0      # reset after the goal
        assert all(env.ep_len[g] == 1 for g in range(1, 6))
    finally:
        env.close()
