"""
T2: a crashed game is restarted on its own. kill 1 of 6 mid-epoch; the step that
hits it comes back truncated with restarted=True (so the training loop bootstraps
the cut path from the critic), that one game relaunches, and the other 5 keep
going untouched.
"""
import numpy as np

from bobble_league_env import Vector_env
from mock_godot import Mock_launcher


def build(tmp_path, n_games=6, goal_after=20):
    launcher = Mock_launcher(n_games, tmp_path, goal_after=goal_after)
    launcher.launch_all()
    bridges = [launcher.wait_for_handshake(i) for i in range(n_games)]
    env = Vector_env(n_games, bridges=bridges, launcher=launcher,
                     opponents=[None] * n_games, step_timeout=3.0, max_ep_len=100)
    return launcher, env


def test_killing_one_restarts_only_that_game(tmp_path):
    launcher, env = build(tmp_path, n_games=6)
    try:
        env.reset()
        actions = np.zeros((6, 6), dtype=np.float32)
        env.step(actions)  # one clean turn

        # crash game 3
        launcher.kill_game(3)

        obs, r, term, trunc, infos = env.step(actions)
        # game 3 was cut as truncated and restarted
        assert trunc[3]
        assert infos[3]['restarted'] is True
        assert launcher.launch_counts[3] == 2  # launched once, relaunched once
        assert env.restart_counts[3] == 1
        # the others were not restarted and kept stepping
        for g in [0, 1, 2, 4, 5]:
            assert infos[g]['restarted'] is False
            assert launcher.launch_counts[g] == 1

        # after the restart the run carries on: another clean step over all 6
        obs, r, term, trunc, infos = env.step(actions)
        assert obs.shape == (6, 14)
    finally:
        env.close()


def test_killing_two_restarts_both(tmp_path):
    """
    Docstring for test_killing_two_restarts_both:

    two games crash in the same turn: both are cut + restarted, the other four keep going.
    """
    launcher, env = build(tmp_path, n_games=6)
    try:
        env.reset()
        actions = np.zeros((6, 6), dtype=np.float32)
        env.step(actions)

        launcher.kill_game(1)
        launcher.kill_game(4)

        obs, r, term, trunc, infos = env.step(actions)
        assert trunc[1] and trunc[4]
        assert infos[1]['restarted'] and infos[4]['restarted']
        assert launcher.launch_counts[1] == 2 and launcher.launch_counts[4] == 2
        for g in [0, 2, 3, 5]:
            assert infos[g]['restarted'] is False
            assert launcher.launch_counts[g] == 1
    finally:
        env.close()
