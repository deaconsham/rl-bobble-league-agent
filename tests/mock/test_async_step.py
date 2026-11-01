"""
T2 (Phase 9): async stepping in Vector_env.step. The games no longer run in a
fixed index order; each is collected the moment its turn is done. These tests give
the mock games very different turn lengths and check that:

  - a fast game is collected first (it finishes before a slow one),
  - every game still gets exactly one result per step, in its own slot, so a
    game's path stays in order and no turn is lost or reordered,
  - a game that crashes mid-turn is still cut (truncated) and restarted while the
    others keep going,
  - a game that scores (and resets) while others are still mid-turn does not
    disturb the others.
"""
import time

import numpy as np

from bobble_league_env import Vector_env
from mock_godot import Mock_launcher


def build(tmp_path, n_games=4, goal_after=20, step_delays=None, step_timeout=5.0):
    launcher = Mock_launcher(n_games, tmp_path, goal_after=goal_after,
                             step_delays=step_delays)
    launcher.launch_all()
    bridges = [launcher.wait_for_handshake(i) for i in range(n_games)]
    env = Vector_env(n_games, bridges=bridges, launcher=launcher,
                     opponents=[None] * n_games, step_timeout=step_timeout,
                     max_ep_len=100)
    return launcher, env


def test_fast_games_collected_before_slow_ones(tmp_path):
    # game 0 is slow (0.3 s a turn), the rest are instant. the slow game must not
    # block the fast ones: they are all collected within one step and in slot order.
    launcher, env = build(tmp_path, n_games=4, step_delays=[0.3, 0.0, 0.0, 0.0])
    try:
        env.reset()
        actions = np.zeros((4, 6), dtype=np.float32)

        start = time.time()
        obs, r, term, trunc, infos = env.step(actions)
        elapsed = time.time() - start

        # one result per game, in slot order, none restarted
        assert obs.shape == (4, 14)
        assert len(infos) == 4
        for g in range(4):
            assert infos[g]["restarted"] is False
        # all four advanced exactly one turn (no lost or doubled turn)
        assert all(env.ep_len[g] == 1 for g in range(4))
        # the step waited for the slow game (~0.3 s) but not 4x that
        assert elapsed < 1.0
    finally:
        env.close()


def test_no_turn_lost_over_many_steps(tmp_path):
    # different turn lengths across games; over many steps every game must tick the
    # SAME number of turns (one per step), proving no turn is lost or reordered.
    launcher, env = build(tmp_path, n_games=4, step_delays=[0.0, 0.05, 0.1, 0.02])
    try:
        env.reset()
        actions = np.zeros((4, 6), dtype=np.float32)
        steps = 5
        for _ in range(steps):
            obs, r, term, trunc, infos = env.step(actions)
            assert obs.shape == (4, 14)
            for g in range(4):
                assert infos[g]["restarted"] is False
        # goal_after is 20 so no game scored; each game ticked exactly `steps` turns
        assert all(env.ep_len[g] == steps for g in range(4))
    finally:
        env.close()


def test_crash_mid_turn_restarts_only_that_game(tmp_path):
    # all games slow so there is a real mid-turn window; crash game 2 during it.
    launcher, env = build(tmp_path, n_games=4, step_delays=[0.2, 0.2, 0.2, 0.2],
                          step_timeout=3.0)
    try:
        env.reset()
        actions = np.zeros((4, 6), dtype=np.float32)
        env.step(actions)  # one clean turn over all four

        launcher.kill_game(2)
        obs, r, term, trunc, infos = env.step(actions)

        assert trunc[2]
        assert infos[2]["restarted"] is True
        assert launcher.launch_counts[2] == 2
        assert env.restart_counts[2] == 1
        # the other three were untouched and each ticked their turn
        for g in [0, 1, 3]:
            assert infos[g]["restarted"] is False
            assert launcher.launch_counts[g] == 1

        # the run carries on cleanly after the restart
        obs, r, term, trunc, infos = env.step(actions)
        assert obs.shape == (4, 14)
    finally:
        env.close()


def test_game_resets_while_others_mid_turn(tmp_path):
    # game 0 scores on turn 1 (and resets); the other three are slow and still
    # mid-turn when it finishes. the reset of game 0 must not disturb the others.
    launcher = Mock_launcher(4, tmp_path, goal_after=20,
                             step_delays=[0.0, 0.25, 0.25, 0.25])
    launcher.launch_all()
    from mock_godot import Mock_godot
    # swap game 0 for one that scores after a single turn, instant
    launcher.stop_one(0)
    launcher.games[0] = Mock_godot(launcher.shm_paths[0], goal_after=1, seed=0,
                                   step_delay=0.0)
    bridges = [launcher.wait_for_handshake(i) for i in range(4)]
    env = Vector_env(4, bridges=bridges, launcher=launcher,
                     opponents=[None] * 4, step_timeout=5.0, max_ep_len=100)
    try:
        env.reset()
        actions = np.zeros((4, 6), dtype=np.float32)
        obs, r, term, trunc, infos = env.step(actions)

        # game 0 scored and reset itself; its ep_len is back to 0
        assert term[0]
        assert env.ep_len[0] == 0
        # the slow games were not terminated and each advanced exactly one turn
        assert not term[1:].any()
        assert all(env.ep_len[g] == 1 for g in [1, 2, 3])
        for g in range(4):
            assert infos[g]["restarted"] is False
    finally:
        env.close()
