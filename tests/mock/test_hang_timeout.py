"""
T2: a hung game times out and restarts. game 2's mock is told to hang, so its
step flag never clears; Vector_env hits the step timeout, cuts that path
(truncated, restarted=True), relaunches the game fresh (not hung), and carries
on.
"""
import numpy as np

from bobble_league_env import Vector_env
from mock_godot import Mock_launcher


def test_hung_game_times_out_then_restarts(tmp_path):
    launcher = Mock_launcher(6, tmp_path, goal_after=20)
    launcher.launch_all()
    bridges = [launcher.wait_for_handshake(i) for i in range(6)]
    # short timeout so the test is quick
    env = Vector_env(6, bridges=bridges, launcher=launcher,
                     opponents=[None] * 6, step_timeout=1.0, max_ep_len=100)
    try:
        env.reset()
        actions = np.zeros((6, 6), dtype=np.float32)

        # make game 2 hang from now on
        launcher.games[2].set_hang(True)

        obs, r, term, trunc, infos = env.step(actions)
        assert trunc[2]
        assert infos[2]["restarted"] is True
        assert launcher.launch_counts[2] == 2  # relaunched (fresh, not hung)
        assert env.restart_counts[2] == 1

        # the relaunched game is healthy: the next step clears normally
        obs, r, term, trunc, infos = env.step(actions)
        assert infos[2]["restarted"] is False
    finally:
        env.close()
