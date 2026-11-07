"""
T2: live-vs-live data and turn length through the Vector_env.

Drives mock games with a self-play pool pinned to the live slot and checks that
every step reports a turn length, and that a live-vs-live game hands back Team B's
transition (obs/act/value/log-prob/reward) in its own mirrored frame so the
training loop can store it. A non-live game reports no opponent transition.
"""
import numpy as np

from bobble_league_env import Vector_env, opponent_view
from neural_network import Actor_critic_neural_network
from opponents import SelfPlayPool
from mock_godot import Mock_launcher


def _live_pool(seed, live_prob):
    net = Actor_critic_neural_network(obs_dim=14, act_dim=6, hidden=[32, 32])
    net.eval()
    pool = SelfPlayPool(scripted_prob=0.0, live_prob=live_prob, hidden=(32, 32), seed=seed)
    pool.attach_live(net)
    # one snapshot so the frozen path exists too
    pool.add(net.state_dict())
    return pool


def test_turn_seconds_reported_every_step(tmp_path):
    launcher = Mock_launcher(2, tmp_path, goal_after=5)
    launcher.launch_all()
    bridges = [launcher.wait_for_handshake(i) for i in range(2)]
    pools = [_live_pool(i, live_prob=1.0) for i in range(2)]
    env = Vector_env(2, bridges=bridges, launcher=launcher, opponents=pools,
                     step_timeout=5.0)
    try:
        env.reset()
        actions = np.zeros((2, 6), dtype=np.float32)
        _, _, _, _, infos = env.step(actions)
        for info in infos:
            assert 'turn_seconds' in info
            assert info['turn_seconds'] >= 0.0
    finally:
        env.close()


def test_live_game_hands_back_team_b_transition(tmp_path):
    launcher = Mock_launcher(1, tmp_path, goal_after=5)
    launcher.launch_all()
    bridges = [launcher.wait_for_handshake(0)]
    pool = _live_pool(0, live_prob=1.0)  # always live
    env = Vector_env(1, bridges=bridges, launcher=launcher, opponents=[pool],
                     step_timeout=5.0)
    try:
        env.reset()
        assert pool.is_live_current() is True
        _, _, _, _, infos = env.step(np.zeros((1, 6), dtype=np.float32))
        opp = infos[0].get('opponent')
        assert opp is not None, "a live-vs-live game must report Team B's turn"
        assert np.asarray(opp['obs']).shape == (14,)
        assert np.asarray(opp['act']).shape == (6,)
        assert np.isfinite(opp['val'])
        assert np.isfinite(opp['logp'])
        assert np.isfinite(opp['reward'])
    finally:
        env.close()


def test_non_live_game_reports_no_opponent_transition(tmp_path):
    launcher = Mock_launcher(1, tmp_path, goal_after=5)
    launcher.launch_all()
    bridges = [launcher.wait_for_handshake(0)]
    # chaser-only pool: never the live slot, so no Team B data is stored
    pool = SelfPlayPool(scripted_prob=1.0, live_prob=0.0, hidden=(32, 32), seed=0)
    net = Actor_critic_neural_network(obs_dim=14, act_dim=6, hidden=[32, 32])
    pool.attach_live(net)
    env = Vector_env(1, bridges=bridges, launcher=launcher, opponents=[pool],
                     step_timeout=5.0)
    try:
        env.reset()
        assert pool.is_live_current() is False
        _, _, _, _, infos = env.step(np.zeros((1, 6), dtype=np.float32))
        assert infos[0].get('opponent') is None
    finally:
        env.close()


def test_pfsp_win_rate_recorded_through_the_env(tmp_path):
    # a frozen-copy game (no live, no chaser) scores after a few turns; the env
    # should fold that outcome into the drawn copy's win rate for PFSP.
    launcher = Mock_launcher(1, tmp_path, goal_after=3)
    launcher.launch_all()
    bridges = [launcher.wait_for_handshake(0)]
    net = Actor_critic_neural_network(obs_dim=14, act_dim=6, hidden=[32, 32])
    net.eval()
    pool = SelfPlayPool(scripted_prob=0.0, live_prob=0.0, hidden=(32, 32),
                        pfsp=True, seed=0)
    pool.add(net.state_dict())  # one frozen copy
    env = Vector_env(1, bridges=bridges, launcher=launcher, opponents=[pool],
                     step_timeout=5.0)
    try:
        env.reset()
        assert pool.name == 'self'      # drew the frozen copy
        assert pool.last_index == 0
        before = list(pool.win_rates)
        # step until the mock scores (goal_after=3) so a result is recorded
        for _ in range(5):
            _, _, term, trunc, _ = env.step(np.zeros((1, 6), dtype=np.float32))
            if term[0] or trunc[0]:
                break
        # the win rate for copy 0 moved off its 0.5 start
        assert pool.win_rates[0] != before[0]
    finally:
        env.close()
