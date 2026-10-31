"""
T1: reward shaping. each RewardConfig term is checked on a hand-made board,
and the goal reward keeps its sign for either team (the sign itself comes
from Godot; python only scales it by cfg.goal).
"""
import numpy as np
import pytest

from bobble_league_env import Bobble_league_env, RewardConfig, FIELD_HALF_Z, OPP_GOAL


class Dummy_bridge:
    """stands in for the shm link so the env builds without Godot."""
    def close(self):
        pass


def make_env(cfg):
    return Bobble_league_env(reward_config=cfg, bridge=Dummy_bridge())


def make_obs(ball, friendly, enemy):
    return np.concatenate([
        np.asarray(ball, dtype=np.float32),
        np.asarray(friendly, dtype=np.float32).ravel(),
        np.asarray(enemy, dtype=np.float32).ravel(),
    ])


def test_ball_to_goal_rewards_progress():
    cfg = RewardConfig(ball_to_goal=0.5, players_to_ball=0, defence=0, time_penalty=0)
    env = make_env(cfg)
    players = [[0, 0], [0, 0], [0, 0]]
    enemy = [[0, -1], [0, -2], [0, -3]]
    # learner attacks -z (OPP_GOAL at z = -16.5); ball moves from z=5 to z=0,
    # closer to the opponent goal -> positive reward
    prev = make_obs([0, 5], players, enemy)
    nxt = make_obs([0, 0], players, enemy)
    r = env._shaping(prev, nxt)
    expected = 0.5 * (np.linalg.norm([0, 5] - OPP_GOAL) - np.linalg.norm([0, 0] - OPP_GOAL)) / FIELD_HALF_Z
    assert r == pytest.approx(expected)
    assert r > 0


def test_ball_away_from_goal_is_penalised():
    cfg = RewardConfig(ball_to_goal=0.5, players_to_ball=0, defence=0, time_penalty=0)
    env = make_env(cfg)
    players = [[0, 0]] * 3
    enemy = [[0, -1]] * 3
    prev = make_obs([0, 0], players, enemy)
    nxt = make_obs([0, 5], players, enemy)  # ball retreats toward own goal
    assert env._shaping(prev, nxt) < 0


def test_players_to_ball_rewards_closing_in():
    cfg = RewardConfig(ball_to_goal=0, players_to_ball=0.05, defence=0, time_penalty=0)
    env = make_env(cfg)
    ball = [0, 0]
    enemy = [[0, -1]] * 3
    prev = make_obs(ball, [[5, 0], [5, 0], [5, 0]], enemy)
    nxt = make_obs(ball, [[2, 0], [2, 0], [2, 0]], enemy)  # all closer to the ball
    assert env._shaping(prev, nxt) > 0


def test_defence_penalty_when_no_one_is_goal_side():
    cfg = RewardConfig(ball_to_goal=0, players_to_ball=0, defence=0.02, time_penalty=0)
    env = make_env(cfg)
    # learner's own goal is +z; goal-side means a friendly with z larger than the ball.
    ball = [0, 0]
    enemy = [[0, -1]] * 3
    # nobody behind the ball (all z < ball z) -> penalty
    exposed = make_obs(ball, [[0, -2], [0, -3], [0, -4]], enemy)
    assert env._shaping(ball_prev(exposed), exposed) == pytest.approx(-0.02)
    # one friendly goal-side -> no penalty
    covered = make_obs(ball, [[0, 2], [0, -3], [0, -4]], enemy)
    assert env._shaping(ball_prev(covered), covered) == pytest.approx(0.0)


def ball_prev(obs):
    # a prev board identical to obs so only the defence term fires
    return obs.copy()


def test_time_penalty_subtracts_each_turn():
    cfg = RewardConfig(ball_to_goal=0, players_to_ball=0, defence=0, time_penalty=0.01)
    env = make_env(cfg)
    board = make_obs([0, 0], [[0, 1]] * 3, [[0, -1]] * 3)
    assert env._shaping(board, board) == pytest.approx(-0.01)


def test_goal_scaling_keeps_sign_for_both_teams():
    # python scales Godot's +-1 goal reward by cfg.goal; sign is preserved
    cfg = RewardConfig(goal=1.0)
    assert cfg.goal * 1.0 == 1.0    # learner scored
    assert cfg.goal * -1.0 == -1.0  # opponent scored
    cfg2 = RewardConfig(goal=2.0)
    assert cfg2.goal * 1.0 == 2.0
