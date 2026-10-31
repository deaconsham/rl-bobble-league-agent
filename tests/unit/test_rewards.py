"""
T1: reward shaping. each RewardConfig term is checked on a hand-made board,
and the goal reward keeps its sign for either team (the sign itself comes
from Godot; python only scales it by cfg.goal).
"""
import numpy as np
import pytest

from bobble_league_env import (
    Bobble_league_env,
    RewardConfig,
    FIELD_HALF_Z,
    OPP_GOAL,
    shaping_fade_fraction,
)


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


# ---- Phase 6: shaping fade schedule (R39) ----

def test_scaled_fades_shaping_but_keeps_the_goal_reward():
    cfg = RewardConfig(goal=1.0, ball_to_goal=0.5, players_to_ball=0.05,
                       defence=0.02, time_penalty=0.01)
    half = cfg.scaled(0.5)
    assert half.goal == 1.0                       # goal never fades
    assert half.ball_to_goal == pytest.approx(0.25)
    assert half.players_to_ball == pytest.approx(0.025)
    assert half.defence == pytest.approx(0.01)
    assert half.time_penalty == pytest.approx(0.005)


def test_scaled_to_zero_turns_off_all_shaping():
    cfg = RewardConfig(goal=1.0, ball_to_goal=0.5, players_to_ball=0.05,
                       defence=0.02, time_penalty=0.01)
    off = cfg.scaled(0.0)
    assert off.goal == 1.0
    assert off.ball_to_goal == 0.0
    assert off.players_to_ball == 0.0
    assert off.defence == 0.0
    assert off.time_penalty == 0.0


def test_fade_fraction_is_one_at_start_zero_at_halfway():
    total = 100
    # 1.0 at epoch 0
    assert shaping_fade_fraction(0, total, 0.5) == pytest.approx(1.0)
    # the halfway epoch of a 100-epoch run is index ~49.5; weight hits 0 there
    # and stays 0 after. with fade_end = 0.5*(100-1) = 49.5:
    assert shaping_fade_fraction(49, total, 0.5) == pytest.approx(1.0 - 49 / 49.5)
    assert shaping_fade_fraction(50, total, 0.5) == pytest.approx(0.0)
    assert shaping_fade_fraction(99, total, 0.5) == pytest.approx(0.0)


def test_fade_fraction_hits_zero_exactly_at_the_halfway_epoch():
    # a run whose halfway epoch is an integer: total 21 -> fade_end = 0.5*20 = 10
    total = 21
    assert shaping_fade_fraction(0, total, 0.5) == pytest.approx(1.0)
    assert shaping_fade_fraction(5, total, 0.5) == pytest.approx(0.5)
    assert shaping_fade_fraction(10, total, 0.5) == pytest.approx(0.0)
    assert shaping_fade_fraction(20, total, 0.5) == pytest.approx(0.0)


def test_fade_fraction_is_monotonic_non_increasing():
    total = 40
    vals = [shaping_fade_fraction(e, total, 0.5) for e in range(total)]
    for a, b in zip(vals, vals[1:]):
        assert b <= a + 1e-9


def test_fade_fraction_degenerate_runs():
    # fade_fraction 0 -> no shaping from the start
    assert shaping_fade_fraction(0, 100, 0.0) == 0.0
    # a one-epoch run has no room to fade -> 0
    assert shaping_fade_fraction(0, 1, 0.5) == 0.0
