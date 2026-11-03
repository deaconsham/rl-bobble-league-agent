"""
T1: reward shaping. Phase 12 makes the position terms exact potential-based
shaping F = gamma*phi(s') - phi(s), with phi=0 at a goal and the turn after an
out-of-bounds kickoff skipped. Each term is checked on a hand-made board, the
telescoping sum over a game is verified, and the goal reward keeps its sign for
either team (the sign itself comes from Godot; python only scales it by cfg.goal).
"""
import numpy as np
import pytest

from bobble_league_env import (
    Bobble_league_env,
    RewardConfig,
    FIELD_HALF_Z,
    OPP_GOAL,
    potential,
    legacy_shaping,
    shaping_fade_fraction,
)


class Dummy_bridge:
    """
    Docstring for Dummy_bridge:

    stands in for the shm link so the env builds without Godot.
    """
    def close(self):
        pass


def make_env(cfg, gamma=0.99, potential_shaping=True):
    # These tests verify the exact potential-based shaping (Phase 12, step 3),
    # which is OFF by default since Phase 12 was a T5 miss. Turn it ON here so the
    # PBRS contract stays tested; the legacy-default path has its own test below.
    return Bobble_league_env(reward_config=cfg, bridge=Dummy_bridge(), gamma=gamma,
                             potential_shaping=potential_shaping)


def make_obs(ball, friendly, enemy):
    return np.concatenate([
        np.asarray(ball, dtype=np.float32),
        np.asarray(friendly, dtype=np.float32).ravel(),
        np.asarray(enemy, dtype=np.float32).ravel(),
    ])


def test_default_env_uses_legacy_shaping_not_pbrs():
    """
    Docstring for test_default_env_uses_legacy_shaping_not_pbrs:

    A plain env (potential_shaping defaults off) must reproduce the Phase 11
    shaping, not the Phase 12 potential form. This is the behaviour the Phase 12
    'features default off' decision guarantees.
    """
    cfg = RewardConfig(ball_to_goal=0.5, players_to_ball=0.05, defence=0.02, time_penalty=0.0)
    env = Bobble_league_env(reward_config=cfg, bridge=Dummy_bridge(), gamma=0.99)
    assert env.potential_shaping is False
    prev = make_obs([0, 2], [[0, 3], [1, 4], [-1, 4]], [[0, -3], [1, -4], [-1, -4]])
    nxt = make_obs([0, 0], [[0, 1], [1, 2], [-1, 2]], [[0, -3], [1, -4], [-1, -4]])
    # the env's shaping equals the legacy helper, and differs from exact PBRS
    assert env._shaping(prev, nxt) == pytest.approx(legacy_shaping(prev, nxt, cfg))
    pbrs = 0.99 * potential(nxt, cfg) - potential(prev, cfg)
    assert env._shaping(prev, nxt) != pytest.approx(pbrs)


def test_ball_to_goal_rewards_progress():
    cfg = RewardConfig(ball_to_goal=0.5, players_to_ball=0, defence=0, time_penalty=0)
    gamma = 0.99
    env = make_env(cfg, gamma=gamma)
    players = [[0, 0], [0, 0], [0, 0]]
    enemy = [[0, -1], [0, -2], [0, -3]]
    # learner attacks -z (OPP_GOAL at z = -16.5); ball moves from z=5 to z=0,
    # closer to the opponent goal -> positive shaping
    prev = make_obs([0, 5], players, enemy)
    nxt = make_obs([0, 0], players, enemy)
    r = env._shaping(prev, nxt)
    # exact PBRS: gamma*phi(next) - phi(prev)
    expected = gamma * potential(nxt, cfg) - potential(prev, cfg)
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


def test_defence_potential_improves_when_a_defender_drops_back():
    """
    Docstring for test_defence_potential_improves_when_a_defender_drops_back:

    defence is now a potential: moving from nobody goal-side to someone
    goal-side raises phi, so F = gamma*phi(next) - phi(prev) is positive.
    """
    cfg = RewardConfig(ball_to_goal=0, players_to_ball=0, defence=0.02, time_penalty=0)
    gamma = 0.99
    env = make_env(cfg, gamma=gamma)
    ball = [0, 0]
    enemy = [[0, -1]] * 3
    exposed = make_obs(ball, [[0, -2], [0, -3], [0, -4]], enemy)   # nobody goal-side
    covered = make_obs(ball, [[0, 2], [0, -3], [0, -4]], enemy)    # one goal-side
    # phi(exposed) = -0.02, phi(covered) = 0.0
    r = env._shaping(exposed, covered)
    expected = gamma * potential(covered, cfg) - potential(exposed, cfg)
    assert r == pytest.approx(expected)
    assert r > 0
    # a board that stays exposed both turns barely moves (only the gamma factor on
    # the unchanged potential): gamma*(-0.02) - (-0.02) = 0.02*(1-gamma)
    flat = env._shaping(exposed, exposed)
    assert flat == pytest.approx(0.02 * (1 - gamma))


def test_phi_is_zero_at_a_goal():
    """
    Docstring for test_phi_is_zero_at_a_goal:

    on a terminal (goal) turn phi(s')=0, so F = -phi(prev).
    """
    cfg = RewardConfig(ball_to_goal=0.5, players_to_ball=0.05, defence=0.02, time_penalty=0)
    env = make_env(cfg)
    players = [[1, 1], [2, -1], [-1, 2]]
    enemy = [[0, -5]] * 3
    prev = make_obs([0, -3], players, enemy)
    nxt = make_obs([0, 0], players, enemy)   # ignored on a goal (Godot already reset)
    r = env._shaping(prev, nxt, terminated=True)
    assert r == pytest.approx(-potential(prev, cfg))


def test_shaping_skipped_after_out_of_bounds():
    """
    Docstring for test_shaping_skipped_after_out_of_bounds:

    the turn right after an OOB kickoff is skipped (positions jumped).
    """
    cfg = RewardConfig(ball_to_goal=0.5, players_to_ball=0.05, defence=0.02, time_penalty=0.01)
    env = make_env(cfg)
    prev = make_obs([0, 5], [[0, 0]] * 3, [[0, -1]] * 3)
    nxt = make_obs([0, 0], [[0, 0]] * 3, [[0, -1]] * 3)
    assert env._shaping(prev, nxt, skip=True) == 0.0


def test_shaping_telescopes_over_a_game():
    """
    Docstring for test_shaping_telescopes_over_a_game:

    the discounted sum of F over a game telescopes to gamma^T*phi(end)-phi(start).
    Phase 12, step 6 T1: this is the defining property of potential-based shaping.
    """
    cfg = RewardConfig(ball_to_goal=0.5, players_to_ball=0.05, defence=0.02, time_penalty=0)
    gamma = 0.97
    env = make_env(cfg, gamma=gamma)
    rng = np.random.default_rng(3)
    # a random walk of boards (no goal, no OOB), so F = gamma*phi(s')-phi(s) each turn
    boards = []
    for _ in range(12):
        boards.append(make_obs(rng.uniform(-8, 8, 2),
                               rng.uniform(-8, 8, (3, 2)),
                               rng.uniform(-8, 8, (3, 2))))
    discounted = 0.0
    for t in range(len(boards) - 1):
        discounted += (gamma ** t) * env._shaping(boards[t], boards[t + 1])
    T = len(boards) - 1
    expected = (gamma ** T) * potential(boards[-1], cfg) - potential(boards[0], cfg)
    assert discounted == pytest.approx(expected, rel=1e-5)


def test_time_penalty_subtracts_each_turn():
    cfg = RewardConfig(ball_to_goal=0, players_to_ball=0, defence=0, time_penalty=0.01)
    env = make_env(cfg)
    board = make_obs([0, 0], [[0, 1]] * 3, [[0, -1]] * 3)
    # phi is 0 (all shaping weights off), so F is just -time_penalty
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
