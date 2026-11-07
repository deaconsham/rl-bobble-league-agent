"""
T1: Elo maths on known results.

Checks the expected-score logistic, a single rating update against hand math, that
a batch of games equals the same sequence of single games, that anchors do not
move, and that a player that wins more ends up rated higher.
"""
import csv

import pytest

from elo import DEFAULT_K, ELO_SCALE, Elo_table, expected_score, update_pair


def test_expected_score_is_half_at_equal_ratings():
    assert expected_score(1000, 1000) == pytest.approx(0.5)


def test_expected_scores_sum_to_one():
    a, b = 1200.0, 900.0
    assert expected_score(a, b) + expected_score(b, a) == pytest.approx(1.0)


def test_expected_score_matches_the_logistic_by_hand():
    # a 400-point lead is a 10:1 favourite: 10/11 ~= 0.909
    assert expected_score(1400, 1000) == pytest.approx(10.0 / 11.0, abs=1e-9)
    # symmetric: the trailing side is 1/11
    assert expected_score(1000, 1400) == pytest.approx(1.0 / 11.0, abs=1e-9)


def test_update_pair_matches_hand_math():
    # equal ratings, A wins: expected 0.5 each, K=32 -> A +16, B -16
    new_a, new_b = update_pair(1000.0, 1000.0, score_a=1.0, k=32.0)
    assert new_a == pytest.approx(1016.0)
    assert new_b == pytest.approx(984.0)
    # rating mass is conserved
    assert (new_a + new_b) == pytest.approx(2000.0)


def test_update_pair_draw_moves_toward_expectation():
    # A is favourite (1200 vs 1000): a draw is below A's expectation, so A loses
    # points and B gains them.
    ea = expected_score(1200.0, 1000.0)
    new_a, new_b = update_pair(1200.0, 1000.0, score_a=0.5, k=32.0)
    assert new_a == pytest.approx(1200.0 + 32.0 * (0.5 - ea))
    assert new_a < 1200.0 and new_b > 1000.0


def test_record_match_equals_game_by_game():
    batch = Elo_table(k=32.0)
    batch.add_player('a')
    batch.add_player('b')
    batch.record_match('a', 'b', wins_a=3, wins_b=1, draws=1)

    one = Elo_table(k=32.0)
    one.add_player('a')
    one.add_player('b')
    for s in (1.0, 1.0, 1.0, 0.0, 0.5):
        one.record_game('a', 'b', s)

    assert batch.rating('a') == pytest.approx(one.rating('a'))
    assert batch.rating('b') == pytest.approx(one.rating('b'))


def test_anchor_does_not_move_but_still_shapes_the_other():
    table = Elo_table(k=32.0)
    table.add_player('chaser', rating=1000.0, anchor=True)
    table.add_player('ckpt', rating=1000.0)
    table.record_match('ckpt', 'chaser', wins_a=8, wins_b=2)
    # the anchor stays pinned
    assert table.rating('chaser') == pytest.approx(1000.0)
    # the checkpoint rose by beating the anchor 80%
    assert table.rating('ckpt') > 1000.0


def test_more_wins_ranks_higher():
    table = Elo_table(k=32.0)
    for name in ('strong', 'weak', 'anchor'):
        table.add_player(name, anchor=(name == 'anchor'))
    table.record_match('strong', 'anchor', 9, 1)
    table.record_match('weak', 'anchor', 2, 8)
    ranked = table.ranked()
    names = [n for n, _ in ranked]
    assert names.index('strong') < names.index('weak')
    assert table.rating('strong') > table.rating('weak')


def test_to_csv_writes_ranked_rows(tmp_path):
    table = Elo_table(k=32.0)
    table.add_player('chaser', rating=1000.0, anchor=True)
    table.add_player('ckpt', rating=1100.0)
    out = tmp_path / 'elo.csv'
    table.to_csv(str(out))

    with open(out, newline='') as f:
        rows = list(csv.reader(f))
    assert rows[0] == ['name', 'rating', 'anchor']
    # best rating first
    assert rows[1][0] == 'ckpt'
    assert rows[1][2] == '0'
    assert rows[2][0] == 'chaser'
    assert rows[2][2] == '1'


def test_constants_are_the_standard_values():
    assert ELO_SCALE == 400.0
    assert DEFAULT_K == 32.0


# ---- best-checkpoint gate ----

def test_best_gate_needs_strictly_more_than_55_percent():
    from league_eval import BEST_GATE_WIN_RATE, beats_previous_best
    assert BEST_GATE_WIN_RATE == 0.55
    # 56/44 over 100 -> promoted
    assert beats_previous_best(56, 44, 0) is True
    # exactly 55/45 -> NOT strictly above the bar
    assert beats_previous_best(55, 45, 0) is False
    # a losing record -> not best
    assert beats_previous_best(40, 60, 0) is False
    # draws count as half a win: 50 W + 12 D over 100 = 0.56 -> promoted
    assert beats_previous_best(50, 38, 12) is True
    # no games -> no promotion
    assert beats_previous_best(0, 0, 0) is False
