"""
T1: the learner-frame maths. opponent_view mirrors on z and swaps
friendly/enemy; doing it twice gives the same board; a symmetric kickoff
looks identical from both sides.
"""
import numpy as np

from bobble_league_env import (
    opponent_view,
    split_obs,
    normalize_obs,
    decode_action,
    vector_to_action,
)


def make_obs(ball, friendly, enemy):
    return np.concatenate([
        np.asarray(ball, dtype=np.float32),
        np.asarray(friendly, dtype=np.float32).ravel(),
        np.asarray(enemy, dtype=np.float32).ravel(),
    ])


def test_split_obs_shapes():
    obs = np.arange(14, dtype=np.float32)
    ball, friendly, enemy = split_obs(obs)
    assert ball.shape == (2,)
    assert friendly.shape == (3, 2)
    assert enemy.shape == (3, 2)
    assert np.allclose(ball, [0, 1])
    assert np.allclose(friendly[0], [2, 3])
    assert np.allclose(enemy[0], [8, 9])


def test_opponent_view_is_an_involution():
    obs = make_obs([1.0, 2.0],
                   [[3, 4], [5, 6], [7, 8]],
                   [[-3, -4], [-5, -6], [-7, -8]])
    once = opponent_view(obs)
    twice = opponent_view(once)
    assert np.allclose(twice, obs)


def test_opponent_view_flips_z_and_swaps_teams():
    obs = make_obs([1.0, 2.0],
                   [[3, 4], [5, 6], [7, 8]],
                   [[9, 10], [11, 12], [13, 14]])
    view = opponent_view(obs)
    ball, friendly, enemy = split_obs(view)
    # ball z flips
    assert np.allclose(ball, [1.0, -2.0])
    # old enemy becomes new friendly, with z flipped
    assert np.allclose(friendly, [[9, -10], [11, -12], [13, -14]])
    # old friendly becomes new enemy, with z flipped
    assert np.allclose(enemy, [[3, -4], [5, -6], [7, -8]])


def test_symmetric_kickoff_looks_the_same_from_both_sides():
    # a board that is its own mirror (x kept, z flipped and teams swapped):
    # friendly and enemy are z-mirror images, ball on the centre line.
    friendly = [[2, 5], [-2, 5], [0, 7]]
    enemy = [[2, -5], [-2, -5], [0, -7]]
    obs = make_obs([0.0, 0.0], friendly, enemy)
    view = opponent_view(obs)
    assert np.allclose(view, obs)


def test_normalize_obs_clips_to_unit_range():
    obs = make_obs([100.0, 100.0],
                   [[100, 100]] * 3,
                   [[-100, -100]] * 3)
    norm = normalize_obs(obs)
    assert norm.max() <= 1.0 and norm.min() >= -1.0


# ---- the new angle+power action still round-trips through the mirror ----

def test_action_vector_round_trip():
    # a (x, z) aim packed to angle+power then decoded recovers the direction
    # (and a power inside the band). this is the chaser/human path.
    vecs = np.array([[0.2, -0.9], [-0.6, 0.4], [0.0, -1.0]], dtype=np.float32)
    back = decode_action(vector_to_action(vecs))
    for v, b in zip(vecs, back):
        uv = v / np.linalg.norm(v)
        ub = b / np.linalg.norm(b)
        assert np.allclose(uv, ub, atol=1e-5)


def test_action_mirror_flips_z_like_the_board():
    # decode in one frame, flip z (the mirror the two teams differ by), and it
    # equals decoding the action you'd pack from the z-flipped aim. this is the
    # action-side of opponent_view's z flip.
    vecs = np.array([[0.3, -0.8], [-0.5, 0.5], [0.1, -0.2]], dtype=np.float32)
    decoded = decode_action(vector_to_action(vecs))
    flipped_in = vecs * np.array([1.0, -1.0], dtype=np.float32)
    decoded_flipped = decode_action(vector_to_action(flipped_in))
    assert np.allclose(decoded * np.array([1.0, -1.0]), decoded_flipped, atol=1e-5)

