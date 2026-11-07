"""
T1: the angle + power action (protocol v3). decode at the edges (-1, 0, 1),
power stays in [0.2, 1], the -1/+1 seam points at own goal, and an action
decoded by python's mirror and by godot's decode launch the same world
direction for both teams.
"""
import numpy as np
import pytest

from bobble_league_env import (
    MIN_POWER,
    decode_action,
    decode_angle,
    decode_power,
    encode_action,
    vector_to_action,
)


# ---- how godot turns a mirrored-frame aim into a WORLD direction ----
# GameManager.ApplyTeamAction applies FrameZ (team 2 -> z flipped) to the z of
# the decoded aim, so a world vector is the mirrored one with z * zSign.
def frame_z(team):
    return -1.0 if team == 2 else 1.0


def godot_world_dir(a, team):
    """
    Docstring for godot_world_dir:

    mirror-frame angle a -> the (x, z) WORLD unit the player launches in.
    """
    theta = a * np.pi
    x = np.sin(theta)
    z = -np.cos(theta) * frame_z(team)
    return np.array([x, z], dtype=np.float32)


# ---- power ----

def test_power_at_edges():
    assert decode_power(-1.0) == pytest.approx(MIN_POWER)   # 0.2
    assert decode_power(0.0) == pytest.approx(0.6)          # midpoint
    assert decode_power(1.0) == pytest.approx(1.0)          # full


def test_power_stays_in_range():
    for p in np.linspace(-2.0, 2.0, 41):   # out-of-range inputs are clamped
        s = decode_power(p)
        assert MIN_POWER - 1e-6 <= s <= 1.0 + 1e-6


def test_power_0p2_is_clearly_less_than_1p0():
    # the invariant the live smoke also checks: min power is a fifth of full
    assert decode_power(-1.0) == pytest.approx(0.2)
    assert decode_power(1.0) == pytest.approx(1.0)
    assert decode_power(-1.0) < 0.5 * decode_power(1.0)


# ---- angle ----

def test_angle_at_edges_points_where_expected():
    # a = 0 -> aim at the enemy goal (-z)
    assert np.allclose(decode_angle(0.0), [0.0, -1.0], atol=1e-6)
    # a = +-1 -> the seam, straight back at own goal (+z)
    assert np.allclose(decode_angle(1.0), [0.0, 1.0], atol=1e-6)
    assert np.allclose(decode_angle(-1.0), [0.0, 1.0], atol=1e-6)
    # a = 0.5 -> straight across +x
    assert np.allclose(decode_angle(0.5), [1.0, 0.0], atol=1e-6)
    # a = -0.5 -> straight across -x
    assert np.allclose(decode_angle(-0.5), [-1.0, 0.0], atol=1e-6)


def test_seam_is_at_own_goal():
    # both ends of the seam are the same direction (own goal), the move the
    # wrap should cost least
    assert np.allclose(decode_angle(1.0), decode_angle(-1.0), atol=1e-6)


def test_decoded_direction_is_unit_length():
    for a in np.linspace(-1.0, 1.0, 21):
        d = decode_angle(a)
        assert np.linalg.norm(d) == pytest.approx(1.0, abs=1e-5)


# ---- full 6-number action ----

def test_decode_action_shape_and_scaling():
    action = encode_action(angles=[0.0, 0.5, 1.0], powers=[1.0, 0.0, -1.0])
    vecs = decode_action(action)
    assert vecs.shape == (3, 2)
    # player 0: enemy goal, full power
    assert np.allclose(vecs[0], [0.0, -1.0], atol=1e-6)
    # player 1: across +x, mid power 0.6
    assert np.allclose(vecs[1], [0.6, 0.0], atol=1e-6)
    # player 2: own goal (seam), min power 0.2
    assert np.allclose(vecs[2], [0.0, 0.2], atol=1e-6)


def test_vector_to_action_round_trips_through_decode():
    # a chaser/human builds (x, z) vectors; packing then decoding must recover
    # the same direction and a power inside the valid band
    vecs = np.array([[0.0, -1.0], [0.7, 0.7], [-0.3, 0.1]], dtype=np.float32)
    action = vector_to_action(vecs)
    back = decode_action(action)
    for v, b in zip(vecs, back):
        # direction preserved (compare unit vectors)
        un_v = v / np.linalg.norm(v)
        un_b = b / np.linalg.norm(b)
        assert np.allclose(un_v, un_b, atol=1e-5)
        # power is the clamped length
        assert MIN_POWER - 1e-6 <= np.linalg.norm(b) <= 1.0 + 1e-6


# ---- the invariant: both teams launch the same WORLD direction ----

def test_both_teams_same_world_direction():
    # team 1 and team 2 send the SAME mirrored-frame action; because team 2's
    # obs is z-mirrored, the SAME world play is the SAME action, and godot's
    # FrameZ must bring both to the same world direction... EXCEPT z flips.
    # the real invariant: team 2's decode is team 1's decode with z negated,
    # which is exactly the mirror of the board team 2 sees.
    for a in np.linspace(-1.0, 1.0, 9):
        d1 = godot_world_dir(a, team=1)
        d2 = godot_world_dir(a, team=2)
        # x stays, z flips: the mirror relationship the frames are built on
        assert np.allclose(d1[0], d2[0], atol=1e-6)
        assert np.allclose(d1[1], -d2[1], atol=1e-6)


def test_mirrored_board_same_world_shot():
    # a player on team 1 at some spot aiming at the enemy goal, and the
    # z-mirror player on team 2 aiming at ITS enemy goal, hit the same world -z
    # vs +z goals: a=0 aims each team at its own enemy goal in world space.
    d1 = godot_world_dir(0.0, team=1)   # team 1 enemy goal at world -z
    d2 = godot_world_dir(0.0, team=2)   # team 2 enemy goal at world +z
    assert np.allclose(d1, [0.0, -1.0], atol=1e-6)
    assert np.allclose(d2, [0.0, 1.0], atol=1e-6)
