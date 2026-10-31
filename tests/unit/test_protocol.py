"""
T1: the 120-byte protocol v2 layout. checks every field's offset and size,
the version byte on create, and that an older buffer is refused.
"""
import struct

import numpy as np
import pytest

from shared_memory_link import Shared_memory_link


def test_total_size_and_offsets_do_not_overlap():
    L = Shared_memory_link
    assert L.TOTAL_SIZE == 120
    # obs 14 f32 at 0, action 6 f32 at 56
    assert L.OBS_OFFSET == 0 and L.OBS_SIZE == 14 * 4
    assert L.ACT_OFFSET == 56 and L.ACT_SIZE == 6 * 4
    assert L.OBS_OFFSET + L.OBS_SIZE <= L.ACT_OFFSET
    # scalars
    assert L.REWARD_OFFSET == 80 and L.REWARD_SIZE == 4
    assert L.DONE_OFFSET == 84
    assert L.STEP_REQ_OFFSET == 85
    assert L.RESET_REQ_OFFSET == 86
    assert L.LEARNER_TEAM_OFFSET == 87
    assert L.SCORER_OFFSET == 88
    assert L.PLAY_MODE_OFFSET == 89
    assert L.OPP_VALID_OFFSET == 90
    assert L.OPP_ACT_OFFSET == 92
    assert L.VERSION_OFFSET == 116
    # opponent action (6 f32) fits before the version byte
    assert L.OPP_ACT_OFFSET + L.ACT_SIZE <= L.VERSION_OFFSET
    # everything fits inside the buffer
    assert L.VERSION_OFFSET < L.TOTAL_SIZE


def test_create_writes_version(tmp_path):
    path = str(tmp_path / "shm.dat")
    link = Shared_memory_link(path=path, create=True)
    try:
        assert link.buf[link.VERSION_OFFSET] == Shared_memory_link.PROTOCOL_VERSION
    finally:
        link.close()


def test_round_trip_action_and_obs(tmp_path):
    path = str(tmp_path / "shm.dat")
    link = Shared_memory_link(path=path, create=True)
    try:
        action = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6], dtype=np.float32)
        link.write_action(action)
        raw = link.buf[link.ACT_OFFSET:link.ACT_OFFSET + link.ACT_SIZE]
        back = np.frombuffer(raw, dtype=np.float32)
        assert np.allclose(back, action)

        # reward/done are read back from raw bytes the way godot would write them
        link.buf[link.REWARD_OFFSET:link.REWARD_OFFSET + 4] = struct.pack("<f", -1.0)
        link.buf[link.DONE_OFFSET] = 1
        assert link.read_reward() == pytest.approx(-1.0)
        assert link.read_done() is True
    finally:
        link.close()


def test_opponent_action_valid_flag(tmp_path):
    path = str(tmp_path / "shm.dat")
    link = Shared_memory_link(path=path, create=True)
    try:
        link.write_opponent_action(None)
        assert link.buf[link.OPP_VALID_OFFSET] == 0
        opp = np.arange(6, dtype=np.float32)
        link.write_opponent_action(opp)
        assert link.buf[link.OPP_VALID_OFFSET] == 1
        raw = link.buf[link.OPP_ACT_OFFSET:link.OPP_ACT_OFFSET + link.ACT_SIZE]
        assert np.allclose(np.frombuffer(raw, dtype=np.float32), opp)
    finally:
        link.close()


def test_play_mode_decode(tmp_path):
    path = str(tmp_path / "shm.dat")
    link = Shared_memory_link(path=path, create=True)
    try:
        for code, name in {0: "AI_VS_AI", 1: "HUMAN_VS_AI", 2: "HUMAN_VS_HUMAN"}.items():
            link.buf[link.PLAY_MODE_OFFSET] = code
            assert link.read_play_mode() == name
        link.buf[link.PLAY_MODE_OFFSET] = 99
        assert link.read_play_mode() == "UNKNOWN"
    finally:
        link.close()


def test_old_version_is_refused(tmp_path):
    path = str(tmp_path / "shm.dat")
    # a build that wrote protocol v1
    maker = Shared_memory_link(path=path, create=True)
    maker.buf[maker.VERSION_OFFSET] = 1
    maker.close()

    with pytest.raises(RuntimeError, match="protocol v1"):
        Shared_memory_link(path=path, create=False)


def test_missing_file_is_refused(tmp_path):
    path = str(tmp_path / "does_not_exist.dat")
    with pytest.raises(FileNotFoundError):
        Shared_memory_link(path=path, create=False)
