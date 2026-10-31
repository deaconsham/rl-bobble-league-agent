"""
T2: one real env talking to the mock Godot. a reset, a few steps, a goal that
ends the episode, then another reset. also the step timeout when the fake
engine hangs, and the version refusal when it speaks an old protocol.
"""
import numpy as np
import pytest

from shared_memory_link import Shared_memory_link
from bobble_league_env import Bobble_league_env
from mock_godot import Mock_godot


def make_env(path, **mock_kwargs):
    game = Mock_godot(path, **mock_kwargs)
    bridge = Shared_memory_link(path=path, create=False)
    env = Bobble_league_env(bridge=bridge, step_timeout=5.0)
    return game, env


def test_reset_then_steps_then_goal(tmp_path):
    path = str(tmp_path / "shm.dat")
    game, env = make_env(path, goal_after=3, scorer=1)
    try:
        obs, info = env.reset()
        assert obs.shape == (14,)
        assert info["play_mode"] == "AI_VS_AI"
        assert info["learner_team"] == 1

        action = np.zeros(6, dtype=np.float32)
        obs, r, terminated, truncated, step_info = env.step(action)
        assert not terminated  # turn 1 of 3
        obs, r, terminated, truncated, step_info = env.step(action)
        assert not terminated  # turn 2 of 3
        obs, r, terminated, truncated, step_info = env.step(action)
        assert terminated       # turn 3 scores
        assert step_info["goal_reward"] == pytest.approx(1.0)
        assert step_info["scorer"] == 1

        # the game resets cleanly for the next episode
        obs, info = env.reset()
        assert obs.shape == (14,)
    finally:
        env.close()
        game.stop()


def test_opponent_scores_gives_negative_reward(tmp_path):
    path = str(tmp_path / "shm.dat")
    game, env = make_env(path, goal_after=2, scorer=2)
    try:
        env.reset()
        action = np.zeros(6, dtype=np.float32)
        env.step(action)
        _, r, terminated, _, step_info = env.step(action)
        assert terminated
        assert step_info["goal_reward"] == pytest.approx(-1.0)
        assert step_info["scorer"] == 2
    finally:
        env.close()
        game.stop()


def test_hang_times_out(tmp_path):
    path = str(tmp_path / "shm.dat")
    game, env = make_env(path, goal_after=10)
    try:
        env.reset()
        game.set_hang(True)
        with pytest.raises(TimeoutError):
            env.step(np.zeros(6, dtype=np.float32))
    finally:
        game.set_hang(False)
        env.close()
        game.stop()


def test_old_protocol_mock_is_refused(tmp_path):
    path = str(tmp_path / "shm.dat")
    game = Mock_godot(path, protocol_version=1)
    try:
        with pytest.raises(RuntimeError, match="protocol v1"):
            Shared_memory_link(path=path, create=False)
    finally:
        game.stop()
