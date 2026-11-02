"""
T2 regression for the Phase 12 live-Elo decade-boundary deadlock (review item).

The in-run live-Elo eval drove a LONG-LIVED spare Godot game with the TRAINING
step_timeout (120 s) and, unlike Vector_env, had NO hang-restart: when the spare
game wedged at a decade boundary the eval kept hammering the same dead game, so
one check cost n_games * anchors * step_timeout of idle polling (the "deadlock",
seen at epochs 60 and 70 in Phase 12). The fix makes the eval resilient like
Vector_env:

  - a wedged game is RELAUNCHED once (a fresh env), then the eval carries on;
  - after `max_hangs` hangs the remaining games are scored draws AT ONCE instead
    of waiting a full timeout per game;
  - an overall `deadline` caps the whole check, so a boundary wedge can never idle
    the training process for minutes.

These tests use the fake-Godot harness (Mock_godot / Mock_launcher) so they run
without a real engine, and keep step_timeout tiny so a "timeout" is milliseconds.
"""
import time

import numpy as np
import torch

from bobble_league_env import Bobble_league_env
from neural_network import Actor_critic_neural_network
from opponents import make_test_opponent
from shared_memory_link import Shared_memory_link
from mock_godot import Mock_godot, Mock_launcher
import live_elo


def _state():
    ac = Actor_critic_neural_network(obs_dim=14, act_dim=6)
    return {k: v.detach().cpu().clone() for k, v in ac.state_dict().items()}


def _env_on(path, step_timeout):
    return Bobble_league_env(opponent=None, max_ep_len=1000, step_timeout=step_timeout,
                             bridge=Shared_memory_link(path=path, create=False))


def test_wedged_spare_game_bails_out_fast_without_relaunch(tmp_path):
    """A spare game that wedges mid-check, with NO relaunch available, must return
    in BOUNDED time (max_hangs), not wait a full timeout on every one of n_games.
    Before the fix this ground through n_games * step_timeout."""
    shm = str(tmp_path / "live_elo.dat")
    mock = Mock_godot(shm, goal_after=3, step_delay=0.02)
    env = _env_on(shm, step_timeout=0.3)
    anchors = {"chaser": make_test_opponent("chaser")}

    # wedge the game almost immediately so the first real turn times out
    mock.set_hang(True)

    t0 = time.time()
    env2, rec = live_elo.score_vs_anchors(env, _state(), anchors, n_games=20,
                                          abs_std=None, relaunch=None,
                                          deadline=time.monotonic() + 30)
    dt = time.time() - t0

    # 20 games at a 0.3 s timeout would be ~6 s if it hammered each one; the fix
    # bails after max_hangs (2) timeouts, so well under 2 s, and every game is a
    # draw (the game never answered).
    assert dt < 2.0, f"did not bail out fast: {dt:.2f}s"
    assert rec["chaser"] == (0, 0, 20)
    mock.stop()
    for e in (env, env2):
        try:
            e.close()
        except Exception:
            pass


def test_check_budget_caps_the_whole_check(tmp_path):
    """An already-spent deadline makes the check return at once, scoring the games
    draws, instead of playing any turn. This is the backstop that bounds one
    live-Elo check even with no launcher to relaunch a wedged game."""
    shm = str(tmp_path / "live_elo.dat")
    mock = Mock_godot(shm, goal_after=3, step_delay=0.02)
    env = _env_on(shm, step_timeout=0.3)
    anchors = {"chaser": make_test_opponent("chaser"),
               "defender": make_test_opponent("defender")}

    t0 = time.time()
    env2, rec = live_elo.score_vs_anchors(env, _state(), anchors, n_games=20,
                                          abs_std=None, relaunch=None,
                                          deadline=time.monotonic() - 1.0)  # already past
    dt = time.time() - t0

    assert dt < 0.5, f"deadline did not short-circuit: {dt:.2f}s"
    # every game of both anchors scored a draw with no turns played
    assert rec["chaser"] == (0, 0, 20)
    assert rec["defender"] == (0, 0, 20)
    mock.stop()
    for e in (env, env2):
        try:
            e.close()
        except Exception:
            pass


def test_relaunch_recovers_and_keeps_scoring(tmp_path):
    """The real-run path: the spare game wedges, the relaunch callback brings up a
    FRESH healthy game (as Vector_env relaunches a training game), and the eval
    recovers and scores real games afterward rather than all draws.

    Uses Mock_launcher so the relaunch follows the same stop_one/launch_one/
    wait_for_handshake path the training loop uses for the real spare game."""
    launcher = Mock_launcher(1, tmp_path, goal_after=3)
    launcher.launch_all()
    shm = launcher.shm_paths[0]
    env = _env_on(shm, step_timeout=0.3)
    anchors = {"chaser": make_test_opponent("chaser")}

    # wedge the only game so the first real turn hangs
    launcher.games[0].set_hang(True)

    def relaunch():
        # mirror training_loop._relaunch_live_elo: stop, relaunch, re-attach a fresh
        # env on the same shm. the relaunched Mock_godot is healthy (not hung).
        try:
            env.close()
        except Exception:
            pass
        launcher.stop_one(0)
        launcher.launch_one(0)
        bridge = launcher.wait_for_handshake(0, timeout=2.0)
        return Bobble_league_env(opponent=None, max_ep_len=1000, step_timeout=0.3,
                                 bridge=bridge)

    env2, rec = live_elo.score_vs_anchors(env, _state(), anchors, n_games=10,
                                          abs_std=None, relaunch=relaunch,
                                          deadline=time.monotonic() + 30)

    w, l, d = rec["chaser"]
    # the relaunched game is healthy, so most of the 10 games actually PLAY to a
    # result (win or loss), proving the eval recovered rather than drawing out.
    assert (w + l) >= 5, f"relaunch did not recover enough games: {rec}"
    assert launcher.launch_counts[0] == 2, "spare game was not relaunched exactly once"
    try:
        env2.close()
    except Exception:
        pass
    launcher.stop_all()
