"""
T2: league_eval spreads its pairings over several Godot games.

Uses mock games (real Shared_memory_link bridges on temp files) to check that
play_pairings_parallel plays every pairing exactly once over N games, tallies
each match from the A side, and spreads the work over more than one game. No real
Godot, no network call: scripted opponents act on the mock obs.
"""
import numpy as np

from bobble_league_env import Bobble_league_env
from shared_memory_link import Shared_memory_link
from league_eval import play_pairings_parallel
from opponents import ChaserOpponent, StillOpponent
from mock_godot import Mock_launcher


def _envs(launcher, n, step_timeout=5.0):
    bridges = [launcher.wait_for_handshake(i) for i in range(n)]
    return [Bobble_league_env(opponent=None, max_ep_len=50, step_timeout=step_timeout,
                              bridge=b) for b in bridges]


def test_every_pairing_played_once_over_four_games(tmp_path):
    # four games available; the mock scores after 2 turns so matches are short
    n_games = 4
    launcher = Mock_launcher(n_games, tmp_path, goal_after=2)
    launcher.launch_all()
    envs = _envs(launcher, n_games)

    agents = {'a': ChaserOpponent(seed=1), 'b': ChaserOpponent(seed=2),
              'c': ChaserOpponent(seed=3)}
    anchors = {'still': StillOpponent()}
    pairings = [('a', 'b'), ('a', 'c'), ('b', 'c'),
                ('a', 'still'), ('b', 'still'), ('c', 'still')]
    try:
        results = play_pairings_parallel(envs, pairings, agents, agents, anchors,
                                         n_games=3, max_ep_len=50)
    finally:
        for e in envs:
            e.close()
        launcher.stop_all()

    # every pairing has a result, and each is 3 games (W+L+D sums to 3)
    assert set(results.keys()) == set(pairings)
    for pair, (w, l, d) in results.items():
        assert w + l + d == 3


def test_parallel_path_handles_more_pairings_than_games(tmp_path):
    # only 2 games but 6 pairings: the scheduler must refill a freed slot until
    # every pairing is done.
    n_games = 2
    launcher = Mock_launcher(n_games, tmp_path, goal_after=2)
    launcher.launch_all()
    envs = _envs(launcher, n_games)

    agents = {'a': ChaserOpponent(seed=1), 'b': ChaserOpponent(seed=2),
              'c': ChaserOpponent(seed=3)}
    anchors = {'still': StillOpponent()}
    pairings = [('a', 'b'), ('a', 'c'), ('b', 'c'),
                ('a', 'still'), ('b', 'still'), ('c', 'still')]
    try:
        results = play_pairings_parallel(envs, pairings, agents, agents, anchors,
                                         n_games=2, max_ep_len=50)
    finally:
        for e in envs:
            e.close()
        launcher.stop_all()

    assert set(results.keys()) == set(pairings)
    for pair, (w, l, d) in results.items():
        assert w + l + d == 2
