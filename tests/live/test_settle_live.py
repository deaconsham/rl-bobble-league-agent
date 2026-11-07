"""
T4-level live check that a turn only ends once the ball and players have really
stopped. Needs the real Godot build, so it is marked `live` and skipped by
`pytest -m "not live"`.

Godot runs with --settle-check, which prints one SETTLE line per turn (see
bobble_league/Scripts/SettleCheck.cs). Before the fix, bodies were snapped to 0
the moment they dropped under a "stop speed" (1.0 in training), so a ball still
rolling at ~1 m/s froze ~3 units short and stop_* read ~1.0 on most turns.
"""
import os
import re
import time

import numpy as np
import pytest

from launch_games import Game_launcher, default_godot_exe

pytestmark = pytest.mark.live

TURNS = 150


class Settle_check_launcher(Game_launcher):
    """Docstring for Settle_check_launcher: a Game_launcher that adds --settle-check."""

    def _args(self, index):
        return super()._args(index) + ['--settle-check']


def parse_settle_lines(text):
    """
    Docstring for parse_settle_lines:

    Turns each "SETTLE key=value ..." line of a Godot log into a dict of floats.
    """
    rows = []
    for line in text.splitlines():
        if line.startswith('SETTLE '):
            rows.append({k: float(v) for k, v in re.findall(r'(\w+)=([-\d.]+)', line)})
    return rows


def play_random_turns(link, rng, turns):
    """
    Docstring for play_random_turns:

    Random actions for both teams, a random-layout reset after a goal or every
    25 turns, and a short pause after each turn so the game sits idle long
    enough for any after-the-fact movement to show up as drift.
    """
    link.request_reset()
    link.wait_for_reset()
    for t in range(turns):
        link.write_action(rng.uniform(-1, 1, 6).astype(np.float32))
        link.write_opponent_action(rng.uniform(-1, 1, 6).astype(np.float32))
        link.request_step()
        link.wait_for_step(timeout=60)
        time.sleep(0.02)
        if link.read_done() or t % 25 == 24:
            link.request_reset(random_start=True)
            link.wait_for_reset()


def test_turns_end_only_once_bodies_have_stopped(tmp_path):
    """
    Docstring for test_turns_end_only_once_bodies_have_stopped:

    Nothing moves after the observation is sent, bodies come to rest from a low
    speed instead of being frozen mid-roll, almost no turn hits the time limit,
    and Godot logs no errors (the old goal reset inside the goal area's signal
    gave Jolt "ref_count <= 0" errors).
    """
    if not os.path.exists(default_godot_exe()):
        pytest.skip('Godot console exe not found (set BOBBLE_GODOT_EXE)')

    launcher = Settle_check_launcher(n_games=1, shm_prefix=str(tmp_path), log_dir=str(tmp_path))
    launcher.launch_all()
    try:
        link = launcher.wait_for_handshake(0, timeout=120)
        try:
            play_random_turns(link, np.random.default_rng(0), TURNS)
        finally:
            link.close()
    finally:
        launcher.stop_all()

    with open(launcher.log_paths[0], errors='ignore') as f:
        log = f.read()
    rows = parse_settle_lines(log)
    assert len(rows) >= TURNS - 10, 'too few SETTLE lines; was the game built with SettleCheck?'

    drift = max(max(r['drift_ball'], r['drift_player']) for r in rows)
    assert drift <= 0.01, f'a body moved {drift:.4f} units after its position was sent'

    # the brake leaves ~0.1 m/s on the last tick; the old hard stop left ~1.0
    stop_ball = np.percentile([r['stop_ball'] for r in rows], 95)
    stop_player = np.percentile([r['stop_player'] for r in rows], 95)
    assert stop_ball <= 0.2, f'ball frozen while still rolling (p95 {stop_ball:.2f} m/s)'
    assert stop_player <= 0.2, f'players frozen while still moving (p95 {stop_player:.2f} m/s)'

    timed_out = sum(r['timed_out'] for r in rows) / len(rows)
    assert timed_out <= 0.02, f'{timed_out:.1%} of turns hit MaxTurnSeconds'

    assert 'ERROR' not in log, 'Godot logged an error:\n' + '\n'.join(
        line for line in log.splitlines() if 'ERROR' in line)
