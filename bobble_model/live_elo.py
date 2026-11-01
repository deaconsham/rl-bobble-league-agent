"""
Live Elo during training (Phase 10, steps 5-7).

Phase 7 got worse for hours with nobody watching, because Elo only came from
separate league_eval.py runs after the fact. This scores the newest save WHILE
training runs: every few epochs it plays the current net against the fixed chaser
and defender anchors on one spare Godot game, turns the win rates into an Elo
number on the SAME anchored scale league_eval uses, and logs it to TensorBoard.

It also keeps best.pt (a save becomes best only if its live Elo is the highest so
far AND it beats the old best head-to-head), and rolls back to best.pt after a
drop of more than ~100 Elo that holds for two checks.

Two pieces, kept apart so the maths is unit-testable without Godot:
  - live_elo_rating: pure maths. Win rates vs the anchors -> one Elo number.
  - score_vs_anchors / Best_tracker: drive a real spare game and hold the best /
    rollback state. The training loop calls these.
"""
import numpy as np

from elo import Elo_table
from opponents import PolicyOpponent

# the anchors the live score is measured against, fixed at these ratings so the
# number means the same thing every check and matches league_eval's scale.
LIVE_ANCHORS = ("chaser", "defender")
LIVE_ANCHOR_RATINGS = {"chaser": 1000.0, "defender": 950.0}

# a save becomes best only if it beats the old best in strictly more than this
# fraction of their head-to-head games (same 0.55 gate league_eval uses, R37).
BEST_GATE_WIN_RATE = 0.55

# a drop of more than this many Elo below the best, sustained for ROLLBACK_CHECKS
# consecutive checks, triggers a roll-back to best.pt.
ROLLBACK_ELO_DROP = 100.0
ROLLBACK_CHECKS = 2


def live_elo_rating(records, k=32.0, anchor_ratings=None, label="live"):
    """
    Docstring for live_elo_rating:

    Turn the newest net's match records against the fixed anchors into one Elo
    number. `records` is {anchor_name: (wins, losses, draws)} from the net's point
    of view. The anchors are pinned at their fixed ratings (so the scale never
    drifts) and only the net's own rating moves, exactly like league_eval.

    Pure maths, no Godot and no torch: feed it hand-worked win/loss/draw counts in
    a test and check the Elo it returns. Returns the net's final rating (float).
    """
    anchor_ratings = anchor_ratings or LIVE_ANCHOR_RATINGS
    table = Elo_table(k=k)
    for name, rating in anchor_ratings.items():
        table.add_player(name, rating=rating, anchor=True)
    table.add_player(label)
    for name in anchor_ratings:
        w, l, d = records.get(name, (0, 0, 0))
        table.record_match(label, name, w, l, d)
    return table.rating(label)


def _play_match(env, agent, n_games):
    """Play n_games with `agent` on the learner side against env's current
    opponent. Returns (wins, losses, draws) from the learner's point of view. A
    turn that never settles counts the game a draw and resets, so one wedged turn
    cannot block the check (mirrors league_eval.play_match)."""
    wins = losses = draws = 0
    env.reset()
    for _ in range(n_games):
        done = False
        while not done:
            action = agent.act(env._raw_obs)
            try:
                _, _, terminated, truncated, info = env.step(action)
            except TimeoutError:
                draws += 1
                done = True
                try:
                    env.reset()
                except TimeoutError:
                    pass
                break
            if terminated or truncated:
                scorer = info.get("scorer", 0) if terminated else 0
                if scorer == 1:
                    wins += 1
                elif scorer == 2:
                    losses += 1
                else:
                    draws += 1
                done = True
                env.reset()
    return wins, losses, draws


def score_vs_anchors(env, state_dict, anchors, n_games=20, abs_std=None):
    """
    Docstring for score_vs_anchors:

    Play a snapshot of the net against each fixed anchor on the given spare game
    (env), n_games each. `state_dict` is the net weights to score; `anchors` is
    {name: opponent}. With abs_std None the net plays its mean move
    (deterministic); with abs_std set it samples at that absolute spread (the
    small-sigma scoring, step 7). Returns {name: (wins, losses, draws)}.
    """
    deterministic = abs_std is None
    agent = PolicyOpponent(state_dict, deterministic=deterministic,
                           abs_std=abs_std, name="live")
    records = {}
    for name, opp in anchors.items():
        env.opponent = opp
        records[name] = _play_match(env, agent, n_games)
    return records


def head_to_head(env, state_dict_a, state_dict_b, n_games=20):
    """Play net A (learner, mean move) vs net B (opponent, mean move) on the spare
    game, n_games. Returns A's (wins, losses, draws). Used for the best.pt gate:
    a challenger must beat the current best here, not just post a higher Elo."""
    agent = PolicyOpponent(state_dict_a, deterministic=True, name="challenger")
    opp = PolicyOpponent(state_dict_b, deterministic=True, name="best")
    env.opponent = opp
    return _play_match(env, agent, n_games)


def beats_best(wins, losses, draws, threshold=BEST_GATE_WIN_RATE):
    """True when the challenger's score rate (a draw is half a win) over the
    head-to-head games is strictly above `threshold`. No games -> False."""
    total = int(wins) + int(losses) + int(draws)
    if total <= 0:
        return False
    score = (int(wins) + 0.5 * int(draws)) / total
    return score > threshold


class Best_tracker:
    """
    Docstring for Best_tracker class:

    Holds the best-so-far save and decides, from each live check, whether the new
    save becomes best and whether to roll back.

    best rule (step 6): a save becomes best only if its live Elo is the highest
    seen so far AND it beats the current best in a head-to-head (>55%). The first
    save checked becomes best with no head-to-head (nothing to beat yet).

    rollback rule (step 6): once there is a best, a check whose live Elo is more
    than ROLLBACK_ELO_DROP below the best's Elo is a "down" check. After
    ROLLBACK_CHECKS consecutive down checks, roll back to best.pt. One check at or
    above the threshold clears the streak.

    The tracker never touches Godot or files itself; it returns decisions the
    training loop acts on (saving best.pt, loading it back). Keeping the policy
    here makes it unit-testable against hand-chosen Elo sequences.
    """

    def __init__(self, elo_drop=ROLLBACK_ELO_DROP, rollback_checks=ROLLBACK_CHECKS,
                 gate=BEST_GATE_WIN_RATE):
        self.elo_drop = float(elo_drop)
        self.rollback_checks = int(rollback_checks)
        self.gate = float(gate)
        self.best_elo = None     # Elo of the current best
        self.best_epoch = None   # epoch the current best was saved at
        self.down_streak = 0     # consecutive checks far below the best

    def consider(self, epoch, elo, head_to_head_record=None):
        """
        Fold one live check in. `elo` is this save's live Elo; when there is
        already a best, `head_to_head_record` is this save's (w, l, d) vs the best
        (play it before calling, or pass None to decline promotion this check).

        Returns a dict:
          became_best: True if this save is the new best (loop should save best.pt),
          rollback:    True if the loop should load best.pt back,
          reason:      short text for the log.
        """
        # first ever check: this save is best outright.
        if self.best_elo is None:
            self.best_elo = float(elo)
            self.best_epoch = int(epoch)
            self.down_streak = 0
            return {"became_best": True, "rollback": False,
                    "reason": f"first best at epoch {epoch}, elo {elo:.1f}"}

        became_best = False
        reason = ""
        if elo > self.best_elo and head_to_head_record is not None:
            w, l, d = head_to_head_record
            if beats_best(w, l, d, self.gate):
                old = self.best_elo
                self.best_elo = float(elo)
                self.best_epoch = int(epoch)
                became_best = True
                reason = (f"new best at epoch {epoch}: elo {elo:.1f} > {old:.1f} "
                          f"and h2h W{w} L{l} D{d} beats best")

        # rollback accounting: a check far below the best is a "down" check.
        if elo < self.best_elo - self.elo_drop:
            self.down_streak += 1
        else:
            self.down_streak = 0

        rollback = False
        if not became_best and self.down_streak >= self.rollback_checks:
            rollback = True
            reason = (f"elo {elo:.1f} is >{self.elo_drop:.0f} below best "
                      f"{self.best_elo:.1f} for {self.down_streak} checks; roll back to best.pt")
            self.down_streak = 0  # reset after acting so it does not fire every check

        if not reason:
            reason = (f"epoch {epoch}: elo {elo:.1f}, best {self.best_elo:.1f}, "
                      f"down streak {self.down_streak}")
        return {"became_best": became_best, "rollback": rollback, "reason": reason}
