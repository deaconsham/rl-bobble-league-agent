"""
Live Elo during training (Phase 10, steps 5-7).

Phase 7 got worse for hours with nobody watching, because Elo only came from
separate league_eval.py runs after the fact. This scores the newest save WHILE
training runs: every few epochs it plays the current net against the fixed chaser
and defender anchors on one spare Godot game, turns the win rates into an Elo
number on the SAME anchored scale league_eval uses, and logs it to TensorBoard.

It also keeps best.pt (a save becomes best when its live Elo clears the current
best by more than the noise margin, unless a head-to-head shows it is clearly the
weaker net one-on-one), and rolls back to best.pt after a drop of more than ~100
Elo that holds for two checks.

Two pieces, kept apart so the maths is unit-testable without Godot:
  - live_elo_rating: pure maths. Win rates vs the anchors -> one Elo number.
  - score_vs_anchors / Best_tracker: drive a real spare game and hold the best /
    rollback state. The training loop calls these.
"""
import time

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

# a challenger must beat the current best's live Elo by MORE than this margin (on
# top of winning the head-to-head) before it replaces best.pt. Phase 11 found the
# 2-anchor / 20-game live gate carries ~20 Elo of run-to-run noise and once handed
# a weaker save to the next phase (it ranked epoch 99 / T5 921.5 over epoch 80 /
# T5 1039.1). Requiring a margin above that noise band stops a save that is only
# noise-ahead from overwriting a genuinely stronger best. The head-to-head gate
# still applies as well, so both must pass.
BEST_MARGIN_ELO = 25.0


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


def _reset_quietly(env):
    """
    Docstring for _reset_quietly:

    Reset the spare game, swallowing a reset timeout so a wedged game can never
    propagate out of the eval. Returns True on a clean reset, False on a timeout.
    """
    try:
        env.reset()
        return True
    except TimeoutError:
        return False


def _play_match(env, agent, n_games, relaunch=None, deadline=None, max_hangs=2):
    """
    Docstring for _play_match:

    Play n_games with `agent` on the learner side against env's current
    opponent. Returns (env, wins, losses, draws) from the learner's point of view.

    Resilient to a wedged spare game (Phase 12 review fix). A turn that never
    settles raises TimeoutError from the bridge; the OLD code then scored a draw
    and moved on, but kept hammering the SAME dead game, so a boundary wedge cost
    n_games * step_timeout of idle polling (the decade-boundary "deadlock"). Now:

      - `relaunch`: optional callable that restarts the spare Godot and returns a
        fresh env (like Vector_env's per-game restart). Called once on the first
        hang; the match then carries on against the fresh game.
      - `max_hangs`: after this many hangs (default 2) the game is treated as
        genuinely dead: the remaining games are scored draws AT ONCE and the match
        returns, instead of waiting out another timeout per game.
      - `deadline`: optional absolute time.monotonic() cap for the WHOLE match;
        once passed, the rest of the games are scored draws immediately. The caller
        sets one per check so a single boundary can never eat the run.

    A turn or reset timeout always counts the game a draw, mirroring
    league_eval.play_match. The returned env may be a FRESH one after a relaunch,
    so the caller must use the returned env from here on.
    """
    wins = losses = draws = 0
    hangs = 0
    _reset_quietly(env)
    games_done = 0
    while games_done < n_games:
        # out of time: score every remaining game a draw and stop.
        if deadline is not None and time.monotonic() > deadline:
            draws += n_games - games_done
            break
        done = False
        while not done:
            if deadline is not None and time.monotonic() > deadline:
                draws += n_games - games_done
                return env, wins, losses, draws
            action = agent.act(env._raw_obs)
            try:
                _, _, terminated, truncated, info = env.step(action)
            except TimeoutError:
                hangs += 1
                draws += 1
                games_done += 1
                done = True
                if relaunch is not None and hangs == 1:
                    # first hang: try to bring the spare game back (Vector_env does
                    # the same per game). a fresh env replaces the wedged one.
                    fresh = relaunch()
                    if fresh is not None:
                        env = fresh
                        _reset_quietly(env)
                        break
                if hangs >= max_hangs:
                    # genuinely dead: score the rest draws at once, do not keep
                    # waiting a full timeout per remaining game.
                    draws += n_games - games_done
                    return env, wins, losses, draws
                _reset_quietly(env)
                break
            if terminated or truncated:
                scorer = info.get("scorer", 0) if terminated else 0
                if scorer == 1:
                    wins += 1
                elif scorer == 2:
                    losses += 1
                else:
                    draws += 1
                games_done += 1
                done = True
                _reset_quietly(env)
    return env, wins, losses, draws


def score_vs_anchors(env, state_dict, anchors, n_games=20, abs_std=None,
                     relaunch=None, deadline=None):
    """
    Docstring for score_vs_anchors:

    Play a snapshot of the net against each fixed anchor on the given spare game
    (env), n_games each. `state_dict` is the net weights to score; `anchors` is
    {name: opponent}. With abs_std None the net plays its mean move
    (deterministic); with abs_std set it samples at that absolute spread (the
    small-sigma scoring, step 7).

    `relaunch` / `deadline` are passed through to `_play_match` so a wedged spare
    game is restarted (or the check bailed out) instead of hanging (Phase 12 review
    fix). Returns (env, {name: (wins, losses, draws)}): env may be a fresh one after
    a relaunch, so the caller keeps the returned env.
    """
    deterministic = abs_std is None
    agent = PolicyOpponent(state_dict, deterministic=deterministic,
                           abs_std=abs_std, name="live")
    records = {}
    for name, opp in anchors.items():
        env.opponent = opp
        env, w, l, d = _play_match(env, agent, n_games,
                                   relaunch=relaunch, deadline=deadline)
        records[name] = (w, l, d)
    return env, records


def head_to_head(env, state_dict_a, state_dict_b, n_games=20, relaunch=None,
                 deadline=None):
    """
    Docstring for head_to_head:

    Play net A (learner, mean move) vs net B (opponent, mean move) on the spare
    game, n_games. Returns (env, A's (wins, losses, draws)). Used for the best.pt
    gate: a challenger must beat the current best here, not just post a higher Elo.

    `relaunch` / `deadline` are passed through to `_play_match` (Phase 12 review
    fix). env may be a fresh one after a relaunch, so the caller keeps it.
    """
    agent = PolicyOpponent(state_dict_a, deterministic=True, name="challenger")
    opp = PolicyOpponent(state_dict_b, deterministic=True, name="best")
    env.opponent = opp
    env, w, l, d = _play_match(env, agent, n_games,
                               relaunch=relaunch, deadline=deadline)
    return env, (w, l, d)


def beats_best(wins, losses, draws, threshold=BEST_GATE_WIN_RATE):
    """
    Docstring for beats_best:

    True when the challenger's score rate (a draw is half a win) over the
    head-to-head games is strictly above `threshold`. No games -> False.
    """
    total = int(wins) + int(losses) + int(draws)
    if total <= 0:
        return False
    score = (int(wins) + 0.5 * int(draws)) / total
    return score > threshold


def loses_head_to_head(wins, losses, draws, threshold=BEST_GATE_WIN_RATE):
    """
    Docstring for loses_head_to_head:

    True when the challenger CLEARLY lost the head-to-head: its score rate (a
    draw is half a win) is strictly below 1 - threshold. No games -> False (an
    empty / wedged h2h is not a clear loss, it is no evidence).

    This is deliberately NOT the negation of beats_best. The best-gate used to
    require beats_best (> 0.55) on a noisy 20-game h2h, which vetoed a challenger
    that was far ahead on anchor Elo but only level in the head-to-head (long run
    2: best.pt froze at epoch 0 for all 200 epochs while epoch 100 posted +240
    Elo). The 25-Elo margin is already the noise guard, so the h2h's only remaining
    job is to catch a save that is ahead on the anchor scale yet CLEARLY behind the
    current best one-on-one (score < 0.45): that, and only that, blocks promotion.
    A level or wedged h2h no longer vetoes a clear margin winner.
    """
    total = int(wins) + int(losses) + int(draws)
    if total <= 0:
        return False
    score = (int(wins) + 0.5 * int(draws)) / total
    return score < (1.0 - threshold)


class Best_tracker:
    """
    Docstring for Best_tracker class:

    Holds the best-so-far save and decides, from each live check, whether the new
    save becomes best and whether to roll back.

    best rule (step 6; revised): a save becomes best when its live Elo clears the
    current best by more than best_margin (the ~25-Elo noise guard). A head-to-head
    vs the current best is folded in only as a veto on a FALSE positive: if the
    challenger clears the margin on the anchor scale but clearly LOSES the h2h
    (score < 1 - gate, i.e. < 0.45), it is not promoted. A level, winning, missing
    or wedged h2h does NOT block a clear margin winner - requiring it to also WIN
    the h2h (> 0.55) on a noisy 20-game match was what froze best.pt at epoch 0 for
    all 200 epochs of long run 2. The first save checked becomes best outright.

    rollback rule (step 6): once there is a best, a check whose live Elo is more
    than ROLLBACK_ELO_DROP below the best's Elo is a "down" check. After
    ROLLBACK_CHECKS consecutive down checks, roll back to best.pt. One check at or
    above the threshold clears the streak.

    The tracker never touches Godot or files itself; it returns decisions the
    training loop acts on (saving best.pt, loading it back). Keeping the policy
    here makes it unit-testable against hand-chosen Elo sequences.
    """

    def __init__(self, elo_drop=ROLLBACK_ELO_DROP, rollback_checks=ROLLBACK_CHECKS,
                 gate=BEST_GATE_WIN_RATE, best_margin=BEST_MARGIN_ELO):
        self.elo_drop = float(elo_drop)
        self.rollback_checks = int(rollback_checks)
        self.gate = float(gate)
        # Elo margin a challenger must clear ABOVE the current best to replace it,
        # the authoritative noise guard (Phase 11 gate-noise finding). The h2h is
        # folded in only to veto a clear one-on-one loser, not as a second gate.
        self.best_margin = float(best_margin)
        self.best_elo = None     # Elo of the current best
        self.best_epoch = None   # epoch the current best was saved at
        self.down_streak = 0     # consecutive checks far below the best

    def consider(self, epoch, elo, head_to_head_record=None):
        """
        Docstring for consider:

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
        # the 25-Elo margin IS the noise guard: a challenger that clears the best
        # by MORE than best_margin is promoted. The head-to-head is no longer a
        # hard veto (it used to require > 55%, which a noisy 20-game h2h fails even
        # when the challenger is far ahead on anchor Elo - long run 2 froze best.pt
        # at epoch 0 for all 200 epochs this way). It now only blocks a save that
        # clears the margin but is CLEARLY behind one-on-one (score < 0.45); a
        # level, winning, missing or wedged h2h lets a clear margin winner through.
        if elo > self.best_elo + self.best_margin:
            h2h_clear_loss = (head_to_head_record is not None
                              and loses_head_to_head(*head_to_head_record, self.gate))
            if not h2h_clear_loss:
                old = self.best_elo
                self.best_elo = float(elo)
                self.best_epoch = int(epoch)
                became_best = True
                if head_to_head_record is not None:
                    w, l, d = head_to_head_record
                    reason = (f"new best at epoch {epoch}: elo {elo:.1f} > {old:.1f}"
                              f"+{self.best_margin:.0f} and h2h W{w} L{l} D{d} not a clear loss")
                else:
                    reason = (f"new best at epoch {epoch}: elo {elo:.1f} > {old:.1f}"
                              f"+{self.best_margin:.0f} (margin cleared, no h2h)")

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
