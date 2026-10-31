"""
League strength evaluation (Phase 6).

Plays saved checkpoints against each other and against the fixed held-out
opponents (random, chaser, defender, replay), turns the match results into Elo
ratings (elo.py), and writes a CSV. Also reports the win rate of the newest
checkpoint against each held-out opponent and how often the replay "winning
move" still scores, which is the Phase 6 / 7 T5 check.

Only checkpoints from AFTER the Phase 2 action change are eligible: the action
shape is angle+power (checkpoint tag action_format == protocol v3), and
`load_state_dict` refuses an older save. Pass checkpoints as EXPLICIT paths, not
a glob, so an old or deleted one is never swept in silently; an ineligible path
is skipped with a printed reason rather than crashing the whole run.

One dedicated Godot game is used (AI_VS_AI): the "A" checkpoint plays the learner
side and the "B" player (another checkpoint or a held-out opponent) plays Team B,
exactly as evaluate.py pits the learner against one opponent. So A vs B is one env
with learner = A and opponent = B.

  python league_eval.py --checkpoints checkpoints/ppo_epoch_20.pt checkpoints/ppo_epoch_40.pt \
      --games 20 --out runs/league/elo.csv --shm %TEMP%\bobble_league_league.dat

Start a Godot game on that shm first (or point --shm at a running one); this
script does not launch Godot itself, to keep the detached-launch policy in the
caller's hands (see scripts/smoke_test_phase6.ps1).
"""
import argparse
import os

from bobble_league_env import Bobble_league_env
from elo import Elo_table
from opponents import (
    PolicyOpponent,
    TEST_ONLY_OPPONENTS,
    load_state_dict,
    make_test_opponent,
)

# which held-out opponents anchor the Elo scale (fixed, never change)
ANCHOR_OPPONENTS = TEST_ONLY_OPPONENTS
# pin the fixed opponents at these ratings so the scale does not drift run to run
ANCHOR_RATINGS = {"random": 800.0, "defender": 950.0, "chaser": 1000.0, "replay": 850.0}


def checkpoint_label(path):
    """Docstring for checkpoint_label: a short, stable name for a checkpoint path."""
    return os.path.basename(path)


# best-checkpoint gate (R37): a new checkpoint only becomes "best" if it beats the
# previous best in strictly more than this fraction of their head-to-head games.
BEST_GATE_WIN_RATE = 0.55


def beats_previous_best(wins, losses, draws, threshold=BEST_GATE_WIN_RATE):
    """
    Docstring for beats_previous_best:

    Decide whether a challenger becomes the new "best" (R37). wins/losses/draws are
    the challenger's head-to-head record against the current best; a draw counts as
    half a win, matching the Elo score. Returns True when that score rate is
    strictly above `threshold` (default 0.55 over the games played). With no games
    it returns False (no evidence to promote on).
    """
    total = int(wins) + int(losses) + int(draws)
    if total <= 0:
        return False
    score = (int(wins) + 0.5 * int(draws)) / total
    return score > threshold


def eligible_checkpoints(paths):
    """
    Docstring for eligible_checkpoints:

    Keep only the checkpoints that load under the current action format (post
    Phase 2). Returns (good, skipped) where good is a list of (label, state_dict)
    and skipped is a list of (path, reason). An ineligible or missing checkpoint
    is skipped with its reason, never crashing the run.
    """
    good, skipped = [], []
    for p in paths:
        try:
            sd = load_state_dict(p)
        except (ValueError, FileNotFoundError, RuntimeError) as exc:
            skipped.append((p, str(exc)))
            continue
        good.append((checkpoint_label(p), sd))
    return good, skipped


def play_match(env, agent, n_games, max_ep_len=200):
    """
    Docstring for play_match:

    Play n_games of the env (its opponent is already set) with `agent` on the
    learner side. Returns (wins, losses, draws) from the LEARNER's point of view:
    scorer 1 = learner win, 2 = opponent win, else a draw/timeout.

    A turn that never settles (TimeoutError from the bridge) counts as a draw and
    the game is reset, so one wedged game cannot block the whole round-robin — the
    single-game env has no crash-restart of its own, unlike Vector_env.
    """
    wins = losses = draws = 0
    obs, info = env.reset()
    for _ in range(n_games):
        done = False
        while not done:
            action = agent.act(env._raw_obs)
            try:
                obs, r, terminated, truncated, step_info = env.step(action)
            except TimeoutError:
                # a turn did not settle: score the game as a draw and reset
                draws += 1
                done = True
                try:
                    obs, info = env.reset()
                except TimeoutError:
                    pass
                break
            if terminated or truncated:
                scorer = step_info.get("scorer", 0) if terminated else 0
                if scorer == 1:
                    wins += 1
                elif scorer == 2:
                    losses += 1
                else:
                    draws += 1
                done = True
                obs, info = env.reset()
    return wins, losses, draws


def run_league(checkpoint_paths, shm_path=None, n_games=20, max_ep_len=200,
               out_csv=None, k=32.0, seed=0, step_timeout=30.0):
    """
    Docstring for run_league:

    Round-robin of the eligible checkpoints plus the held-out anchors, folded into
    an Elo table. Each unordered pair of checkpoints plays n_games; each checkpoint
    also plays n_games against every anchor. Returns (elo_table, report) where
    report carries the per-anchor win rates of the newest checkpoint and the
    replay score rate (the Phase 6/7 T5 numbers). Writes out_csv when given.
    """
    if shm_path:
        os.environ["BOBBLE_SHM_PATH"] = shm_path

    good, skipped = eligible_checkpoints(checkpoint_paths)
    for path, reason in skipped:
        print(f"skip {path}: {reason}")
    if not good:
        raise SystemExit("no eligible (post-Phase-2) checkpoints to evaluate")

    table = Elo_table(k=k)
    for name in ANCHOR_OPPONENTS:
        table.add_player(name, rating=ANCHOR_RATINGS.get(name, 1000.0), anchor=True)
    for label, _ in good:
        table.add_player(label)

    # build the agents (learner-side policies) and the anchor opponents (Team B)
    agents = {label: PolicyOpponent(sd, deterministic=True, name=label) for label, sd in good}
    anchors = {name: make_test_opponent(name, seed=seed) for name in ANCHOR_OPPONENTS}

    env = Bobble_league_env(opponent=None, max_ep_len=max_ep_len, step_timeout=step_timeout)

    per_anchor = {}  # newest checkpoint win rate vs each anchor
    newest = good[-1][0]
    previous = good[-2][0] if len(good) >= 2 else None
    best_gate = None  # (wins, losses, draws, is_best) of newest vs previous
    try:
        # checkpoint vs checkpoint (each unordered pair once)
        labels = [lbl for lbl, _ in good]
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                a, b = labels[i], labels[j]
                env.opponent = agents[b]
                w, l, d = play_match(env, agents[a], n_games, max_ep_len)
                table.record_match(a, b, w, l, d)
                print(f"{a} vs {b}: W{w} L{l} D{d}")
                # best gate: newest (a==newest) head-to-head vs the previous best
                if a == previous and b == newest:
                    # record is for `a` vs `b`; flip to the newest's point of view
                    best_gate = (l, w, d, beats_previous_best(l, w, d))
                elif a == newest and b == previous:
                    best_gate = (w, l, d, beats_previous_best(w, l, d))

        # checkpoint vs each held-out anchor
        for label, _ in good:
            for name in ANCHOR_OPPONENTS:
                env.opponent = anchors[name]
                w, l, d = play_match(env, agents[label], n_games, max_ep_len)
                table.record_match(label, name, w, l, d)
                print(f"{label} vs {name}: W{w} L{l} D{d}")
                if label == newest:
                    total = max(1, w + l + d)
                    per_anchor[name] = w / total
    finally:
        env.close()

    report = {
        "newest": newest,
        "previous": previous,
        "best_gate": best_gate,
        "per_anchor_winrate": per_anchor,
        "replay_score_rate": 1.0 - per_anchor.get("replay", 0.0),
        "skipped": skipped,
    }
    if out_csv:
        os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
        table.to_csv(out_csv)
        print(f"wrote {out_csv}")
    return table, report


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoints", nargs="+", required=True,
                        help="EXPLICIT checkpoint paths (post-Phase-2); a glob is not used on purpose")
    parser.add_argument("--games", type=int, default=20, help="games per pairing")
    parser.add_argument("--max-ep-len", type=int, default=200)
    parser.add_argument("--shm", default=None, help="shm path of the running Godot game")
    parser.add_argument("--out", default="runs/league/elo.csv", help="CSV output path")
    parser.add_argument("--k", type=float, default=32.0, help="Elo K-factor")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--step-timeout", type=float, default=30.0,
                        help="seconds to wait for one turn before counting the game a draw")
    args = parser.parse_args()

    table, report = run_league(args.checkpoints, shm_path=args.shm, n_games=args.games,
                               max_ep_len=args.max_ep_len, out_csv=args.out, k=args.k,
                               seed=args.seed, step_timeout=args.step_timeout)

    print("\n== Elo (best first) ==")
    for name, rating in table.ranked():
        tag = " [anchor]" if name in table.anchors else ""
        print(f"  {name:28s} {rating:7.1f}{tag}")

    print(f"\n== newest checkpoint: {report['newest']} ==")
    for name, wr in report["per_anchor_winrate"].items():
        print(f"  vs {name:10s} win rate {wr:.0%}")
    print(f"  replay 'winning move' score rate: {report['replay_score_rate']:.0%}")
    if report["best_gate"] is not None:
        w, l, d, is_best = report["best_gate"]
        verdict = "becomes best" if is_best else "not best"
        print(f"\n== best gate: {report['newest']} vs {report['previous']} "
              f"W{w} L{l} D{d} -> {verdict} (>{BEST_GATE_WIN_RATE:.0%} needed) ==")


if __name__ == "__main__":
    main()
