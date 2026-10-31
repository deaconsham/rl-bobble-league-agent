"""
Elo ratings for the league.

Pure maths, no Godot and no torch, so it can be unit tested on hand-worked
numbers. league_eval.py plays the games and feeds the match results in here.

Elo in one line: each player has a rating; the expected score of A against B is
a logistic of their rating gap, and after a game each rating moves by
`k * (actual - expected)`. A draw counts as half a win to both. Ratings are
anchored by the fixed opponents (random, chaser) that never change, so the
checkpoint numbers are comparable across runs.
"""
import csv

# standard Elo constants: 400 sets how a rating gap maps to a win probability
# (a 400-point lead is ~10:1), K is how far one game can move a rating.
ELO_SCALE = 400.0
DEFAULT_K = 32.0
DEFAULT_RATING = 1000.0


def expected_score(rating_a, rating_b):
    """Docstring for expected_score: A's expected result (0..1) against B.

    The logistic curve: 0.5 at equal ratings, toward 1 as A leads, toward 0 as
    A trails. expected_score(a, b) + expected_score(b, a) == 1 always."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / ELO_SCALE))


def update_pair(rating_a, rating_b, score_a, k=DEFAULT_K):
    """Docstring for update_pair: one game's rating move for both players.

    score_a is A's result: 1.0 win, 0.0 loss, 0.5 draw. B's score is 1 - score_a.
    Returns the new (rating_a, rating_b). Rating mass is conserved: what A gains
    B loses."""
    exp_a = expected_score(rating_a, rating_b)
    exp_b = 1.0 - exp_a
    score_b = 1.0 - score_a
    new_a = rating_a + k * (score_a - exp_a)
    new_b = rating_b + k * (score_b - exp_b)
    return new_a, new_b


class Elo_table:
    """
    Docstring for Elo_table class:

    Holds a rating per named player and folds match results into it. Anchors are
    players whose rating is pinned (the fixed random / chaser baselines), so the
    scale does not drift between runs and the learner's numbers mean the same
    thing every time.

    Feed results one of two ways:
      - record_game(a, b, score_a): a single game, updated straight away;
      - record_match(a, b, wins_a, wins_b, draws): a batch of games between the
        same pair, applied game by game (so the maths matches many record_game
        calls).
    An anchored player is read in the expected-score maths but its rating is not
    written, so only the non-anchored (checkpoint) ratings move.
    """

    def __init__(self, k=DEFAULT_K, default_rating=DEFAULT_RATING):
        self.k = float(k)
        self.default_rating = float(default_rating)
        self.ratings = {}
        self.anchors = set()

    def add_player(self, name, rating=None, anchor=False):
        """Docstring for add_player: register a player, optionally anchored."""
        self.ratings[name] = self.default_rating if rating is None else float(rating)
        if anchor:
            self.anchors.add(name)
        return self.ratings[name]

    def rating(self, name):
        """Docstring for rating: current rating, defaulting an unseen player."""
        if name not in self.ratings:
            self.ratings[name] = self.default_rating
        return self.ratings[name]

    def record_game(self, a, b, score_a):
        """Docstring for record_game: fold one game (score_a in {0, 0.5, 1})."""
        ra, rb = self.rating(a), self.rating(b)
        new_a, new_b = update_pair(ra, rb, score_a, self.k)
        # an anchor's rating is pinned: read for the expected score, never written
        if a not in self.anchors:
            self.ratings[a] = new_a
        if b not in self.anchors:
            self.ratings[b] = new_b

    def record_match(self, a, b, wins_a, wins_b, draws=0):
        """Docstring for record_match: fold a batch of games between one pair.

        Applied one game at a time (wins_a wins, then wins_b losses, then draws),
        so the result is identical to that sequence of record_game calls."""
        for _ in range(int(wins_a)):
            self.record_game(a, b, 1.0)
        for _ in range(int(wins_b)):
            self.record_game(a, b, 0.0)
        for _ in range(int(draws)):
            self.record_game(a, b, 0.5)

    def ranked(self):
        """Docstring for ranked: (name, rating) pairs, best rating first."""
        return sorted(self.ratings.items(), key=lambda kv: kv[1], reverse=True)

    def to_csv(self, path):
        """Docstring for to_csv: write name,rating,anchor rows, best first."""
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["name", "rating", "anchor"])
            for name, rating in self.ranked():
                writer.writerow([name, f"{rating:.1f}", int(name in self.anchors)])
        return path
