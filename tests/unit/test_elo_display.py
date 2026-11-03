"""
T1: the play-time Elo lookup (evaluate.checkpoint_elo).

Shows the right number when a sidecar '<checkpoint>.elo.json' is present, falls
back to an 'elo' key inside the checkpoint, and returns None (so play shows
"unrated") when the number is missing, the file is unreadable, or 'elo' is bad.
The matching Godot HUD reads the same sidecar; its missing/bad handling is the
C# ReadEloSidecar, not covered here.
"""
import json

import torch

from evaluate import checkpoint_elo


def _write_ckpt(path, extra=None):
    """A minimal checkpoint dict torch can load; extra merges in metadata."""
    ckpt = {"model": {}, "action_format": 5}
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, path)
    return str(path)


def test_sidecar_present(tmp_path):
    ckpt = _write_ckpt(tmp_path / "best_final.pt")
    (tmp_path / "best_final.elo.json").write_text(
        json.dumps({"elo": 1104, "source": "Phase 16 T5"}), encoding="utf-8")
    assert checkpoint_elo(ckpt) == 1104


def test_sidecar_float_is_rounded(tmp_path):
    ckpt = _write_ckpt(tmp_path / "best_final.pt")
    (tmp_path / "best_final.elo.json").write_text(
        json.dumps({"elo": 1103.6}), encoding="utf-8")
    assert checkpoint_elo(ckpt) == 1104


def test_no_sidecar_no_metadata_is_none(tmp_path):
    ckpt = _write_ckpt(tmp_path / "best_final.pt")
    assert checkpoint_elo(ckpt) is None


def test_bad_sidecar_json_is_none(tmp_path):
    ckpt = _write_ckpt(tmp_path / "best_final.pt")
    (tmp_path / "best_final.elo.json").write_text("not json {", encoding="utf-8")
    assert checkpoint_elo(ckpt) is None


def test_sidecar_without_elo_key_is_none(tmp_path):
    ckpt = _write_ckpt(tmp_path / "best_final.pt")
    (tmp_path / "best_final.elo.json").write_text(
        json.dumps({"source": "nowhere"}), encoding="utf-8")
    assert checkpoint_elo(ckpt) is None


def test_sidecar_non_numeric_elo_is_none(tmp_path):
    ckpt = _write_ckpt(tmp_path / "best_final.pt")
    (tmp_path / "best_final.elo.json").write_text(
        json.dumps({"elo": "high"}), encoding="utf-8")
    assert checkpoint_elo(ckpt) is None


def test_checkpoint_metadata_fallback(tmp_path):
    # no sidecar, but the checkpoint itself carries an 'elo' tag
    ckpt = _write_ckpt(tmp_path / "best_final.pt", extra={"elo": 1156})
    assert checkpoint_elo(ckpt) == 1156


def test_sidecar_wins_over_metadata(tmp_path):
    ckpt = _write_ckpt(tmp_path / "best_final.pt", extra={"elo": 1156})
    (tmp_path / "best_final.elo.json").write_text(
        json.dumps({"elo": 1104}), encoding="utf-8")
    assert checkpoint_elo(ckpt) == 1104
