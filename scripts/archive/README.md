# Archived one-off scripts

These are one-off scripts, each kept as the source of the numbers in one phase
report under `docs/phase_reports/`. They ran once, produced that phase's T4/T5
figures, and are kept only so those figures can be traced back to the commands
that made them.

**They are not maintained.** They may reference flags, paths or a code layout
from the phase they belong to and are not expected to run against the final
code. For a live smoke test use `scripts/smoke_test.ps1`; for a live-Elo proof
run use `scripts/proof_live_elo.ps1`.

| Script | Phase report it backs |
|---|---|
| `smoke_test_phase1.ps1` | `docs/phase_reports/phase1.md` (T4 live smoke) |
| `smoke_test_phase2.ps1` | `docs/phase_reports/phase2.md`, `phase3.md` (T4 live smoke, power probe) |
| `smoke_test_phase5.ps1` | `docs/phase_reports/phase5.md` (T4 live smoke) |
| `smoke_test_phase6.ps1` | `docs/phase_reports/phase6.md` (T4 live smoke) |
| `smoke_test_phase9.ps1` | `docs/phase_reports/phase9.md`, `review.md` (6-game T4 at one time scale) |
| `eval_epoch60.ps1` | `docs/phase_reports/phase8.md` (epoch-60 40-game Elo re-test) |
| `t5_warm_start.ps1` | `docs/phase_reports/phase4.md` (T5 warm-start / chaser demos) |
| `verify_no_oob.ps1` | `docs/phase_reports/phase8.md` (200-game out-of-bounds check) |
