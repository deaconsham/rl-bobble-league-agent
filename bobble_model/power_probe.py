"""
Live check for Phase 2: a power of 0.2 must move a player clearly less than a
power of 1.0. Attaches to one running Godot (same --shm / BOBBLE_SHM_PATH rule
as evaluate.py), shoots player 0 straight at the enemy goal at each power over
several turns, and reports the mean displacement per power.

Start a headless Godot in AI_VS_AI first (the smoke script does this), then:
  BOBBLE_SHM_PATH=<file> python power_probe.py --turns 6
Exit code 0 only if the 0.2 displacement is clearly under the 1.0 one.
"""
import argparse

import numpy as np

from bobble_league_env import Bobble_league_env, encode_action, split_obs


def player0_pos(raw_obs):
    """player 0's (x, z) in the learner frame, from the raw obs."""
    _, friendly, _ = split_obs(raw_obs)
    return friendly[0].copy()


def shoot_at_power(env, power_number, settle_turns=1):
    """one turn: player 0 aims at the enemy goal (angle 0) at the given power,
    the other two aim back at their own goal (min power, harmless). returns how
    far player 0 moved."""
    # angle 0 = enemy goal (-z); other players angle +-1 = own goal
    angles = [0.0, 1.0, 1.0]
    powers = [power_number, -1.0, -1.0]
    action = encode_action(angles, powers)

    before = player0_pos(env._raw_obs)
    env.step(action)
    after = player0_pos(env._raw_obs)
    # settle (player 0 stays put at min power) so the next measure starts at rest
    for _ in range(settle_turns):
        env.step(encode_action([1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]))
    return float(np.linalg.norm(after - before))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--turns", type=int, default=6, help="shots per power to average")
    parser.add_argument("--ratio", type=float, default=2.0,
                        help="require mean(full) >= ratio * mean(soft)")
    args = parser.parse_args()

    env = Bobble_league_env(opponent=None, max_ep_len=10_000, step_timeout=120.0)
    soft, full = [], []
    try:
        env.reset()
        for _ in range(args.turns):
            soft.append(shoot_at_power(env, power_number=-1.0))   # 0.2
            full.append(shoot_at_power(env, power_number=1.0))    # 1.0
    finally:
        env.close()

    soft_mean = float(np.mean(soft))
    full_mean = float(np.mean(full))
    print(f"power 0.2: mean move {soft_mean:.3f} over {len(soft)} shots {np.round(soft, 3).tolist()}")
    print(f"power 1.0: mean move {full_mean:.3f} over {len(full)} shots {np.round(full, 3).tolist()}")

    ok = full_mean >= args.ratio * soft_mean and soft_mean < full_mean
    print(f"POWER CHECK {'PASS' if ok else 'FAIL'}: "
          f"full {full_mean:.3f} vs soft {soft_mean:.3f} "
          f"(ratio {full_mean / max(soft_mean, 1e-6):.2f}x, need >= {args.ratio}x)")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
