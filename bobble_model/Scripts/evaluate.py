"""
Evaluate a trained checkpoint, or play against it. No PPO buffer, no training.
Start the Godot game first and pick the mode on GameManager.CurrentPlayMode.
To just play, Scripts/play.py starts the game and this for you.

  AI_VS_AI    : python Scripts/evaluate.py --opponent chaser --episodes 20
                python Scripts/evaluate.py --opponent Models/ppo_epoch_10.pt
  HUMAN_VS_AI : python Scripts/evaluate.py --human
                (you are Team A: left-drag to aim, right-click to end your turn)

With no --checkpoint, both modes default to Models/best_final.pt (the
shipped champion), falling back to the newest ppo_epoch_*.pt.
"""
import argparse
import glob
import os
import re

import numpy as np

from bobble_league_env import Bobble_league_env
from opponents import PolicyOpponent, load_state_dict, make_opponent

# bobble_model/Models, found from this file so it works from any folder
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Models')


def latest_checkpoint(folder=MODELS_DIR):
    paths = glob.glob(os.path.join(folder, 'ppo_epoch_*.pt'))
    if not paths:
        raise FileNotFoundError(f'No checkpoints in {folder}/. Train first or pass --checkpoint.')
    return max(paths, key=lambda p: int(re.search(r'ppo_epoch_(\d+)\.pt$', p).group(1)))


def default_model(folder=MODELS_DIR):
    """
    Docstring for default_model:

    The model used when --checkpoint is not given.

    The shipped champion is Models/best_final.pt, so AI
    play and --human default to it. If it is missing (a fresh clone with no final
    model yet), fall back to the newest ppo_epoch_*.pt so a just-trained run still
    just works.
    """
    final = os.path.join(folder, 'best_final.pt')
    if os.path.exists(final):
        return final
    return latest_checkpoint(folder)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', default=None,
                        help='default: Models/best_final.pt, else newest Models/ppo_epoch_*.pt')
    parser.add_argument('--episodes', type=int, default=10, help='0 = play until Ctrl+C')
    parser.add_argument('--opponent', default='chaser',
                        help='AI_VS_AI only: chaser | random | still | defender | replay | none | path/to/checkpoint.pt')
    parser.add_argument('--human', action='store_true', help='Godot is in HUMAN_VS_AI; wait for you forever')
    parser.add_argument('--stochastic', action='store_true', help='sample actions instead of using the mean')
    parser.add_argument('--max-ep-len', type=int, default=200)
    args = parser.parse_args(argv)

    ckpt = args.checkpoint or default_model()
    # --human: sample with a shrunk spread (std x0.3) so the AI is not perfectly
    # repeatable and one fixed counter does not always work. AI_VS_AI eval
    # keeps the deterministic mean unless --stochastic is passed.
    if args.human:
        agent = PolicyOpponent(load_state_dict(ckpt), deterministic=False,
                               std_scale=0.3, name=ckpt)
    else:
        agent = PolicyOpponent(load_state_dict(ckpt), deterministic=not args.stochastic, name=ckpt)
    opponent = None if args.human else make_opponent(args.opponent, deterministic=True)

    env = Bobble_league_env(opponent=opponent, max_ep_len=args.max_ep_len,
                          step_timeout=None if args.human else 120.0)
    results = {'win': 0, 'loss': 0, 'timeout': 0}
    returns = []

    try:
        obs, info = env.reset()
        print(f'Model: {ckpt}')
        print(f"Godot mode: {info['play_mode']}, AI is team {info['learner_team']}, "
              f"opponent: {'you' if args.human else getattr(opponent, 'name', 'none')}")
        if args.human and info['play_mode'] != 'HUMAN_VS_AI':
            print('Warning: --human given but Godot is not in HUMAN_VS_AI.')

        episode = 0
        ep_ret = 0.0
        while args.episodes == 0 or episode < args.episodes:
            # the agent network expects normalized obs; PolicyOpponent normalizes raw obs,
            # so hand it the env's raw learner-frame obs
            action = agent.act(env._raw_obs)
            obs, r, terminated, truncated, step_info = env.step(action)
            ep_ret += r

            if terminated or truncated:
                scorer = step_info['scorer'] if terminated else 0
                outcome = 'win' if scorer == 1 else 'loss' if scorer == 2 else 'timeout'
                results[outcome] += 1
                returns.append(ep_ret)
                episode += 1
                print(f'Episode {episode}: {outcome}, return {ep_ret:.2f}, {env.ep_len} turns')
                ep_ret = 0.0
                obs, info = env.reset()
    except KeyboardInterrupt:
        pass
    finally:
        env.close()

    n = max(1, sum(results.values()))
    print(f"\nW {results['win']} / L {results['loss']} / T {results['timeout']}  "
          f"win rate {results['win'] / n:.0%}, mean return {np.mean(returns) if returns else 0:.2f}")


if __name__ == '__main__':
    main()
