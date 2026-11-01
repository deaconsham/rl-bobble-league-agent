"""
Warm start the policy from the scripted chaser (phase 4).

The first PPO epochs are mostly spent learning the obvious thing the chaser
already does: point at the ball and shoot. So before PPO starts we clone the
chaser into the network, then hand PPO a warm checkpoint to begin from.

Two steps:
  1. record (obs, action) pairs by driving the LEARNER slot with a ChaserOpponent
     over the N parallel games (opponent is the chaser too), the way phase 1 set
     the games up. The learner-frame raw obs goes to the chaser, so the recorded
     action is already in the learner frame the network trains in. Returns are
     computed with the same GAE/discount the training loop uses, to fit the critic.
  2. behaviour clone: train the policy mean to match the recorded actions (MSE)
     and the critic to match the recorded returns (MSE), then raise log_std back
     to its init value so PPO still explores.

The result is saved in the full checkpoint format (phase 1), so
`training_loop.py --init warm_start.pt` picks the weights up (and the training
loop seeds the self-play pool from them).

The INIT_LOG_STD here must match Actor_critic_neural_network's log_std init, so
cloning does not quietly lock the policy into near-deterministic chaser play.
"""
import os

import numpy as np
import torch
import torch.nn as nn

from bobble_league_env import (
    RewardConfig, Vector_env, normalize_obs, opponent_view, wrap_action,
)
from neural_network import Actor_critic_neural_network, INIT_LOG_STD
from opponents import ChaserOpponent, SelfPlayPool
from training_loop import (
    PPO_buffer, get_device, save_checkpoint,
)

# INIT_LOG_STD is imported from neural_network so the warm start keeps the SAME
# small spread the network starts at (-1.6, sigma ~0.2). cloning drives the mean,
# not the spread, but we reset log_std to this value after cloning so a wide
# spread is never baked into the warm checkpoint. keeping it one shared constant
# means the two files cannot drift apart (Phase 10, step 2).


def collect_demos(vec_env, chaser_factory, n_turns, gamma=0.99, lam=0.97,
                  max_ep_len=1000, verbose=True):
    """
    Docstring for collect_demos:

    Play the N games with the chaser on BOTH sides and record what the chaser
    does from the learner's point of view.

    The learner slot is driven here (not by the env's opponent) using one chaser
    per game, fed the learner-frame raw obs, so each recorded action is already in
    the frame the network learns in. The env's own opponents (also chasers) play
    the other side. One PPO_buffer per game gives GAE returns for the critic, with
    paths kept apart exactly like training.

    Args:
        vec_env: a ready Vector_env (real games via a launcher, or mock games).
        chaser_factory: callable returning a fresh ChaserOpponent (one per game).
        n_turns: TOTAL turns to record across all games.
        gamma, lam: GAE discount / lambda (match the training loop).
        max_ep_len: truncate an episode after this many turns.
        verbose: print progress.

    Returns:
        dict with numpy arrays 'obs' (M, 14), 'act' (M, 6) and 'ret' (M,), where
        M is n_turns rounded down to a whole number per game.
    """
    n_games = vec_env.n_games
    obs_dim = vec_env.observation_space.shape[0]
    act_dim = vec_env.action_space.shape[0]

    per_game = max(1, n_turns // n_games)
    buffers = [PPO_buffer(obs_dim, act_dim, per_game, gamma, lam) for _ in range(n_games)]
    # one chaser per game drives the LEARNER side; its action is what we clone.
    learner_bots = [chaser_factory() for _ in range(n_games)]

    o, infos = vec_env.reset()
    ep_len = np.zeros(n_games, dtype=np.int64)

    for t in range(per_game):
        # the chaser reads RAW obs in its own frame; the learner frame IS its own
        # frame here, so feed the games' current raw obs straight to it.
        actions = np.zeros((n_games, act_dim), dtype=np.float32)
        for g in range(n_games):
            raw = vec_env._raw_obs[g]
            # wrap the angle slots, clip the power slots (Phase 10, step 3). the
            # chaser's angles are already in range (atan2/pi), so this is a no-op
            # for it, but it keeps every clip site consistent.
            actions[g] = wrap_action(learner_bots[g].act(raw))

        next_o, r, terminated, truncated, step_infos = vec_env.step(actions)

        for g in range(n_games):
            # value 0: the critic is fit to returns afterwards, so the stored
            # value only feeds GAE here. with a scripted policy the advantage is
            # unused (we clone the action, not the advantage), so 0 is fine and
            # returns are then plain discounted reward-to-go.
            buffers[g].store(o[g], actions[g], r[g], 0.0, 0.0)
            ep_len[g] += 1

        o = next_o
        epoch_ended = t == per_game - 1
        for g in range(n_games):
            term = bool(terminated[g])
            trunc = bool(truncated[g]) or ep_len[g] >= max_ep_len
            if term or trunc or epoch_ended:
                buffers[g].finish_path(0.0)
                if term or trunc:
                    ep_len[g] = 0
                    if learner_bots[g] is not None and hasattr(learner_bots[g], "reset"):
                        learner_bots[g].reset()

        if verbose and per_game >= 10 and (t + 1) % max(1, per_game // 10) == 0:
            print(f"  recorded {(t + 1) * n_games}/{per_game * n_games} turns", flush=True)

    data = [buf.get(device="cpu", normalize_adv=False) for buf in buffers]
    obs = torch.cat([d["obs"] for d in data], dim=0).numpy()
    act = torch.cat([d["act"] for d in data], dim=0).numpy()
    ret = torch.cat([d["ret"] for d in data], dim=0).numpy()
    return {"obs": obs.astype(np.float32),
            "act": act.astype(np.float32),
            "ret": ret.astype(np.float32)}


def behaviour_clone(ac, demos, epochs=10, batch_size=256, pi_lr=3e-4, vf_lr=1e-3,
                    device="cpu", reset_log_std=True, verbose=True):
    """
    Docstring for behaviour_clone:

    Fit the policy mean to the recorded actions (MSE) and the critic to the
    recorded returns (MSE), in minibatches over several epochs. Supervised, so no
    PPO ratio, no clipping, no entropy.

    After cloning, log_std is reset to its init value (INIT_LOG_STD) so PPO starts
    with real exploration noise rather than whatever cloning left it at. Cloning
    only touches the mean (pi_net) and the critic (v_net), but a reset makes the
    intent explicit and survives any future change that does touch it.

    Args:
        ac: the Actor_critic_neural_network to clone into (modified in place).
        demos: dict with 'obs' (M,14), 'act' (M,6), 'ret' (M,) numpy arrays.
        epochs: passes over the recorded set.
        batch_size: minibatch size.
        pi_lr, vf_lr: learning rates for the policy and critic.
        device: torch device.
        reset_log_std: raise log_std back to INIT_LOG_STD after cloning.
        verbose: print the loss each epoch.

    Returns:
        a list of (pi_loss, v_loss) per epoch, so a test can check the loss drops.
    """
    device = get_device(device) if isinstance(device, str) else device
    ac.to(device)

    obs = torch.as_tensor(demos["obs"], dtype=torch.float32, device=device)
    act = torch.as_tensor(demos["act"], dtype=torch.float32, device=device)
    ret = torch.as_tensor(demos["ret"], dtype=torch.float32, device=device)
    n = obs.shape[0]
    mb = min(batch_size, n)

    # clone only the mean and the critic; the policy head is ac.pi_net, the critic
    # ac.v_net. log_std is left out of the optimizers and reset at the end.
    pi_optimizer = torch.optim.Adam(ac.pi_net.parameters(), lr=pi_lr)
    vf_optimizer = torch.optim.Adam(ac.v_net.parameters(), lr=vf_lr)
    mse = nn.MSELoss()

    history = []
    for epoch in range(epochs):
        idx = torch.randperm(n, device=device)
        pi_total, v_total, batches = 0.0, 0.0, 0
        for start in range(0, n, mb):
            mb_idx = idx[start:start + mb]
            o_mb, a_mb, r_mb = obs[mb_idx], act[mb_idx], ret[mb_idx]

            # policy: MSE between the network mean and the recorded action
            pi_optimizer.zero_grad()
            mu = ac.pi_net(o_mb)
            pi_loss = mse(mu, a_mb)
            pi_loss.backward()
            pi_optimizer.step()

            # critic: MSE between the value estimate and the recorded return
            vf_optimizer.zero_grad()
            value = ac.v_net(o_mb).squeeze(-1)
            v_loss = mse(value, r_mb)
            v_loss.backward()
            vf_optimizer.step()

            pi_total += pi_loss.item()
            v_total += v_loss.item()
            batches += 1

        pi_avg = pi_total / max(1, batches)
        v_avg = v_total / max(1, batches)
        history.append((pi_avg, v_avg))
        if verbose:
            print(f"  clone epoch {epoch}: pi_mse {pi_avg:.5f}  v_mse {v_avg:.5f}", flush=True)

    if reset_log_std:
        with torch.no_grad():
            ac.log_std.fill_(INIT_LOG_STD)

    return history


def save_warm_checkpoint(path, ac, pi_lr=3e-4, vf_lr=1e-3):
    """
    Docstring for save_warm_checkpoint:

    Write the warm-started network as a FULL checkpoint (phase 1 format) so
    `training_loop.py --init <path>` loads it. Fresh optimizers are saved (cloning
    used its own supervised optimizers, which do not carry into PPO), epoch and
    lr_step are 0, and the action_format tag is set, so it reads like the start of
    a fresh run.
    """
    pi_optimizer = torch.optim.Adam(list(ac.pi_net.parameters()) + [ac.log_std], lr=pi_lr)
    vf_optimizer = torch.optim.Adam(ac.v_net.parameters(), lr=vf_lr)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    return save_checkpoint(path, ac, pi_optimizer, vf_optimizer, epoch=0, lr_step=0,
                           opponent=None, save_rng=True)


def warm_start(vec_env, out_path, n_turns=12000, clone_epochs=30, gamma=0.99,
               lam=0.97, max_ep_len=1000, device="cpu", chaser_kwargs=None,
               verbose=True):
    """
    Docstring for warm_start:

    End to end: record chaser demos over the given Vector_env, clone them into a
    fresh network, reset log_std, and save the warm checkpoint. Returns
    (ac, demos, history) for inspection and testing.

    The demo chaser (the one driving the LEARNER slot, whose actions become the
    clone target) defaults to DETERMINISTIC: noise=0 and kickoff_noise=0. MSE
    cloning fits the mean, so a noisy target would regress to a blurred average
    that plays worse than the chaser it came from; a clean, repeatable target is
    what the mean can actually reproduce. The env's own opponents stay the normal
    noisy chaser, so the recorded STATES still vary. Override with chaser_kwargs.
    """
    chaser_kwargs = {"noise": 0.0, "kickoff_noise": 0.0} if chaser_kwargs is None else chaser_kwargs

    def chaser_factory():
        return ChaserOpponent(**chaser_kwargs)

    if verbose:
        print(f"recording ~{n_turns} chaser turns over {vec_env.n_games} games...", flush=True)
    demos = collect_demos(vec_env, chaser_factory, n_turns, gamma=gamma, lam=lam,
                          max_ep_len=max_ep_len, verbose=verbose)
    if verbose:
        print(f"recorded {demos['obs'].shape[0]} turns; cloning...", flush=True)

    obs_dim = vec_env.observation_space.shape[0]
    act_dim = vec_env.action_space.shape[0]
    ac = Actor_critic_neural_network(obs_dim=obs_dim, act_dim=act_dim)
    history = behaviour_clone(ac, demos, epochs=clone_epochs, device=device, verbose=verbose)

    save_warm_checkpoint(out_path, ac)
    if verbose:
        print(f"warm checkpoint saved to {out_path}", flush=True)
    return ac, demos, history


def main():
    """Docstring for main: record + clone + save against N real Godot games."""
    import argparse

    from launch_games import Game_launcher

    parser = argparse.ArgumentParser(
        description="Warm start the policy from the chaser (record demos, behaviour clone, save).")
    parser.add_argument("--out", default="checkpoints/warm_start.pt",
                        help="where to write the warm checkpoint")
    parser.add_argument("--games", type=int, default=6, help="parallel games")
    parser.add_argument("--turns", type=int, default=12000, help="total chaser turns to record")
    parser.add_argument("--clone-epochs", type=int, default=30, help="behaviour-cloning passes")
    parser.add_argument("--max-ep-len", type=int, default=1000)
    parser.add_argument("--device", default=None)
    parser.add_argument("--timescale", type=int, default=32)
    parser.add_argument("--godot", default=None, help="Godot console exe")
    parser.add_argument("--project", default=None, help="bobble_league project path")
    parser.add_argument("--shm-prefix", default=None, help="folder for per-game shm files")
    parser.add_argument("--step-timeout", type=float, default=120.0)
    parser.add_argument("--random-init", default=None, metavar="PATH",
                        help="save a FRESH random-init net to PATH and exit (the T5 "
                             "baseline to compare the warm start against); no Godot")
    args = parser.parse_args()

    # the T5 baseline: an untrained net saved in the same checkpoint format, so
    # evaluate.py can play it vs the chaser the same way. no games needed.
    if args.random_init:
        ac = Actor_critic_neural_network(obs_dim=14, act_dim=6)
        save_warm_checkpoint(args.random_init, ac)
        print(f"random-init baseline saved to {args.random_init}")
        return

    launcher = Game_launcher(args.games, godot_exe=args.godot, project_path=args.project,
                             shm_prefix=args.shm_prefix, log_dir="logs", timescale=args.timescale)
    # the env's own opponents play the OTHER side, also chasers.
    opponents = [ChaserOpponent() for _ in range(args.games)]
    vec_env = Vector_env(args.games, launcher=launcher, opponents=opponents,
                         reward_config=RewardConfig(), max_ep_len=args.max_ep_len,
                         step_timeout=args.step_timeout, auto_launch=True)
    try:
        warm_start(vec_env, args.out, n_turns=args.turns, clone_epochs=args.clone_epochs,
                   max_ep_len=args.max_ep_len, device=args.device)
    finally:
        vec_env.close()


if __name__ == "__main__":
    main()
