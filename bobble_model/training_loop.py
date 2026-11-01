import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from neural_network import Actor_critic_neural_network
from shared_memory_link import Shared_memory_link

# the action meaning is tied to the wire protocol, so tag checkpoints with the
# same version. an old save (pre angle+power) has no tag and is refused on load.
ACTION_FORMAT_VERSION = Shared_memory_link.PROTOCOL_VERSION

def get_device(device=None):
    """
    Docstring for get_device:

    Get the device to use for training, (GPU acceleration if we have a GPU, otherwise just train using a CPU)
    
    Args:
        device: Optional device string ('cuda', 'cpu', 'mps') or None for auto-detection
    
    Returns:
        torch.device: The device to use
    """
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps') # for mac users
    else:
        return torch.device('cpu')

class PPO_buffer:
    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.97):
        """
        Docstring for PPO_buffer class:

        Buffer for storing trajectories experienced by an agent interacting 
        with the environment, and using generalized advantage estimation (GAE)
        for calculating the advantages of state-action pairs

        Args:
            obs_dim: dimension of observations
            act_dim: dimension of actions  
            size: number of steps to store before computing advantages
            gamma: discount factor for future rewards
            lambda: lambda parameter for GAE
        """
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        
        self.gamma = gamma
        self.lam = lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        """
        Docstring for store:

        Append one timestep of agent-environment interaction to the buffer
        """
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Docstring for finish_path:

        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This computes advantage estimates using GAE-Lambda (temporal difference)
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        
        # Compute TD residuals: gamma_t = r_t + gamma*V(s_{t+1}) - V(s_t)
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        
        # Compute GAE advantages using discounted sum of deltas
        self.adv_buf[path_slice] = self.discount_cumsum(deltas, self.gamma * self.lam)
        
        # Return: R_t = r_t + gamma*r_{t+1} + gamma^(2)*r_{t+2} + ...
        self.ret_buf[path_slice] = self.discount_cumsum(rews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr

    def get(self, device='cpu', normalize_adv=True):
        """
        Docstring for get:

        Call this at the end of an epoch to get all stored data. Resets pointers
        for the next epoch.

        Args:
            device: Device to place tensors on ('cpu', 'cuda', or torch.device)
            normalize_adv: normalize advantages here (single-buffer use). Pass False
                when several per-game buffers are joined and normalized together over
                the whole batch (Vector_env training).
        """
        assert self.ptr == self.max_size  # buffer has to be full
        self.ptr, self.path_start_idx = 0, 0

        # Normalize advantages: (A - mean(A)) / std(A) for training stability
        if normalize_adv:
            adv_mean = np.mean(self.adv_buf)
            adv_std = np.std(self.adv_buf)
            self.adv_buf = (self.adv_buf - adv_mean) / (adv_std + 1e-8)
        
        # Return everything as PyTorch tensors on specified device
        data = dict(
            obs=self.obs_buf, 
            act=self.act_buf, 
            ret=self.ret_buf,
            adv=self.adv_buf, 
            logp=self.logp_buf
        )
        return {k: torch.as_tensor(v, dtype=torch.float32, device=device) for k, v in data.items()}

    @staticmethod
    def discount_cumsum(x, discount):
        """
        Docstring for discount_cumsum:

        Computing discounted cumulative sums of vectors
        """
        result = np.zeros_like(x)
        result[-1] = x[-1]
        for t in reversed(range(len(x)-1)):
            result[t] = x[t] + discount * result[t+1]
        return result

class Opponent_buffer:
    """
    Docstring for Opponent_buffer class:

    Variable-length buffer for the OPPONENT (Team B) side of live-vs-live games.

    Unlike PPO_buffer (one fixed-size buffer per game, filled every turn), the
    opponent side is only stored on the turns a game is live vs live, so its
    length is not known ahead of time and differs per game. It keeps the two
    teams' paths apart exactly like PPO_buffer: store() appends turns, finish_path()
    closes the current path with GAE (bootstrapping from the critic on a cut path),
    and get() returns the same obs/act/ret/adv/logp dict update_from_data wants.

    The data is already in the opponent's own mirrored frame (the LiveOpponent
    acted there), so no frame conversion happens here.
    """

    def __init__(self, gamma=0.99, lam=0.97):
        self.gamma = gamma
        self.lam = lam
        self._reset_lists()

    def _reset_lists(self):
        self.obs, self.act, self.rew, self.val, self.logp = [], [], [], [], []
        self.adv, self.ret = [], []
        self.path_start = 0

    def store(self, obs, act, rew, val, logp):
        """Append one opponent turn (obs/act already in the opponent frame)."""
        self.obs.append(np.asarray(obs, dtype=np.float32))
        self.act.append(np.asarray(act, dtype=np.float32))
        self.rew.append(float(rew))
        self.val.append(float(val))
        self.logp.append(float(logp))

    def finish_path(self, last_val=0.0):
        """
        Close the open path with GAE-Lambda, same maths as PPO_buffer.finish_path.
        last_val bootstraps a cut (truncated) path from the critic; a real goal
        passes 0. Does nothing when no turns were stored since the last finish.
        """
        n = len(self.rew) - self.path_start
        if n <= 0:
            return
        sl = slice(self.path_start, len(self.rew))
        rews = np.append(np.asarray(self.rew[sl], dtype=np.float32), last_val)
        vals = np.append(np.asarray(self.val[sl], dtype=np.float32), last_val)
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        adv = PPO_buffer.discount_cumsum(deltas, self.gamma * self.lam)
        ret = PPO_buffer.discount_cumsum(rews, self.gamma)[:-1]
        self.adv.extend(adv.tolist())
        self.ret.extend(ret.tolist())
        self.path_start = len(self.rew)

    def __len__(self):
        return len(self.ret)

    def get(self, device='cpu'):
        """
        Return the FINISHED turns as a data dict (advantages NOT normalized here;
        join_and_normalize normalizes over the whole batch). Only fully-finished
        paths are returned, so call finish_path() before get(). Clears the buffer.
        An empty buffer returns correctly-shaped zero-row arrays.
        """
        n = len(self.ret)
        obs_dim = self.obs[0].shape[-1] if self.obs else 14
        act_dim = self.act[0].shape[-1] if self.act else 6
        data = dict(
            obs=np.asarray(self.obs[:n], dtype=np.float32).reshape(n, obs_dim),
            act=np.asarray(self.act[:n], dtype=np.float32).reshape(n, act_dim),
            ret=np.asarray(self.ret, dtype=np.float32),
            adv=np.asarray(self.adv, dtype=np.float32),
            logp=np.asarray(self.logp[:n], dtype=np.float32),
        )
        self._reset_lists()
        return {k: torch.as_tensor(v, dtype=torch.float32, device=device) for k, v in data.items()}


def compute_loss_pi(ac, data, clip_ratio=0.2):
    """
    Docstring for compute_loss_pi:

    Compute the PPO clipped surrogate objective (policy loss), thanks shawn
    """
    obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']
    
    # Get policy distribution from current policy
    pi, _ = ac(obs)
    
    # Compute log probability of actions under current policy
    logp = pi.log_prob(act)
    
    # Compute ratio: pi_theta(a|s) / pi_theta_old(a|s)
    # which is done with logs because it's easier: exp(log pi_theta - log pi_theta_old)
    ratio = torch.exp(logp - logp_old)
    
    # Clipped surrogate objective
    clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv
    loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()
    
    # Useful extra info for logging
    approx_kl = (logp_old - logp).mean().item()  # Approximate KL divergence
    ent_tensor = pi.entropy().mean()  # Policy entropy (tensor, for the entropy bonus)
    ent = ent_tensor.item()
    clipped = ratio.gt(1 + clip_ratio) | ratio.lt(1 - clip_ratio)
    clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
    
    pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac, ent_tensor=ent_tensor)
    
    return loss_pi, pi_info


def compute_loss_v(ac, data):
    """
    Docstring for compute_loss_v:

    Compute value function loss
    
    MSE between predicted values and actual returns:
    value_loss = (V(s) - R)^2
    
    Where R is the actual discounted return we got
    """
    obs, ret = data['obs'], data['ret']
    
    # Get value estimate from critic
    _, value = ac(obs)
    
    # MSE loss
    return ((value - ret) ** 2).mean()


def explained_variance(values, returns):
    """
    Docstring for explained_variance:

    How much of the return variance the critic explains: 1 - Var(ret - V) / Var(ret).
    1 is a perfect fit, 0 is no better than guessing the mean, below 0 is worse.
    Takes torch tensors or numpy arrays.
    """
    v = values.detach().cpu().numpy() if isinstance(values, torch.Tensor) else np.asarray(values)
    r = returns.detach().cpu().numpy() if isinstance(returns, torch.Tensor) else np.asarray(returns)
    var_r = np.var(r)
    if var_r == 0:
        return 0.0
    return float(1 - np.var(r - v) / var_r)


class Linear_lr_schedule:
    """
    Docstring for Linear_lr_schedule class:

    Linear learning-rate decay to 0 over the whole run. One schedule drives both
    optimizers from their own starting LRs. 'step' is the number of updates done so
    far (0-based), 'total' is the planned number of updates (epochs). The fraction
    left is (total - step) / total, so the last update runs at ~0.

    The step is saved in the checkpoint (lr_step) and restored on resume, so the
    LR picks up exactly where it left off instead of snapping back to full.
    """

    def __init__(self, optimizers, base_lrs, total, step=0):
        self.optimizers = list(optimizers)
        self.base_lrs = list(base_lrs)
        self.total = max(1, int(total))
        self.step_count = int(step)
        self.apply()

    def frac_left(self):
        """fraction of the run still to go, clamped to [0, 1]."""
        left = (self.total - self.step_count) / self.total
        return max(0.0, min(1.0, left))

    def current_lrs(self):
        """the LRs the schedule wants right now (one per optimizer)."""
        f = self.frac_left()
        return [base * f for base in self.base_lrs]

    def apply(self):
        """write the current LRs onto the optimizers' param groups."""
        for opt, lr in zip(self.optimizers, self.current_lrs()):
            for group in opt.param_groups:
                group['lr'] = lr

    def advance(self):
        """call once per update (after it) to move one step down the ramp."""
        self.step_count += 1
        self.apply()

def update(ac, buf, pi_optimizer, vf_optimizer, train_pi_iters=80, 
           train_v_iters=80, target_kl=0.02, clip_ratio=0.2, device='cpu',
           minibatch_size=256, n_passes=10, ent_coef=0.005, max_grad_norm=0.5):
    """
    Perform a PPO update on a single full buffer.

    Thin wrapper: pulls the buffer (advantages normalized for the single-game
    case) and hands it to update_from_data, so the minibatch + entropy + grad-clip
    + LR logic lives in one place (Phase 3).

    Args:
        ac: actor-critic model
        buf: PPO_buffer with trajectory data
        pi_optimizer, vf_optimizer: the two optimizers
        target_kl: per-pass KL early-stop threshold
        clip_ratio: epsilon for PPO clipping
        device: device for the tensors
        minibatch_size, n_passes, ent_coef, max_grad_norm: Phase 3 knobs
    """
    data = buf.get(device=device)
    return update_from_data(ac, data, pi_optimizer, vf_optimizer, train_pi_iters,
                            train_v_iters, target_kl, clip_ratio,
                            minibatch_size=minibatch_size, n_passes=n_passes,
                            ent_coef=ent_coef, max_grad_norm=max_grad_norm)


def rng_state():
    """
    Docstring for rng_state:

    Grab the python / numpy / torch RNG state so a resumed run continues the same
    random stream (opponent picks, action samples).
    """
    import random as _random
    return {
        'python': _random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
    }


def set_rng_state(state):
    """Docstring for set_rng_state: restore the RNG state saved by rng_state()."""
    import random as _random
    if not state:
        return
    if 'python' in state:
        _random.setstate(state['python'])
    if 'numpy' in state:
        np.random.set_state(state['numpy'])
    if 'torch' in state:
        t = state['torch']
        if not isinstance(t, torch.Tensor):
            t = torch.as_tensor(t, dtype=torch.uint8)
        torch.set_rng_state(t.cpu().to(torch.uint8))


def save_checkpoint(path, ac, pi_optimizer, vf_optimizer, epoch, lr_step,
                    opponent=None, save_rng=True):
    """
    Docstring for save_checkpoint:

    Write a FULL checkpoint so a crashed run resumes exactly: the network weights,
    both optimizer states, the LR schedule step, the epoch, the self-play pool
    (snapshot weights + win rates) when the opponent is a pool, and the RNG state.

    'model' stays a plain state_dict at the top so load_state_dict in opponents.py
    (which unwraps {'model': ...}) keeps working for eval and as a frozen opponent.
    """
    from opponents import SelfPlayPool

    ckpt = {
        'model': ac.state_dict(),
        'pi_optimizer': pi_optimizer.state_dict(),
        'vf_optimizer': vf_optimizer.state_dict(),
        'epoch': epoch,
        'lr_step': lr_step,
        # action shape changed in Phase 2 (angle+power, protocol v3). the tensor
        # shapes did NOT change, so an old save would load silently and play
        # garbage; this tag lets the loaders refuse it with a clear message.
        'action_format': ACTION_FORMAT_VERSION,
    }
    if isinstance(opponent, SelfPlayPool):
        ckpt['pool'] = opponent.state()
    if save_rng:
        ckpt['rng'] = rng_state()
    tmp = f'{path}.tmp'
    torch.save(ckpt, tmp)
    os.replace(tmp, path)
    return path


def load_checkpoint(path, map_location='cpu'):
    """Docstring for load_checkpoint: read a full checkpoint dict and refuse an
    old pre-Phase-2 action format (the tensor shapes match, so without the tag an
    old save would load silently and play garbage)."""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(ckpt, dict) or 'model' not in ckpt:
        raise ValueError(
            f"'{path}' is a pre-Phase-2 checkpoint (no action_format tag). The "
            f"action shape changed to angle+power (protocol v{ACTION_FORMAT_VERSION}); "
            f"old checkpoints no longer apply, train a fresh run."
        )
    found = ckpt.get('action_format')
    if found != ACTION_FORMAT_VERSION:
        raise ValueError(
            f"'{path}' is action_format {found!r}, expected {ACTION_FORMAT_VERSION} "
            f"(angle+power). Old checkpoints no longer apply, train a fresh run."
        )
    return ckpt

def join_and_normalize(buffers, device='cpu', extra_data=None):
    """
    Docstring for join_and_normalize:

    Pull each game's buffer (one per game, GAE already computed per path so the
    paths never mix), concatenate them into one batch, and normalize the
    advantages over the WHOLE batch rather than per game. Returns the same dict
    shape update() expects.

    extra_data: optional list of already-extracted data dicts (e.g. the opponent
    side of live-vs-live games from Opponent_buffer.get()) to fold into the same
    batch before the single whole-batch advantage normalization.
    """
    parts = [buf.get(device=device, normalize_adv=False) for buf in buffers]
    if extra_data:
        parts += [d for d in extra_data if d['obs'].shape[0] > 0]
    data = {k: torch.cat([p[k] for p in parts], dim=0) for k in parts[0]}
    adv = data['adv']
    data['adv'] = (adv - adv.mean()) / (adv.std() + 1e-8)
    return data


class _Single_env_as_vector:
    """
    Docstring for _Single_env_as_vector class:

    Wraps one Bobble_league_env so the N-games training loop can drive it with the
    same (N, ...) shapes when games == 1. Keeps the original single-game path
    working without a second copy of the epoch loop.
    """

    def __init__(self, env):
        self.env = env
        self.n_games = 1
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    # the shaping-fade loop reads and rewrites reward_config each epoch; forward
    # it to the wrapped env so the single-game path (games == 1) behaves like the
    # vector path, which carries reward_config itself.
    @property
    def reward_config(self):
        return self.env.reward_config

    @reward_config.setter
    def reward_config(self, value):
        self.env.reward_config = value

    def reset(self):
        o, info = self.env.reset()
        return o[None, :], [info]

    def step(self, actions):
        o, r, term, trunc, info = self.env.step(np.asarray(actions).reshape(6))
        info = dict(info)
        info['restarted'] = False
        if term or trunc:
            o, _ = self.env.reset()
        return (o[None, :], np.array([r], dtype=np.float32),
                np.array([term], dtype=bool), np.array([trunc], dtype=bool), [info])

    def close(self):
        self.env.close()


def ppo_train(env_fn=None, actor_critic=Actor_critic_neural_network, ac_kwargs=dict(),
              steps_per_epoch=4000, epochs=50, gamma=0.99, clip_ratio=0.2,
              pi_lr=3e-4, vf_lr=1e-3, train_pi_iters=80, train_v_iters=80,
              lam=0.97, max_ep_len=1000, target_kl=0.02, save_freq=10, device=None,
              log_dir='runs/ppo_experiment', opponent='self', snapshot_freq=5,
              init_checkpoint=None, reward_config=None, step_timeout=120.0,
              games=1, launcher=None, vec_env=None, resume=None,
              minibatch_size=256, n_passes=10, ent_coef=0.005, max_grad_norm=0.5,
              pfsp=False, shaping_fade=False, shaping_fade_fraction=0.5):
    """
    Docstring for ppo_train:

    Proximal policy optimization (PPO-Clip) training loop (copied argument definitions from OpenAI).
    Runs one game (games=1, the original path) or N games at once over a Vector_env.

    Args:
        env_fn : makes ONE single-game env, called as
            env_fn(opponent=..., reward_config=..., max_ep_len=..., step_timeout=...).
            Used only when games == 1 and no vec_env is given.
        actor_critic: constructor for the actor-critic module.
        ac_kwargs (dict): kwargs for the actor_critic.
        steps_per_epoch (int): TOTAL steps across all games per epoch (e.g. 6000 with 6 games = 1000 each).
        epochs (int): number of policy updates.
        gamma, lam, clip_ratio, pi_lr, vf_lr, train_pi_iters, train_v_iters, target_kl:
            standard PPO hyperparameters (see the original Spinning Up docs).
        max_ep_len (int): turns before an episode is truncated.
        save_freq (int): epochs between full checkpoints.
        device: 'cuda' / 'cpu' / 'mps' / None for auto.
        log_dir: TensorBoard directory.
        opponent: 'self' | 'chaser' | 'random' | 'still' | 'none' | checkpoint path.
            With N games each game gets its OWN opponent, and all 'self' opponents
            share one snapshot pool so a snapshot added once is seen by all.
        snapshot_freq (int): with opponent='self', add the learner to the pool every N epochs.
        init_checkpoint: optional weights to start from (fresh run).
        reward_config: RewardConfig for shaping.
        step_timeout: seconds to wait per game per turn.
        games (int): number of parallel games.
        launcher: a Game_launcher for the Vector_env (real games); None for tests that inject vec_env.
        vec_env: a ready Vector_env (tests inject one with mock bridges); overrides games/launcher.
        resume: path to a FULL checkpoint to pick up from (restores weights, both
            optimizers, LR step, epoch, pool and RNG).
    """
    from opponents import make_opponent, SelfPlayPool

    device = get_device(device)
    print(f'Using device: {device}')

    # ---- build the opponents: one per game. self-play games each get their own
    # pool (own 'current' opponent) that SHARES one snapshot list + the live net,
    # so every game picks its own opponent while a snapshot added once is seen by all ----
    if vec_env is not None:
        n_games = vec_env.n_games
        opponents = vec_env.opponents
        pools = [o for o in opponents if isinstance(o, SelfPlayPool)]
        shared_pool = pools[0] if pools else None
    else:
        n_games = games
        if opponent == 'self':
            shared_pool = SelfPlayPool(pfsp=pfsp)
            opponents = shared_pool.make_siblings(n_games)
        else:
            shared_pool = None
            opponents = [make_opponent(opponent) for _ in range(n_games)]

    # ---- build the env ----
    if vec_env is not None:
        env = vec_env
    elif n_games > 1 or launcher is not None:
        from bobble_league_env import Vector_env
        env = Vector_env(n_games, launcher=launcher, opponents=opponents,
                         reward_config=reward_config, max_ep_len=max_ep_len,
                         step_timeout=step_timeout, auto_launch=launcher is not None)
    else:
        single = env_fn(opponent=opponents[0], reward_config=reward_config,
                        max_ep_len=max_ep_len, step_timeout=step_timeout)
        env = _Single_env_as_vector(single)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    writer = SummaryWriter(log_dir=log_dir)

    # ---- model + optimizers ----
    ac = actor_critic(obs_dim=obs_dim, act_dim=act_dim, **ac_kwargs)
    ac.to(device)
    pi_optimizer = Adam(list(ac.pi_net.parameters()) + [ac.log_std], lr=pi_lr)
    vf_optimizer = Adam(ac.v_net.parameters(), lr=vf_lr)

    # self-play: hand the live net to the pool so its 'live' slot can play the
    # opponent side (and its turns become training data in live-vs-live games)
    if shared_pool is not None:
        shared_pool.attach_live(ac)

    start_epoch = 0
    resumed_lr_step = 0
    if resume:
        ckpt = load_checkpoint(resume, map_location=device)
        ac.load_state_dict(ckpt['model'])
        if 'pi_optimizer' in ckpt:
            pi_optimizer.load_state_dict(ckpt['pi_optimizer'])
        if 'vf_optimizer' in ckpt:
            vf_optimizer.load_state_dict(ckpt['vf_optimizer'])
        if shared_pool is not None and ckpt.get('pool'):
            shared_pool.load_state(ckpt['pool'])
        set_rng_state(ckpt.get('rng'))
        start_epoch = int(ckpt.get('epoch', -1)) + 1
        resumed_lr_step = int(ckpt.get('lr_step', start_epoch))
        print(f'Resumed from {resume} at epoch {start_epoch} (lr_step {resumed_lr_step})')
    elif init_checkpoint:
        from opponents import load_state_dict
        ac.load_state_dict(load_state_dict(init_checkpoint))
        print(f'Loaded weights from {init_checkpoint}')
        if shared_pool is not None:
            shared_pool.add(ac.state_dict())

    # ---- linear LR decay to 0 over the run (one update per epoch) ----
    # step is restored from the checkpoint so a resumed run keeps decaying from
    # where it left off instead of snapping back to the full LR.
    lr_schedule = Linear_lr_schedule([pi_optimizer, vf_optimizer], [pi_lr, vf_lr],
                                     total=epochs, step=resumed_lr_step)

    # ---- per-game buffers: steps split evenly, GAE paths stay separate ----
    per_game = max(1, steps_per_epoch // n_games)
    total_steps = per_game * n_games
    buffers = [PPO_buffer(obs_dim, act_dim, per_game, gamma, lam) for _ in range(n_games)]

    o, infos = env.reset()
    ep_ret = np.zeros(n_games, dtype=np.float64)
    ep_len = np.zeros(n_games, dtype=np.int64)
    print(f"Games: {n_games}, steps/epoch total: {total_steps} ({per_game} each), "
          f"mode: {infos[0]['play_mode']}, opponent: {getattr(opponents[0], 'name', 'none')}")

    opp_name = getattr(opponents[0], 'name', 'none')

    # ---- shaping fade (Phase 6, R39): the env's base shaping config; each epoch
    # it is scaled toward 0 over the first `shaping_fade_fraction` of the run so
    # the AI ends up playing for goals, not "ball moved forward". ----
    from bobble_league_env import opponent_view, shaping_fade_fraction as _fade_frac
    base_reward_config = env.reward_config

    # ---- main loop ----

    for epoch in range(start_epoch, epochs):
        # fade the per-turn shaping for this epoch (goal reward is kept)
        if shaping_fade:
            frac = _fade_frac(epoch, epochs, shaping_fade_fraction)
            env.reward_config = base_reward_config.scaled(frac)
        epoch_returns, epoch_lengths = [], []
        results = {'win': 0, 'loss': 0, 'timeout': 0}
        restarts = 0
        turn_seconds_all = []          # every turn's settle time (game seconds)
        opp_turns_stored = 0           # how many Team B turns went into training this epoch
        # opponent-side (Team B) buffers: only filled on live-vs-live turns
        opp_buffers = [Opponent_buffer(gamma, lam) for _ in range(n_games)]

        for t in range(per_game):
            a, v, logp = ac.step(o)                 # one batched call over all games
            a_np = a.cpu().numpy().reshape(n_games, act_dim)
            v_np = v.cpu().numpy().reshape(n_games)
            logp_np = logp.cpu().numpy().reshape(n_games)

            next_o, r, terminated, truncated, step_infos = env.step(a_np)

            for g in range(n_games):
                buffers[g].store(o[g], a_np[g], r[g], float(v_np[g]), float(logp_np[g]))
                ep_ret[g] += r[g]
                ep_len[g] += 1
                # turn length (game seconds); real Godot sends simTime, mocks a small spread
                ts = step_infos[g].get('turn_seconds')
                if ts is not None and not step_infos[g].get('restarted'):
                    turn_seconds_all.append(float(ts))
                # live vs live: store Team B's turn (already in the opponent frame)
                opp = step_infos[g].get('opponent')
                if opp is not None:
                    opp_buffers[g].store(opp['obs'], opp['act'], opp['reward'],
                                         opp['val'], opp['logp'])
                    opp_turns_stored += 1

            o = next_o
            epoch_ended = t == per_game - 1

            for g in range(n_games):
                if step_infos[g].get('restarted'):
                    restarts += 1
                term = bool(terminated[g])
                trunc = bool(truncated[g]) or ep_len[g] >= max_ep_len
                if term or trunc or epoch_ended:
                    if term:
                        last_val = 0.0
                        opp_last_val = 0.0
                    else:
                        # truncated / cut by epoch / restarted -> bootstrap from the critic
                        _, v_last, _ = ac.step(o[g])
                        last_val = v_last.item()
                        # Team B's cut path bootstraps from the live net's value of
                        # the opponent-frame next obs (the same net, mirrored view)
                        _, opp_v_last, _ = ac.step(opponent_view(o[g]))
                        opp_last_val = opp_v_last.item()
                    buffers[g].finish_path(last_val)
                    opp_buffers[g].finish_path(opp_last_val)
                    if term or trunc:
                        epoch_returns.append(float(ep_ret[g]))
                        epoch_lengths.append(int(ep_len[g]))
                        scorer = step_infos[g].get('scorer', 0) if term else 0
                        outcome = 'win' if scorer == 1 else 'loss' if scorer == 2 else 'timeout'
                        results[outcome] += 1
                        ep_ret[g] = 0.0
                        ep_len[g] = 0

        # join per-game buffers + the opponent (Team B) side of live-vs-live games,
        # normalize advantages over the whole batch, update
        opp_data = [buf.get(device=device) for buf in opp_buffers if len(buf) > 0]
        data = join_and_normalize(buffers, device=device, extra_data=opp_data)
        update_info = update_from_data(ac, data, pi_optimizer, vf_optimizer,
                                       train_pi_iters, train_v_iters, target_kl, clip_ratio,
                                       minibatch_size=minibatch_size, n_passes=n_passes,
                                       ent_coef=ent_coef, max_grad_norm=max_grad_norm)
        # LR before stepping the schedule is the LR this update actually ran at
        cur_lr = lr_schedule.current_lrs()[0]

        # TensorBoard
        if epoch_returns:
            n_eps = len(epoch_returns)
            writer.add_scalar('Performance/Mean_Reward', np.mean(epoch_returns), epoch)
            writer.add_scalar('Performance/Mean_Episode_Length', np.mean(epoch_lengths), epoch)
            writer.add_scalar('Performance/Win_Rate', results['win'] / n_eps, epoch)
            writer.add_scalar('Performance/Loss_Rate', results['loss'] / n_eps, epoch)
        writer.add_scalar('Loss/Policy_Loss', update_info['LossPi'], epoch)
        writer.add_scalar('Loss/Value_Loss', update_info['LossV'], epoch)
        writer.add_scalar('Training/KL_Divergence', update_info['KL'], epoch)
        writer.add_scalar('Training/Entropy', update_info['Entropy'], epoch)
        writer.add_scalar('Training/Clip_Fraction', update_info['ClipFrac'], epoch)
        writer.add_scalar('Training/Explained_Variance', update_info['ExplainedVariance'], epoch)
        writer.add_scalar('Training/Passes_Run', update_info['PassesRun'], epoch)
        writer.add_scalar('Training/Learning_Rate', cur_lr, epoch)
        writer.add_scalar('Training/Restarts', restarts, epoch)
        writer.add_scalar('Data/Opponent_Turns_Stored', opp_turns_stored, epoch)
        writer.add_scalar('Data/Batch_Size', int(data['obs'].shape[0]), epoch)

        # turn-length histogram + summary stats in game seconds (R33): tells us
        # whether MaxTurnSeconds (20 s) is far above where turns actually settle.
        ts_arr = np.asarray(turn_seconds_all, dtype=np.float32)
        if ts_arr.size:
            writer.add_histogram('Turn_Length/Seconds', ts_arr, epoch)
            writer.add_scalar('Turn_Length/Mean_Seconds', float(ts_arr.mean()), epoch)
            writer.add_scalar('Turn_Length/Median_Seconds', float(np.median(ts_arr)), epoch)
            writer.add_scalar('Turn_Length/P95_Seconds', float(np.percentile(ts_arr, 95)), epoch)
            writer.add_scalar('Turn_Length/Max_Seconds', float(ts_arr.max()), epoch)

        print(f'\nEpoch: {epoch}')
        print(f"Episodes: {len(epoch_returns)} (W {results['win']} / L {results['loss']} / T {results['timeout']}), "
              f"restarts {restarts}")
        print(f"LossPi: {update_info['LossPi']:.4f}  LossV: {update_info['LossV']:.4f}  "
              f"KL: {update_info['KL']:.4f}  Entropy: {update_info['Entropy']:.4f}  "
              f"ClipFrac: {update_info['ClipFrac']:.4f}  EV: {update_info['ExplainedVariance']:.3f}  "
              f"passes: {update_info['PassesRun']}  lr: {cur_lr:.2e}")
        if ts_arr.size:
            print(f"Turn length (s): mean {ts_arr.mean():.2f}  median {np.median(ts_arr):.2f}  "
                  f"p95 {np.percentile(ts_arr, 95):.2f}  max {ts_arr.max():.2f}")
        print(f"Team B turns stored (live-vs-live): {opp_turns_stored}  "
              f"batch {int(data['obs'].shape[0])}\n")

        # step the LR ramp down one notch for the next epoch
        lr_schedule.advance()

        # self-play: let the shared pool catch up with the learner
        if shared_pool is not None and (epoch + 1) % snapshot_freq == 0:
            shared_pool.add(ac.state_dict())
            print(f'Self-play pool: {len(shared_pool.snapshots)} snapshot(s)')

        # full checkpoint
        if (epoch % save_freq == 0) or (epoch == epochs - 1):
            os.makedirs('checkpoints', exist_ok=True)
            save_checkpoint(f'checkpoints/ppo_epoch_{epoch}.pt', ac, pi_optimizer,
                            vf_optimizer, epoch, lr_step=lr_schedule.step_count,
                            opponent=shared_pool)
            print(f'Model saved at epoch {epoch}')

    writer.close()
    env.close()
    return ac


def update_from_data(ac, data, pi_optimizer, vf_optimizer, train_pi_iters=80,
                     train_v_iters=80, target_kl=0.02, clip_ratio=0.2,
                     minibatch_size=256, n_passes=10, ent_coef=0.005,
                     max_grad_norm=0.5):
    """
    Docstring for update_from_data:

    PPO update with minibatch SGD (Phase 3). Takes a prepared data dict (already
    joined across games and advantage-normalized).

    Each "pass" is one sweep over the whole batch in shuffled minibatches of
    'minibatch_size'. We run up to 'n_passes' passes (replacing the old 80
    full-batch steps). On each minibatch:
      - policy loss = clipped surrogate - ent_coef * entropy (entropy bonus),
      - value loss on the same minibatch,
      - gradients clipped to max_grad_norm (0.5) for both nets before stepping.
    KL early stop is checked per pass (on the whole batch), with target_kl 0.02.
    train_pi_iters / train_v_iters are kept for signature compatibility but the
    minibatch schedule (n_passes) drives the loop now.

    Returns diagnostics: losses, approx_kl, clip fraction, entropy, explained
    variance and how many passes actually ran.
    """
    n = data['obs'].shape[0]
    mb = min(minibatch_size, n)

    # old losses + the pre-update KL (0 by definition) for logging
    with torch.no_grad():
        pi_l_old, pi_info_old = compute_loss_pi(ac, data, clip_ratio)
        v_l_old = compute_loss_v(ac, data).item()

    last_kl = pi_info_old['kl']
    last_cf = pi_info_old['cf']
    last_ent = pi_info_old['ent']
    passes_run = 0

    for p in range(n_passes):
        # KL early stop, per pass, on the whole batch
        with torch.no_grad():
            _, full_info = compute_loss_pi(ac, data, clip_ratio)
        last_kl, last_cf, last_ent = full_info['kl'], full_info['cf'], full_info['ent']
        if last_kl > 1.5 * target_kl:
            print(f'Early stopping at pass {p} due to reaching max kl ({last_kl:.4f}).')
            break

        # shuffle and sweep in minibatches
        idx = torch.randperm(n, device=data['obs'].device)
        for start in range(0, n, mb):
            mb_idx = idx[start:start + mb]
            mb_data = {k: v[mb_idx] for k, v in data.items()}

            # policy step with entropy bonus
            pi_optimizer.zero_grad()
            loss_pi, pi_info = compute_loss_pi(ac, mb_data, clip_ratio)
            loss_pi = loss_pi - ent_coef * pi_info['ent_tensor']
            loss_pi.backward()
            torch.nn.utils.clip_grad_norm_(
                list(ac.pi_net.parameters()) + [ac.log_std], max_grad_norm)
            pi_optimizer.step()

            # value step
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(ac, mb_data)
            loss_v.backward()
            torch.nn.utils.clip_grad_norm_(ac.v_net.parameters(), max_grad_norm)
            vf_optimizer.step()

        passes_run += 1

    # explained variance of the critic over the whole batch after the update
    with torch.no_grad():
        _, values_now = ac(data['obs'])
    ev = explained_variance(values_now, data['ret'])

    return dict(
        LossPi=pi_l_old.item(),
        LossV=v_l_old,
        KL=last_kl,
        Entropy=last_ent,
        ClipFrac=last_cf,
        ExplainedVariance=ev,
        PassesRun=passes_run,
    )


if __name__ == '__main__':
    import argparse
    from bobble_league_env import bobble_league_env_helper

    parser = argparse.ArgumentParser(description='Train PPO on Bobble League (start Godot first, or use --games to launch them).')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--steps-per-epoch', type=int, default=4000,
                        help='TOTAL steps across all games per epoch')
    parser.add_argument('--max-ep-len', type=int, default=1000)
    parser.add_argument('--opponent', default='self',
                        help="self | chaser | random | still | none | path/to/checkpoint.pt")
    parser.add_argument('--snapshot-freq', type=int, default=5)
    parser.add_argument('--init-checkpoint', default=None)
    parser.add_argument('--device', default=None)
    parser.add_argument('--log-dir', default='runs/ppo_experiment')
    parser.add_argument('--save-freq', type=int, default=10)
    # ---- phase 1: parallel games + resume ----
    parser.add_argument('--games', type=int, default=1, help='number of parallel games')
    parser.add_argument('--resume', default=None, help='full checkpoint to resume from')
    parser.add_argument('--launch', action='store_true',
                        help='launch the Godot games here (needs --games > 1 or --launch)')
    parser.add_argument('--godot', default=None, help='Godot console exe (with --launch)')
    parser.add_argument('--project', default=None, help='bobble_league project path (with --launch)')
    parser.add_argument('--shm-prefix', default=None, help='folder for per-game shm files (with --launch)')
    parser.add_argument('--timescale', type=int, default=48, help='AI_VS_AI time scale (with --launch)')
    # ---- phase 3: PPO fixes (minibatch, KL, entropy, grad clip) ----
    parser.add_argument('--target-kl', type=float, default=0.02,
                        help='per-pass KL early-stop threshold (0.02-0.03)')
    parser.add_argument('--ent-coef', type=float, default=0.005, help='entropy bonus coefficient')
    parser.add_argument('--minibatch-size', type=int, default=256, help='SGD minibatch size')
    parser.add_argument('--n-passes', type=int, default=10, help='passes over the batch per update')
    parser.add_argument('--max-grad-norm', type=float, default=0.5, help='gradient clip norm for both nets')
    # ---- phase 6: PFSP + shaping fade ----
    parser.add_argument('--pfsp', action='store_true',
                        help='pick frozen pool copies by (1-winrate)^2 instead of uniform')
    parser.add_argument('--shaping-fade', action='store_true',
                        help='fade the per-turn shaping weights to 0 over the first half of training')
    parser.add_argument('--shaping-fade-fraction', type=float, default=0.5,
                        help='fraction of the run over which shaping fades to 0 (default 0.5)')
    args = parser.parse_args()

    launcher = None
    if args.launch or args.games > 1:
        from launch_games import Game_launcher
        launcher = Game_launcher(args.games, godot_exe=args.godot, project_path=args.project,
                                 shm_prefix=args.shm_prefix, log_dir='logs', timescale=args.timescale)

    ppo_train(env_fn=bobble_league_env_helper,
              epochs=args.epochs,
              steps_per_epoch=args.steps_per_epoch,
              max_ep_len=args.max_ep_len,
              opponent=args.opponent,
              snapshot_freq=args.snapshot_freq,
              init_checkpoint=args.init_checkpoint,
              device=args.device,
              log_dir=args.log_dir,
              save_freq=args.save_freq,
              games=args.games,
              launcher=launcher,
              resume=args.resume,
              target_kl=args.target_kl,
              ent_coef=args.ent_coef,
              minibatch_size=args.minibatch_size,
              n_passes=args.n_passes,
              max_grad_norm=args.max_grad_norm,
              pfsp=args.pfsp,
              shaping_fade=args.shaping_fade,
              shaping_fade_fraction=args.shaping_fade_fraction)
