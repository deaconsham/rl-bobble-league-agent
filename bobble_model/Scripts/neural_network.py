"""
Actor-critic network for PPO, plus the running normalizers for values, rewards
and observations.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal, Independent

# starting log std for the policy's action spread. -1.6 is sigma ~0.2 (~36 deg of
# the +-180 deg aim range): a small starting spread aims the first shots tightly,
# which a large study found speeds up training (Andrychowicz et al. 2020,
# https://arxiv.org/abs/2006.05990). warm_start.py keeps this same value through
# cloning (its INIT_LOG_STD must match), and the training loop resets log_std to
# it after loading an init checkpoint so an old save's wide spread is not reused.
INIT_LOG_STD = -1.6


class Running_mean_std:
    """
    Docstring for Running_mean_std class:

    Running mean and variance of a scalar stream, updated in one batch at a time
    with Welford's parallel (Chan et al.) formula so it stays stable over a whole
    run. The critic is trained on value targets normalized by
    these running stats (37 PPO details, https://iclr-blog-track.github.io/2022/
    03/25/ppo-implementation-details/; MAPPO found value normalization "often
    helps and never hurts", https://bair.berkeley.edu/blog/2021/07/14/mappo/).

    The stats are saved in every checkpoint under their own key, so a resumed run
    normalizes on the same scale. A save written before this phase has no stats
    key; load_state on a fresh normalizer then just leaves it unfit (count 0,
    mean 0, var 1), which is the identity transform, so an old save still loads
    and trains, it only re-learns the scale.
    """

    def __init__(self, epsilon=1e-4):
        # start count at a tiny epsilon (not 0) so the first update's parallel
        # merge has a defined denominator; mean 0 / var 1 is the identity.
        self.mean = 0.0
        self.var = 1.0
        self.count = float(epsilon)

    def update(self, x):
        """
        Docstring for update:

        Fold a batch of scalars in (numpy array or torch tensor).
        """
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        x = np.asarray(x, dtype=np.float64).ravel()
        if x.size == 0:
            return
        batch_mean = float(x.mean())
        batch_var = float(x.var())
        batch_count = float(x.size)

        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta * delta * self.count * batch_count / total
        self.mean = new_mean
        self.var = m2 / total
        self.count = total

    @property
    def std(self):
        """
        Docstring for std:

        Standard deviation with a small floor so normalize never divides by 0.
        """
        return float(np.sqrt(self.var) + 1e-8)

    def normalize(self, x):
        """
        Docstring for normalize:

        (x - mean) / std. Takes/returns a torch tensor or a numpy array.
        """
        if isinstance(x, torch.Tensor):
            return (x - self.mean) / self.std
        return (np.asarray(x) - self.mean) / self.std

    def denormalize(self, x):
        """
        Docstring for denormalize:

        std * x + mean: turn a normalized value back to the raw scale (so
        explained variance and logging stay on the real return scale).
        """
        if isinstance(x, torch.Tensor):
            return x * self.std + self.mean
        return np.asarray(x) * self.std + self.mean

    def state(self):
        """
        Docstring for state:

        Plain dict for the checkpoint.
        """
        return {'mean': self.mean, 'var': self.var, 'count': self.count}

    def load_state(self, state):
        """
        Docstring for load_state:

        Restore from a checkpoint dict. A None / empty state leaves the fresh
        identity stats in place (an old save without value-norm stats).
        """
        if not state:
            return
        self.mean = float(state.get('mean', 0.0))
        self.var = float(state.get('var', 1.0))
        self.count = float(state.get('count', 1e-4))


class RewardScaler:
    """
    Docstring for RewardScaler class:

    Scale rewards by a running std of the DISCOUNTED RETURN
    ("37 PPO details", https://iclr-blog-track.github.io/2022/03/25/
    ppo-implementation-details/). It keeps a running estimate R_t = gamma*R_{t-1}
    + r_t of the per-step discounted return and tracks the std of that stream with
    a Running_mean_std; each reward is then divided by that std (mean NOT
    subtracted, so a positive reward stays positive). This keeps the reward
    magnitude ~unit-scale however the shaping and goal rewards combine, which steadies
    the value targets and the advantages.

    One running return per game is kept (games run in parallel and their streams
    are independent), reset to 0 when that game's episode ends. The std stats are
    saved in the checkpoint so a resumed run scales on the same basis; an old save
    with no stats loads as the identity (std ~1, i.e. no scaling) and re-learns.
    """

    def __init__(self, n_games=1, gamma=0.99, epsilon=1e-8):
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.ret_rms = Running_mean_std()
        self._returns = np.zeros(int(n_games), dtype=np.float64)

    def scale(self, rewards, dones):
        """
        Docstring for scale:

        Update the running discounted-return std from this step's rewards and
        return the SCALED rewards. rewards/dones are length-n_games arrays. A game
        whose episode ended (done True) has its running return reset to 0 AFTER
        this step, like the standard implementation.

        """
        rewards = np.asarray(rewards, dtype=np.float64).reshape(-1)
        dones = np.asarray(dones, dtype=bool).reshape(-1)
        if self._returns.shape[0] != rewards.shape[0]:
            self._returns = np.zeros(rewards.shape[0], dtype=np.float64)
        self._returns = self.gamma * self._returns + rewards
        self.ret_rms.update(self._returns)
        # cold start: with only a handful of samples the running variance is not a
        # reliable scale and can be tiny, which would blow rewards up. Pass rewards
        # through until enough have been seen, then scale by the std.
        if self.ret_rms.count < 100.0:
            scaled = rewards
        else:
            std = np.sqrt(self.ret_rms.var) + self.epsilon
            scaled = rewards / std
        # reset the running return where the episode ended
        self._returns[dones] = 0.0
        return scaled.astype(np.float32)

    def state(self):
        """
        Docstring for state:

        Plain dict for the checkpoint (just the return stats; the per-game
        running returns are transient and not saved).
        """
        return {'ret_rms': self.ret_rms.state(), 'gamma': self.gamma}

    def load_state(self, state):
        """
        Docstring for load_state:

        Restore the return stats. A None / empty state leaves the identity
        (std ~1 = no scaling), so an old save loads and re-learns the scale.
        """
        if not state:
            return
        self.ret_rms.load_state(state.get('ret_rms'))
        self.gamma = float(state.get('gamma', self.gamma))


def ortho_init(layer, gain):
    """
    Docstring for ortho_init

    Orthogonal init on one Linear layer (weights by gain, bias to zero). This is
    the PPO default from "The 37 Implementation Details of PPO": it keeps the
    activations well scaled so the policy does not start off saturated.
    """
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.zeros_(layer.bias)
    return layer


class Input_norm(nn.Module):
    """
    Docstring for Input_norm class:

    Running per-dimension normalization of the 14-value observation. On top of the fixed OBS_SCALE clip the env already applies, the
    network subtracts a running mean and divides by a running std learned from the
    observations it actually sees, so every input dimension reaches the net at
    ~unit scale however the fixed scaling leaves it (37 PPO details,
    https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/).

    The stats live as registered BUFFERS (mean, var, count), so they travel inside
    the network's state_dict automatically: every checkpoint carries them, every
    consumer that loads the weights (eval, opponents, live Elo) gets them for free,
    and a resume restores them. A save written before this phase has no obs-norm
    buffers; load_state_dict(strict=False) then leaves the fresh identity stats
    (mean 0, var 1), which is the identity transform, so an old save still loads
    and the run just re-learns the input scale.

    update() is called ONLY in training (once per batch, from the training loop),
    with Welford's parallel merge per dimension so it stays stable over a run. At
    eval the module is never updated, so the stats are frozen (the plan's "frozen
    at eval time"). forward() always just normalizes with the current buffers.
    """

    def __init__(self, dim=14, epsilon=1e-4, clip=5.0):
        super().__init__()
        self.register_buffer('mean', torch.zeros(dim, dtype=torch.float32))
        self.register_buffer('var', torch.ones(dim, dtype=torch.float32))
        # a tiny starting count (not 0) so the first parallel merge has a defined
        # denominator; stored as a buffer so it survives save/load.
        self.register_buffer('count', torch.tensor(float(epsilon), dtype=torch.float32))
        self.clip = float(clip)

    @torch.no_grad()
    def update(self, x):
        """
        Docstring for update:

        Fold a batch of observations (N, dim) into the running stats with a
        per-dimension Welford parallel merge. Training only.
        """
        x = torch.as_tensor(x, dtype=torch.float32)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x.shape[0] == 0:
            return
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = float(x.shape[0])

        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta * delta * self.count * batch_count / total
        self.mean.copy_(new_mean)
        self.var.copy_(m2 / total)
        self.count.copy_(total)

    def forward(self, x):
        """
        Docstring for forward:

        Normalize: (x - mean) / sqrt(var), clipped. No grad flows to the stats
        (they are buffers), so this is a fixed transform for the optimizer.
        """
        std = torch.sqrt(self.var + 1e-8)
        out = (x - self.mean) / std
        if self.clip is not None:
            out = torch.clamp(out, -self.clip, self.clip)
        return out


def neural_network(sizes, activation = nn.Tanh, hidden_gain=None, out_gain=None,
                   use_layernorm=False):
    """
    Docstring for neural_network

    Helper function to build a neural network
    
    :param sizes: List of integers defining the number of neurons in each layer
    :param activation: Activation function used between layers
    :param hidden_gain: orthogonal gain for the hidden Linear layers (None skips init)
    :param out_gain: orthogonal gain for the final Linear layer (None skips init)
    :param use_layernorm: add a LayerNorm after each hidden Linear (before the
        activation). LayerNorm helps a net keep learning while
        its targets shift, which self-play does all the time (Lyle et al. 2024,
        https://arxiv.org/abs/2402.18762). This changes the saved weight shape, so
        old saves and warm_start.pt no longer load; it is only turned on when a run
        needs it (EV still under 0.4 after steps 1-4).
    :return: A neural network
    """
    layers = []
    for i in range(len(sizes) - 1):
        linear = nn.Linear(sizes[i], sizes[i + 1])
        is_last = i == len(sizes) - 2
        # orthogonal init: hidden layers use sqrt(2), the output head its own gain
        if is_last:
            if out_gain is not None:
                ortho_init(linear, out_gain)
        else:
            if hidden_gain is not None:
                ortho_init(linear, hidden_gain)
        layers.append(linear)
        if not is_last:
            # LayerNorm sits between the hidden Linear and its activation
            if use_layernorm:
                layers.append(nn.LayerNorm(sizes[i + 1]))
            layers.append(activation())
    return nn.Sequential(*layers)

class Actor_critic_neural_network(nn.Module):
    """
        Docstring for Actor_critic_neural_network

        Class contains two different neural networks:
        Actor (pi_net): Decides what to do (returns action means)
        Critic (v_net): Estimates how good the situation is (returns value)
        Standard deviation (log_std): Natural log of the standard deviation, initialized to -1.6 (sigma ~0.2) as a parameter to be optimized. A small starting spread aims the first shots tightly; a large study found a low starting spread speeds up training (Andrychowicz et al. 2020).
        
        :param obs_dim: Size of the observation vector
        :param act_dim: Size of the action vector
        :param hidden: List defining hidden layers in neural network
        """
    def __init__(self, obs_dim=14, act_dim=6, hidden=[256, 256], use_layernorm=False,
                 input_norm=False):
        super().__init__()

        # orthogonal init (PPO default): sqrt(2) on the hidden layers, a small 0.01
        # on the policy head so the first actions are near zero (gentle start), and
        # 1.0 on the value head. use_layernorm adds a LayerNorm after each hidden
        # layer and changes the saved weight shape.
        self.use_layernorm = use_layernorm
        # a single shared running input-normalizer applied to the
        # obs before BOTH heads. It is a submodule, so its buffers travel in the
        # state_dict; input_norm=False builds it anyway (as the identity) so the
        # state_dict keys are stable, but the training loop only update()s it when
        # input_norm is on. Kept shared so pi_net and v_net see the same scaling.
        self.input_norm = input_norm
        self.obs_norm = Input_norm(dim=obs_dim)
        root_two = float(np.sqrt(2))
        self.pi_net = neural_network([obs_dim] + hidden + [act_dim],
                                     hidden_gain=root_two, out_gain=0.01,
                                     use_layernorm=use_layernorm)
        self.v_net = neural_network([obs_dim] + hidden + [1],
                                    hidden_gain=root_two, out_gain=1.0,
                                    use_layernorm=use_layernorm)
        self.log_std = nn.Parameter(INIT_LOG_STD * torch.ones(act_dim))

    def norm_obs(self, obs):
        """
        Docstring for norm_obs:

        Apply the shared running input-normalizer. A fixed
        transform (the stats are buffers); call it on every obs before pi_net /
        v_net so direct callers (kl_to_teacher, PolicyOpponent) match forward().
        """
        return self.obs_norm(obs)

    def forward(self, obs):
        """
        Docstring for forward

        Forward propagation through both neural networks
        
        :param obs: Observation tensor
        :return dist: Normal distribution over actions
        :return value: Value estimate
        """

        obs = self.obs_norm(obs)
        mu = self.pi_net(obs)
        std = self.log_std.exp()
        dist = Normal(mu, std)
        dist = Independent(dist, 1)
        value = self.v_net(obs).squeeze(-1)
        return dist, value
    
    def step(self, obs):
        """
        Docstring for step
        
        Training method: returns action, value, log_prob as torch tensors.
        """
        with torch.no_grad():
            device = next(self.parameters()).device
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)

            dist, value = self.forward(obs_tensor)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return action, value, log_prob
    
    def act(self, obs, deterministic=False):
        """
        Docstring for act
        
        Inference method: returns action as torch tensor.
        """
        with torch.no_grad():
            device = next(self.parameters()).device
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
            obs_tensor = self.obs_norm(obs_tensor)

            mu = self.pi_net(obs_tensor)
            
            if deterministic:
                action = mu
            else:
                std = self.log_std.exp()
                dist = Independent(Normal(mu, std), 1)
                action = dist.sample()

            return action


def load_model_state(ac, state_dict):
    """
    Docstring for load_model_state:

    Load a saved model state into an Actor_critic_neural_network, tolerating a
    PRE-Phase-12 save that has no input-normalizer buffers.

    A save written before the obs-norm submodule existed is missing the
    `obs_norm.mean/var/count` keys; loading it strict would crash. We load with
    strict=False and then check that the ONLY difference is those obs-norm
    buffers (any other missing/unexpected key is a real mismatch and is
    re-raised). When the buffers are absent the net keeps its fresh identity
    stats, so an old save loads and the run re-learns the input scale.

    """
    result = ac.load_state_dict(state_dict, strict=False)
    allowed_prefix = 'obs_norm.'
    bad_missing = [k for k in result.missing_keys if not k.startswith(allowed_prefix)]
    bad_unexpected = [k for k in result.unexpected_keys if not k.startswith(allowed_prefix)]
    if bad_missing or bad_unexpected:
        raise RuntimeError(
            f'state_dict does not fit this network. missing={bad_missing} '
            f'unexpected={bad_unexpected}'
        )
    return ac