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
    run. Phase 11, step 1: the critic is trained on value targets normalized by
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
        """Fold a batch of scalars in (numpy array or torch tensor)."""
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
        """Standard deviation with a small floor so normalize never divides by 0."""
        return float(np.sqrt(self.var) + 1e-8)

    def normalize(self, x):
        """(x - mean) / std. Takes/returns a torch tensor or a numpy array."""
        if isinstance(x, torch.Tensor):
            return (x - self.mean) / self.std
        return (np.asarray(x) - self.mean) / self.std

    def denormalize(self, x):
        """std * x + mean: turn a normalized value back to the raw scale (so
        explained variance and logging stay on the real return scale)."""
        if isinstance(x, torch.Tensor):
            return x * self.std + self.mean
        return np.asarray(x) * self.std + self.mean

    def state(self):
        """Plain dict for the checkpoint."""
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state(self, state):
        """Restore from a checkpoint dict. A None / empty state leaves the fresh
        identity stats in place (an old save without value-norm stats)."""
        if not state:
            return
        self.mean = float(state.get("mean", 0.0))
        self.var = float(state.get("var", 1.0))
        self.count = float(state.get("count", 1e-4))


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
        activation). Phase 11, step 5: LayerNorm helps a net keep learning while
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
    def __init__(self, obs_dim=14, act_dim=6, hidden=[256, 256], use_layernorm=False):
        super().__init__()

        # orthogonal init (PPO default): sqrt(2) on the hidden layers, a small 0.01
        # on the policy head so the first actions are near zero (gentle start), and
        # 1.0 on the value head. use_layernorm adds a LayerNorm after each hidden
        # layer (Phase 11, step 5) and changes the saved weight shape.
        self.use_layernorm = use_layernorm
        root_two = float(np.sqrt(2))
        self.pi_net = neural_network([obs_dim] + hidden + [act_dim],
                                     hidden_gain=root_two, out_gain=0.01,
                                     use_layernorm=use_layernorm)
        self.v_net = neural_network([obs_dim] + hidden + [1],
                                    hidden_gain=root_two, out_gain=1.0,
                                    use_layernorm=use_layernorm)
        self.log_std = nn.Parameter(INIT_LOG_STD * torch.ones(act_dim))
    
    def forward(self, obs):
        """
        Docstring for forward

        Forward propagation through both neural networks
        
        :param obs: Observation tensor
        :return dist: Normal distribution over actions
        :return value: Value estimate
        """

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

            mu = self.pi_net(obs_tensor)
            
            if deterministic:
                action = mu
            else:
                std = self.log_std.exp()
                dist = Independent(Normal(mu, std), 1)
                action = dist.sample()

            return action