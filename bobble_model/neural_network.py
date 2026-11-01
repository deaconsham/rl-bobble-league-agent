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

def neural_network(sizes, activation = nn.Tanh, hidden_gain=None, out_gain=None):
    """
    Docstring for neural_network

    Helper function to build a neural network
    
    :param sizes: List of integers defining the number of neurons in each layer
    :param activation: Activation function used between layers
    :param hidden_gain: orthogonal gain for the hidden Linear layers (None skips init)
    :param out_gain: orthogonal gain for the final Linear layer (None skips init)
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
    def __init__(self, obs_dim=14, act_dim=6, hidden=[256, 256]):
        super().__init__()

        # orthogonal init (PPO default): sqrt(2) on the hidden layers, a small 0.01
        # on the policy head so the first actions are near zero (gentle start), and
        # 1.0 on the value head.
        root_two = float(np.sqrt(2))
        self.pi_net = neural_network([obs_dim] + hidden + [act_dim],
                                     hidden_gain=root_two, out_gain=0.01)
        self.v_net = neural_network([obs_dim] + hidden + [1],
                                    hidden_gain=root_two, out_gain=1.0)
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