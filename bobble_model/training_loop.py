import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from neural_network import Actor_critic_neural_network

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

    def get(self, device='cpu'):
        """
        Docstring for get:

        Call this at the end of an epoch to get all stored data and normalized advantage.
        Resets pointers for next epoch
        
        Args:
            device: Device to place tensors on ('cpu', 'cuda', or torch.device)
        """
        assert self.ptr == self.max_size  # buffer has to be full
        self.ptr, self.path_start_idx = 0, 0
        
        # Normalize advantages: (A - mean(A)) / std(A)
        # This helps with training stability
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
    ent = pi.entropy().mean().item()  # Policy entropy
    clipped = ratio.gt(1 + clip_ratio) | ratio.lt(1 - clip_ratio)
    clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
    
    pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)
    
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

def update(ac, buf, pi_optimizer, vf_optimizer, train_pi_iters=80, 
           train_v_iters=80, target_kl=0.01, clip_ratio=0.2, device='cpu'):
    """
    Perform PPO update on policy and value function
    
    Args:
        ac: actor-critic model
        buf: PPO_buffer with trajectory data
        pi_optimizer: optimizer for policy network
        vf_optimizer: optimizer for value network
        train_pi_iters: number of gradient steps for policy
        train_v_iters: number of gradient steps for value function
        target_kl: early stopping threshold for KL divergence
        clip_ratio: epsilon for ppo clipping
        device: device to use for computation
    
    Multiple epochs of gradient descent on the same batch of data, but use clipping to prevent too large updates
    """
    data = buf.get(device=device)

    pi_l_old, pi_info_old = compute_loss_pi(ac, data, clip_ratio)
    v_l_old = compute_loss_v(ac, data).item()

    # Train policy with multiple steps of gradient descent
    for i in range(train_pi_iters):
        pi_optimizer.zero_grad()
        loss_pi, pi_info = compute_loss_pi(ac, data, clip_ratio)
        kl = pi_info['kl']
        
        # Early stopping if KL divergence gets too large
        # If the policy changed too much then stop updating
        if kl > 1.5 * target_kl:
            print(f'Early stopping at step {i} due to reaching max kl.')
            break
            
        loss_pi.backward()
        pi_optimizer.step()

    # Train value function with multiple steps of gradient descent  
    for i in range(train_v_iters):
        vf_optimizer.zero_grad()
        loss_v = compute_loss_v(ac, data)
        loss_v.backward()
        vf_optimizer.step()

    # Log changes from update
    kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
    
    return dict(
        LossPi=pi_l_old.item(),
        LossV=v_l_old,
        KL=kl,
        Entropy=ent,
        ClipFrac=cf
    )

def ppo_train(env_fn, actor_critic=Actor_critic_neural_network, ac_kwargs=dict(),
              steps_per_epoch=4000, epochs=50, gamma=0.99, clip_ratio=0.2,
              pi_lr=3e-4, vf_lr=1e-3, train_pi_iters=80, train_v_iters=80,
              lam=0.97, max_ep_len=1000, target_kl=0.01, save_freq=10, device=None,
              log_dir='runs/ppo_experiment'):
    """
    Docstring for ppo_train:
    
    Proximal policy optimization (PPO-Clip) training loop (copied argument definitions from OpenAI)

    Args:
        env_fn : A function which creates a copy of the environment.
        actor_critic: The constructor method for a PyTorch Module with an ``act`` method and ``pi`` module and ``v`` module.
        ac_kwargs (dict): Any kwargs appropriate for the actor_critic.
        steps_per_epoch (int): Number of steps of interaction (state-action pairs) for the agent and the environment in each epoch.
        epochs (int): Number of epochs of interaction (equivalent to number of policy updates) to perform.
        gamma (float): Discount factor. (Always between 0 and 1.)
        clip_ratio (float): Hyperparameter for clipping in the policy objective.
            Roughly: how far can the new policy go from the old policy while 
            still profiting (improving the objective function)? The new policy 
            can still go farther than the clip_ratio says, but it doesn't help
            on the objective anymore. (Usually small, 0.1 to 0.3.)
        pi_lr (float): Learning rate for policy optimizer.
        vf_lr (float): Learning rate for value function optimizer.
        train_pi_iters (int): Maximum number of gradient descent steps to take on policy loss per epoch. (Early stopping may cause optimizer
            to take fewer than this.)
        train_v_iters (int): Number of gradient descent steps to take on value function per epoch.
        lam (float): Lambda for GAE-Lambda. (Always between 0 and 1, close to 1.)
        max_ep_len (int): Maximum length of trajectory / episode / rollout.
        target_kl (float): Roughly what KL divergence we think is appropriate between new and old policies after an update. This will get used 
            for early stopping. (Usually small, 0.01 or 0.05.)
        save_freq (int): How often (in terms of gap between epochs) to save the current policy and value function.
        device: Device to use for training ('cuda', 'cpu', 'mps', or None for auto-detect).
        log_dir: Directory to TensorBoard.
    """
    
    device = get_device(device)
    print(f'Using device: {device}')
    env = env_fn()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    
    # Initialize TensorBoard
    writer = SummaryWriter(log_dir=log_dir)

    # Create actor-critic module and move to device
    ac = actor_critic(obs_dim=obs_dim, act_dim=act_dim, **ac_kwargs)
    ac.to(device)
    
    # Set up experience buffer
    local_steps_per_epoch = steps_per_epoch
    buf = PPO_buffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)
    
    # Set up optimizers for policy and value function
    # NOTE: Adam is for "Adaptive Moment Estimation," extension of SGD method
    pi_optimizer = Adam(ac.pi_net.parameters(), lr=pi_lr)
    vf_optimizer = Adam(ac.v_net.parameters(), lr=vf_lr)

    # Prepare for interaction with environment
    o, ep_ret, ep_len = env.reset(), 0, 0

    # Main loop: collect experience in env and update and log each epoch
    for epoch in range(epochs):

        # Track lengths and returns for this epoch to average them

        epoch_returns = []
        epoch_lengths = []
        for t in range(local_steps_per_epoch):
            # Get action from policy
            # NOTE: ac.step() internally handles device placement
            a, v, logp = ac.step(o)

            # Step the env (convert action to numpy for environment)
            next_o, r, d, _ = env.step(a.cpu().numpy())
            ep_ret += r
            ep_len += 1

            # Save experience to buffer (stored as numpy on CPU)
            buf.store(o, a.cpu().numpy(), r, v.item(), logp.item())

            # Update obs
            o = next_o

            timeout = ep_len == max_ep_len
            terminal = d or timeout
            epoch_ended = t == local_steps_per_epoch - 1

            if terminal or epoch_ended:
                if epoch_ended and not terminal:
                    print('Warning: trajectory cut off by epoch at %d steps.' % ep_len, flush=True)
                # Bootstrap value if not done
                if timeout or epoch_ended:
                    _, v, _ = ac.step(o)
                else:
                    v = 0
                buf.finish_path(v.item())
                if terminal:
                    # Save episode return and length to average them at end of epoch
                    epoch_returns.append(ep_ret)
                    epoch_lengths.append(ep_len)
                    print(f'Episode {epoch}: return={ep_ret:.2f}, length={ep_len}')
                o, ep_ret, ep_len = env.reset(), 0, 0 

        # Perform PPO update
        update_info = update(ac, buf, pi_optimizer, vf_optimizer, 
                           train_pi_iters, train_v_iters, target_kl, clip_ratio, device)
        
        # TensorBoard
        if len(epoch_returns) > 0:
            writer.add_scalar('Performance/Mean_Reward', np.mean(epoch_returns), epoch)
            writer.add_scalar('Performance/Mean_Episode_Length', np.mean(epoch_lengths), epoch)
            
        writer.add_scalar('Loss/Policy_Loss', update_info['LossPi'], epoch)
        writer.add_scalar('Loss/Value_Loss', update_info['LossV'], epoch)
        writer.add_scalar('Training/KL_Divergence', update_info['KL'], epoch)
        writer.add_scalar('Training/Entropy', update_info['Entropy'], epoch)
        writer.add_scalar('Training/Clip_Fraction', update_info['ClipFrac'], epoch)

        # Log info about epoch
        print(f'\nEpoch: {epoch}')
        print(f"LossPi: {update_info['LossPi']:.4f}")
        print(f"LossV: {update_info['LossV']:.4f}")
        print(f"KL: {update_info['KL']:.4f}")
        print(f"Entropy: {update_info['Entropy']:.4f}")
        print(f"ClipFrac: {update_info['ClipFrac']:.4f}\n")
        
        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs - 1):
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(ac.state_dict(), f'checkpoints/ppo_epoch_{epoch}.pt')
            print(f'Model saved at epoch {epoch}')

    writer.close()
    return ac


if __name__ == '__main__':
    from bobble_fuck_env import bobble_fuck_env_helper
    ppo_train(env_fn=bobble_fuck_env_helper)