"""
T1: warm start from the chaser.

- behaviour cloning drives the loss down on a fixed recorded set, and the cloned
  policy mean lands near the recorded actions,
- log_std is reset to its init after cloning, so PPO still explores,
- collect_demos over mock games returns the right shapes with actions in range,
- the saved warm checkpoint loads through the training loop's --init path
  (opponents.load_state_dict), so training_loop.py --init works.

no real Godot here: demos are collected over Mock_godot games, and the cloning
test uses a fixed synthetic set so the loss-drop check is deterministic.
"""
import numpy as np
import pytest
import torch

from bobble_league_env import Vector_env, decode_action
from mock_godot import Mock_launcher
from neural_network import Actor_critic_neural_network
from opponents import ChaserOpponent, load_state_dict as opp_load
from warm_start import (
    INIT_LOG_STD, behaviour_clone, collect_demos, save_warm_checkpoint, warm_start,
)


def make_fixed_demos(n=512, seed=0):
    """
    Docstring for make_fixed_demos:

    a fixed recorded set: random obs, a deterministic action that depends on
    the obs (so a network CAN fit it), and returns that depend on the obs too.
    """
    rng = np.random.default_rng(seed)
    obs = rng.uniform(-1.0, 1.0, size=(n, 14)).astype(np.float32)
    # target action is a smooth function of a few obs components, in [-1, 1]
    act = np.tanh(np.stack([
        obs[:, 0] + obs[:, 2], obs[:, 1],
        obs[:, 4] - obs[:, 6], obs[:, 5],
        obs[:, 8], obs[:, 9] + obs[:, 11],
    ], axis=1)).astype(np.float32)
    ret = (obs[:, 0] * 0.5 + obs[:, 1] * 0.2).astype(np.float32)
    return {'obs': obs, 'act': act, 'ret': ret}


def test_cloning_loss_drops_on_fixed_set():
    """
    Docstring for test_cloning_loss_drops_on_fixed_set:

    the central T1: behaviour-cloning pi and v losses fall over the epochs.
    """
    demos = make_fixed_demos()
    ac = Actor_critic_neural_network(obs_dim=14, act_dim=6)
    history = behaviour_clone(ac, demos, epochs=40, batch_size=128,
                              device='cpu', verbose=False)

    first_pi, first_v = history[0]
    last_pi, last_v = history[-1]
    # both losses drop clearly from the first epoch to the last
    assert last_pi < first_pi * 0.5, f'pi loss did not drop enough: {first_pi} -> {last_pi}'
    assert last_v < first_v * 0.5, f'v loss did not drop enough: {first_v} -> {last_v}'


def test_cloned_mean_is_near_the_recorded_action():
    """
    Docstring for test_cloned_mean_is_near_the_recorded_action:

    after cloning, the policy MEAN sits close to the recorded actions.
    """
    demos = make_fixed_demos()
    ac = Actor_critic_neural_network(obs_dim=14, act_dim=6)
    behaviour_clone(ac, demos, epochs=60, batch_size=128, device='cpu', verbose=False)

    obs = torch.as_tensor(demos['obs'], dtype=torch.float32)
    with torch.no_grad():
        mu = ac.pi_net(obs).numpy()
    mae = float(np.mean(np.abs(mu - demos['act'])))
    assert mae < 0.1, f'cloned mean too far from recorded actions: MAE {mae}'


def test_log_std_is_reset_after_cloning():
    """
    Docstring for test_log_std_is_reset_after_cloning:

    log_std ends at INIT_LOG_STD so PPO starts with real exploration noise.
    """
    demos = make_fixed_demos()
    ac = Actor_critic_neural_network(obs_dim=14, act_dim=6)
    # move log_std away first so the reset is visible
    with torch.no_grad():
        ac.log_std.fill_(-3.0)
    behaviour_clone(ac, demos, epochs=5, device='cpu', reset_log_std=True, verbose=False)
    assert torch.allclose(ac.log_std, torch.full((6,), float(INIT_LOG_STD)), atol=1e-6)


def test_log_std_left_alone_when_not_reset():
    demos = make_fixed_demos()
    ac = Actor_critic_neural_network(obs_dim=14, act_dim=6)
    with torch.no_grad():
        ac.log_std.fill_(-3.0)
    behaviour_clone(ac, demos, epochs=3, device='cpu', reset_log_std=False, verbose=False)
    # cloning touches only pi_net and v_net, so log_std stays where it was put
    assert torch.allclose(ac.log_std, torch.full((6,), -3.0), atol=1e-6)


def test_collect_demos_shapes_and_range(tmp_path):
    """
    Docstring for test_collect_demos_shapes_and_range:

    collect_demos over mock games gives (M,14)/(M,6)/(M,) with actions in [-1,1].
    """
    n_games = 4
    launcher = Mock_launcher(n_games, tmp_path, goal_after=3)
    launcher.launch_all()
    bridges = [launcher.wait_for_handshake(i) for i in range(n_games)]
    opponents = [ChaserOpponent() for _ in range(n_games)]
    vec = Vector_env(n_games, bridges=bridges, launcher=launcher,
                     opponents=opponents, step_timeout=5.0, max_ep_len=30)
    try:
        demos = collect_demos(vec, lambda: ChaserOpponent(), n_turns=80,
                              max_ep_len=30, verbose=False)
    finally:
        vec.close()

    per_game = 80 // n_games
    assert demos['obs'].shape == (per_game * n_games, 14)
    assert demos['act'].shape == (per_game * n_games, 6)
    assert demos['ret'].shape == (per_game * n_games,)
    assert np.all(demos['act'] >= -1.0) and np.all(demos['act'] <= 1.0)
    # the chaser never shoots at full power everywhere; decoded powers stay in [0.2, 1]
    dirs = np.stack([decode_action(a) for a in demos['act']])
    lengths = np.linalg.norm(dirs, axis=2)
    assert np.all(lengths >= 0.2 - 1e-5) and np.all(lengths <= 1.0 + 1e-5)


def test_warm_checkpoint_loads_through_init_path(tmp_path):
    """
    Docstring for test_warm_checkpoint_loads_through_init_path:

    the saved warm checkpoint loads via opponents.load_state_dict, which is the
    exact path training_loop.py --init uses, and the weights come back intact.
    """
    demos = make_fixed_demos()
    ac = Actor_critic_neural_network(obs_dim=14, act_dim=6)
    behaviour_clone(ac, demos, epochs=3, device='cpu', verbose=False)

    path = str(tmp_path / 'warm.pt')
    save_warm_checkpoint(path, ac)

    sd = opp_load(path)                 # the --init loader
    ac2 = Actor_critic_neural_network(obs_dim=14, act_dim=6)
    ac2.load_state_dict(sd)
    for k, v in ac.state_dict().items():
        assert torch.allclose(v, ac2.state_dict()[k]), f'{k} did not round trip'


def test_warm_start_end_to_end_over_mock_games(tmp_path, monkeypatch):
    """
    Docstring for test_warm_start_end_to_end_over_mock_games:

    warm_start() records, clones and saves against mock games, and the result
    loads through --init. ties the whole phase-4 path together Godot-free.
    """
    monkeypatch.chdir(tmp_path)
    n_games = 4
    launcher = Mock_launcher(n_games, tmp_path, goal_after=3)
    launcher.launch_all()
    bridges = [launcher.wait_for_handshake(i) for i in range(n_games)]
    opponents = [ChaserOpponent() for _ in range(n_games)]
    vec = Vector_env(n_games, bridges=bridges, launcher=launcher,
                     opponents=opponents, step_timeout=5.0, max_ep_len=30)
    out = str(tmp_path / 'warm.pt')
    try:
        ac, demos, history = warm_start(vec, out, n_turns=120, clone_epochs=5,
                                        max_ep_len=30, verbose=False)
    finally:
        vec.close()

    assert demos['obs'].shape[0] == (120 // n_games) * n_games
    assert history[-1][0] <= history[0][0]          # pi loss did not rise
    assert torch.allclose(ac.log_std, torch.full((6,), float(INIT_LOG_STD)), atol=1e-6)
    opp_load(out)                                   # loads via --init path
