import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="Phase 3 requires PyTorch; install requirements.txt to run RL tests")

from src.rl import DDPGAgent, DDPGConfig, ReplayBuffer
from src.rl.networks import Actor, Critic


def test_replay_buffer_stores_and_samples_transitions():
    buffer = ReplayBuffer(4, state_dim=2, action_dim=1, seed=2)
    for index in range(4):
        buffer.add(np.array([index, index]), np.array([index]), float(index), np.array([index + 1, index + 1]), False)
    batch = buffer.sample(3)
    assert len(buffer) == 4
    assert batch[0].shape == (3, 2)
    assert batch[1].shape == (3, 1)


def test_actor_and_critic_dimensions_and_bounds():
    actor = Actor(3, 2)
    critic = Critic(3, 2)
    states = torch.randn(5, 3)
    actions = actor(states)
    assert actions.shape == (5, 2)
    assert torch.all(actions <= 1.0) and torch.all(actions >= -1.0)
    assert critic(states, actions).shape == (5, 1)


def test_soft_target_update_moves_target_toward_source():
    agent = DDPGAgent(DDPGConfig(state_dim=2, action_dim=1, actor_hidden_layers=(8,), critic_hidden_layers=(8,), target_tau=0.5))
    before = next(agent.target_actor.parameters()).detach().clone()
    with torch.no_grad():
        next(agent.actor.parameters()).add_(1.0)
    source = next(agent.actor.parameters()).detach().clone()
    agent.soft_update_targets()
    after = next(agent.target_actor.parameters()).detach()
    assert torch.allclose(after, 0.5 * before + 0.5 * source)


def test_agent_action_bounds_and_training_step():
    agent = DDPGAgent(DDPGConfig(state_dim=2, action_dim=1, actor_hidden_layers=(8,), critic_hidden_layers=(8,), batch_size=4, replay_capacity=8))
    action = agent.act(np.array([0.2, -0.3], dtype=np.float32), explore=True)
    assert action.shape == (1,)
    assert -1.0 <= action[0] <= 1.0
    for _ in range(4):
        agent.observe(np.zeros(2), np.zeros(1), 1.0, np.ones(2), True)
    losses = agent.train_step()
    assert losses is not None
    assert np.isfinite(losses["actor_loss"])
    assert np.isfinite(losses["critic_loss"])
