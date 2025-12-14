"""
Comprehensive unit tests for PPO algorithm implementation.
Tests all components: RolloutBuffer, ActorNetwork, CriticNetwork, and PPO main class.
"""
import pytest
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from unittest.mock import Mock, patch, MagicMock
import sys
import os
import tempfile
import shutil

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from algorithms.ppo import PPO, RolloutBuffer, ActorNetwork, CriticNetwork


def calculate_action_dim(N_t, K, M):
    """Calculate the correct action dimension for RIS environment."""
    return 2 * M + 2 * N_t * K


class TestRolloutBuffer:
    """Test cases for RolloutBuffer class."""

    def test_init(self):
        """Test RolloutBuffer initialization."""
        buffer_size = 1000
        state_dim = 20
        action_dim = 16
        device = torch.device('cpu')
        
        buffer = RolloutBuffer(buffer_size, state_dim, action_dim, device)
        
        assert buffer.buffer_size == buffer_size
        assert buffer.state_dim == state_dim
        assert buffer.action_dim == action_dim
        assert buffer.device == device
        assert buffer.ptr == 0
        assert buffer.states.shape == (buffer_size, state_dim)
        assert buffer.actions.shape == (buffer_size, action_dim)
        assert buffer.raw_actions.shape == (buffer_size, action_dim)

    def test_reset(self):
        """Test buffer reset functionality."""
        buffer_size = 100
        state_dim = 10
        action_dim = 8
        device = torch.device('cpu')
        
        buffer = RolloutBuffer(buffer_size, state_dim, action_dim, device)
        
        # Add some data
        buffer.ptr = 50
        buffer.states[0] = torch.ones(state_dim)
        
        # Reset
        buffer.reset()
        
        assert buffer.ptr == 0
        assert torch.all(buffer.states == 0)
        assert torch.all(buffer.actions == 0)
        assert torch.all(buffer.raw_actions == 0)

    def test_add_single_transition(self):
        """Test adding a single transition."""
        buffer_size = 100
        state_dim = 10
        action_dim = 8
        device = torch.device('cpu')
        
        buffer = RolloutBuffer(buffer_size, state_dim, action_dim, device)
        
        state = torch.randn(state_dim)
        action = torch.randn(action_dim)
        raw_action = torch.randn(action_dim)
        reward = torch.tensor([1.5])
        done = torch.tensor([0.0])
        value = torch.tensor([2.0])
        logprob = torch.tensor([-1.2])
        
        buffer.add(state, action, raw_action, reward, done, value, logprob)
        
        assert buffer.ptr == 1
        assert torch.allclose(buffer.states[0], state)
        assert torch.allclose(buffer.actions[0], action)
        assert torch.allclose(buffer.raw_actions[0], raw_action)
        assert torch.allclose(buffer.rewards[0], reward)
        assert torch.allclose(buffer.dones[0], done)
        assert torch.allclose(buffer.values[0], value)
        assert torch.allclose(buffer.logprobs[0], logprob)

    def test_add_multiple_transitions(self):
        """Test adding multiple transitions."""
        buffer_size = 100
        state_dim = 10
        action_dim = 8
        device = torch.device('cpu')
        
        buffer = RolloutBuffer(buffer_size, state_dim, action_dim, device)
        
        for i in range(5):
            state = torch.randn(state_dim)
            action = torch.randn(action_dim)
            raw_action = torch.randn(action_dim)
            reward = torch.tensor([float(i)])
            done = torch.tensor([0.0])
            value = torch.tensor([float(i + 1)])
            logprob = torch.tensor([-float(i)])
            
            buffer.add(state, action, raw_action, reward, done, value, logprob)
        
        assert buffer.ptr == 5
        assert torch.allclose(buffer.rewards[:5], torch.tensor([[0.], [1.], [2.], [3.], [4.]]))

    def test_buffer_overflow(self):
        """Test buffer overflow handling."""
        buffer_size = 3
        state_dim = 2
        action_dim = 2
        device = torch.device('cpu')
        
        buffer = RolloutBuffer(buffer_size, state_dim, action_dim, device)
        
        # Fill buffer
        for i in range(buffer_size):
            state = torch.tensor([float(i), float(i)])
            action = torch.tensor([float(i), float(i)])
            raw_action = torch.tensor([float(i), float(i)])
            reward = torch.tensor([float(i)])
            done = torch.tensor([0.0])
            value = torch.tensor([float(i)])
            logprob = torch.tensor([-float(i)])
            
            buffer.add(state, action, raw_action, reward, done, value, logprob)
        
        # Add one more to trigger overflow
        with patch('builtins.print') as mock_print:
            buffer.add(torch.tensor([10., 10.]), torch.tensor([10., 10.]), 
                     torch.tensor([10., 10.]), torch.tensor([10.]), 
                     torch.tensor([0.]), torch.tensor([10.]), torch.tensor([-10.]))
            
            # Should reset and add the new transition
            assert buffer.ptr == 1
            mock_print.assert_called()

    def test_compute_advantages(self):
        """Test advantage computation with GAE."""
        buffer_size = 10
        state_dim = 4
        action_dim = 2
        device = torch.device('cpu')
        
        buffer = RolloutBuffer(buffer_size, state_dim, action_dim, device)
        
        # Add some transitions
        for i in range(5):
            state = torch.randn(state_dim)
            action = torch.randn(action_dim)
            raw_action = torch.randn(action_dim)
            reward = torch.tensor([1.0])
            done = torch.tensor([0.0])
            value = torch.tensor([2.0])
            logprob = torch.tensor([-1.0])
            
            buffer.add(state, action, raw_action, reward, done, value, logprob)
        
        # Add terminal transition
        buffer.add(torch.randn(state_dim), torch.randn(action_dim), torch.randn(action_dim),
                  torch.tensor([1.0]), torch.tensor([1.0]), torch.tensor([2.0]), torch.tensor([-1.0]))
        
        last_value = torch.tensor([3.0])
        gamma = 0.99
        gae_lambda = 0.95
        
        buffer.compute_advantages(last_value, gamma, gae_lambda)
        
        assert hasattr(buffer, 'advantages')
        assert hasattr(buffer, 'returns')
        assert buffer.advantages.shape == (6, 1)  # 5 + 1 terminal
        assert buffer.returns.shape == (6, 1)
        assert torch.isfinite(buffer.advantages).all()
        assert torch.isfinite(buffer.returns).all()

    def test_get_minibatches(self):
        """Test minibatch generation."""
        buffer_size = 100
        state_dim = 10
        action_dim = 8
        device = torch.device('cpu')
        
        buffer = RolloutBuffer(buffer_size, state_dim, action_dim, device)
        
        # Add some data
        for i in range(20):
            state = torch.randn(state_dim)
            action = torch.randn(action_dim)
            raw_action = torch.randn(action_dim)
            reward = torch.tensor([float(i)])
            done = torch.tensor([0.0])
            value = torch.tensor([float(i)])
            logprob = torch.tensor([-float(i)])
            
            buffer.add(state, action, raw_action, reward, done, value, logprob)
        
        # Set up advantages and returns
        buffer.advantages = torch.randn(20, 1)
        buffer.returns = torch.randn(20, 1)
        
        minibatch_size = 8
        batches = list(buffer.get_minibatches(minibatch_size))
        
        assert len(batches) == 3  # 20 / 8 = 2.5, so 3 batches
        assert len(batches[0]) == 7  # states, actions, raw_actions, logprobs, advantages, returns, values
        
        # Check first batch
        states, actions, raw_actions, logprobs, advantages, returns, values = batches[0]
        assert states.shape[0] == minibatch_size
        assert actions.shape[0] == minibatch_size
        assert raw_actions.shape[0] == minibatch_size


class TestActorNetwork:
    """Test cases for ActorNetwork class."""

    def test_init(self):
        """Test ActorNetwork initialization."""
        state_dim = 20
        action_dim = 16
        N_t = 4
        K = 2
        P_max = 1.0
        actor_linear_layers = [128, 64]
        
        actor = ActorNetwork(state_dim, action_dim, N_t, K, P_max, actor_linear_layers)
        
        assert actor.N_t == N_t
        assert actor.K == K
        assert actor.P_max == P_max
        assert actor.action_dim == action_dim
        assert len(actor.linear_layers) == len(actor_linear_layers)
        assert actor.mean_output.out_features == action_dim
        assert actor.log_std_param.shape == (action_dim,)

    def test_forward(self):
        """Test forward pass."""
        state_dim = 20
        action_dim = 16
        N_t = 4
        K = 2
        P_max = 1.0
        
        actor = ActorNetwork(state_dim, action_dim, N_t, K, P_max)
        
        # Test single input
        state = torch.randn(state_dim)
        mean, log_std = actor.forward(state)
        
        assert mean.shape == (action_dim,)
        assert log_std.shape == (action_dim,)
        assert torch.isfinite(mean).all()
        assert torch.isfinite(log_std).all()
        
        # Test batch input
        batch_size = 32
        states = torch.randn(batch_size, state_dim)
        mean, log_std = actor.forward(states)
        
        assert mean.shape == (batch_size, action_dim)
        assert log_std.shape == (batch_size, action_dim)

    def test_get_distribution(self):
        """Test distribution creation."""
        state_dim = 20
        action_dim = 16
        N_t = 4
        K = 2
        P_max = 1.0
        
        actor = ActorNetwork(state_dim, action_dim, N_t, K, P_max)
        
        state = torch.randn(state_dim)
        dist = actor.get_distribution(state)
        
        assert isinstance(dist, torch.distributions.Normal)
        assert dist.mean.shape == (action_dim,)
        assert dist.scale.shape == (action_dim,)

    def test_get_action_and_log_prob(self):
        """Test action selection and log probability computation."""
        state_dim = 20
        action_dim = 16
        N_t = 4
        K = 2
        P_max = 1.0
        
        actor = ActorNetwork(state_dim, action_dim, N_t, K, P_max)
        
        # Test with batch input (as expected by the method)
        state = torch.randn(1, state_dim)  # Add batch dimension
        action, log_prob, raw_action = actor.get_action_and_log_prob(state)
        
        assert action.shape == (1, action_dim)
        assert log_prob.shape == (1, 1)
        assert raw_action.shape == (1, action_dim)
        assert torch.isfinite(action).all()
        assert torch.isfinite(log_prob).all()
        assert torch.isfinite(raw_action).all()

    def test_get_action_and_log_prob_deterministic(self):
        """Test deterministic action selection."""
        state_dim = 20
        action_dim = 16
        N_t = 4
        K = 2
        P_max = 1.0
        
        actor = ActorNetwork(state_dim, action_dim, N_t, K, P_max)
        
        # Test with batch input (as expected by the method)
        state = torch.randn(1, state_dim)  # Add batch dimension
        action1, log_prob1, raw_action1 = actor.get_action_and_log_prob(state, deterministic=True)
        action2, log_prob2, raw_action2 = actor.get_action_and_log_prob(state, deterministic=True)
        
        # Deterministic actions should be the same
        assert torch.allclose(action1, action2, atol=1e-6)
        assert torch.allclose(log_prob1, log_prob2, atol=1e-6)
        assert torch.allclose(raw_action1, raw_action2, atol=1e-6)

    def test_evaluate_actions(self):
        """Test action evaluation."""
        state_dim = 20
        action_dim = 16
        N_t = 4
        K = 2
        P_max = 1.0
        
        actor = ActorNetwork(state_dim, action_dim, N_t, K, P_max)
        
        # Test single state-action pair
        state = torch.randn(state_dim)
        raw_action = torch.randn(action_dim)
        log_probs, entropy = actor.evaluate_actions(state, raw_action)
        
        assert log_probs.shape == (1,)
        assert entropy.shape == (1,)
        assert torch.isfinite(log_probs).all()
        assert torch.isfinite(entropy).all()
        
        # Test batch
        batch_size = 32
        states = torch.randn(batch_size, state_dim)
        raw_actions = torch.randn(batch_size, action_dim)
        log_probs, entropy = actor.evaluate_actions(states, raw_actions)
        
        assert log_probs.shape == (batch_size, 1)
        assert entropy.shape == (batch_size, 1)

    def test_actor_W_projection_operator(self):
        """Test W projection operator for power constraints."""
        state_dim = 20
        action_dim = 16
        N_t = 4
        K = 2
        P_max = 1.0
        
        actor = ActorNetwork(state_dim, action_dim, N_t, K, P_max)
        
        # Test with matrices that exceed power constraint
        batch_size = 5
        raw_W = torch.randn(batch_size, K, N_t) + 1j * torch.randn(batch_size, K, N_t)
        raw_W = raw_W * 10  # Scale up to exceed constraint
        
        projected_W = actor.actor_W_projection_operator(raw_W)
        
        # Check power constraint satisfaction
        for i in range(batch_size):
            trace = torch.einsum('ij,ji->', projected_W[i], projected_W[i].conj().T).real
            assert trace <= P_max + 1e-6

    def test_actor_process_raw_actions(self):
        """Test raw action processing for domain constraints."""
        state_dim = 20
        N_t = 4
        K = 2
        M = 4  # Number of RIS elements
        action_dim = calculate_action_dim(N_t, K, M)
        P_max = 1.0
        
        actor = ActorNetwork(state_dim, action_dim, N_t, K, P_max)
        
        # Test single action (with batch dimension)
        raw_actions = torch.randn(1, action_dim)
        processed_actions = actor.actor_process_raw_actions(raw_actions)
        
        assert processed_actions.shape == (1, action_dim)
        assert torch.isfinite(processed_actions).all()
        
        # Test batch processing
        batch_size = 32
        raw_actions = torch.randn(batch_size, action_dim)
        processed_actions = actor.actor_process_raw_actions(raw_actions)
        
        assert processed_actions.shape == (batch_size, action_dim)
        
        # Check Theta constraints (unit modulus)
        theta_actions = processed_actions[:, 2 * N_t * K:]
        theta_real = theta_actions[:, 0::2]
        theta_imag = theta_actions[:, 1::2]
        magnitudes = torch.sqrt(theta_real**2 + theta_imag**2)
        assert torch.allclose(magnitudes, torch.ones_like(magnitudes), atol=1e-6)


class TestCriticNetwork:
    """Test cases for CriticNetwork class."""

    def test_init(self):
        """Test CriticNetwork initialization."""
        state_dim = 20
        critic_linear_layers = [128, 64]
        
        critic = CriticNetwork(state_dim, critic_linear_layers)
        
        assert len(critic.linear_layers) == len(critic_linear_layers)
        assert critic.output.out_features == 1

    def test_forward(self):
        """Test forward pass."""
        state_dim = 20
        critic_linear_layers = [128, 64]
        
        critic = CriticNetwork(state_dim, critic_linear_layers)
        
        # Test single input
        state = torch.randn(state_dim)
        value = critic.forward(state)
        
        assert value.shape == (1,)
        assert torch.isfinite(value).all()
        
        # Test batch input
        batch_size = 32
        states = torch.randn(batch_size, state_dim)
        values = critic.forward(states)
        
        assert values.shape == (batch_size, 1)
        assert torch.isfinite(values).all()


class TestPPO:
    """Test cases for main PPO class."""

    def test_init(self):
        """Test PPO initialization."""
        state_dim = 20
        action_dim = 16
        N_t = 4
        K = 2
        P_max = 1.0
        device = torch.device('cpu')
        
        ppo = PPO(
            state_dim=state_dim,
            action_dim=action_dim,
            N_t=N_t,
            K=K,
            P_max=P_max,
            device=device
        )
        
        assert ppo.state_dim == state_dim
        assert ppo.action_dim == action_dim
        assert ppo.N_t == N_t
        assert ppo.K == K
        assert ppo.P_max == P_max
        assert ppo.device == device
        assert hasattr(ppo, 'actor')
        assert hasattr(ppo, 'critic')
        assert hasattr(ppo, 'rollout')

    def test_init_with_custom_parameters(self):
        """Test PPO initialization with custom parameters."""
        state_dim = 20
        action_dim = 16
        N_t = 4
        K = 2
        P_max = 1.0
        device = torch.device('cpu')
        
        ppo = PPO(
            state_dim=state_dim,
            action_dim=action_dim,
            N_t=N_t,
            K=K,
            P_max=P_max,
            device=device,
            actor_lr=1e-3,
            critic_lr=1e-3,
            gamma=0.95,
            gae_lambda=0.9,
            clip_range=0.1,
            entropy_coef=0.02,
            value_coef=0.5,
            max_grad_norm=1.0,
            rollout_size=1024,
            ppo_epochs=5,
            minibatch_size=32
        )
        
        assert ppo.gamma == 0.95
        assert ppo.gae_lambda == 0.9
        assert ppo.clip_range == 0.1
        assert ppo.entropy_coef == 0.02
        assert ppo.value_coef == 0.5
        assert ppo.max_grad_norm == 1.0
        assert ppo.rollout_size == 1024
        assert ppo.ppo_epochs == 5
        assert ppo.minibatch_size == 32

    def test_select_action(self):
        """Test action selection."""
        state_dim = 20
        action_dim = 16
        N_t = 4
        K = 2
        P_max = 1.0
        device = torch.device('cpu')
        
        ppo = PPO(
            state_dim=state_dim,
            action_dim=action_dim,
            N_t=N_t,
            K=K,
            P_max=P_max,
            device=device
        )
        
        # Test single state
        state = np.random.randn(state_dim)
        action, log_prob, raw_action = ppo.select_action(state)
        
        assert action.shape == (action_dim,)
        assert log_prob.shape == (1,)
        assert raw_action.shape == (action_dim,)
        assert isinstance(action, torch.Tensor)
        assert isinstance(log_prob, torch.Tensor)
        assert isinstance(raw_action, torch.Tensor)
        assert torch.isfinite(action).all()
        assert torch.isfinite(log_prob).all()
        assert torch.isfinite(raw_action).all()

    def test_select_action_eval_mode(self):
        """Test deterministic action selection in eval mode."""
        state_dim = 20
        action_dim = 16
        N_t = 4
        K = 2
        P_max = 1.0
        device = torch.device('cpu')
        
        ppo = PPO(
            state_dim=state_dim,
            action_dim=action_dim,
            N_t=N_t,
            K=K,
            P_max=P_max,
            device=device
        )
        
        state = np.random.randn(state_dim)
        action1, log_prob1, raw_action1 = ppo.select_action(state, eval_mode=True)
        action2, log_prob2, raw_action2 = ppo.select_action(state, eval_mode=True)
        
        # In eval mode, actions should be deterministic
        assert torch.allclose(action1, action2, atol=1e-6)
        assert torch.allclose(log_prob1, log_prob2, atol=1e-6)
        assert torch.allclose(raw_action1, raw_action2, atol=1e-6)

    def test_store_transition(self):
        """Test transition storage."""
        state_dim = 20
        action_dim = 16
        N_t = 4
        K = 2
        P_max = 1.0
        device = torch.device('cpu')
        
        ppo = PPO(
            state_dim=state_dim,
            action_dim=action_dim,
            N_t=N_t,
            K=K,
            P_max=P_max,
            device=device
        )
        
        state = np.random.randn(state_dim)
        action = np.random.randn(action_dim)
        raw_action = np.random.randn(action_dim)
        reward = 1.5
        next_state = np.random.randn(state_dim)
        done = False
        
        ppo.store_transition(state, action, raw_action, reward, next_state, done)
        
        assert ppo.rollout.ptr == 1
        assert torch.allclose(ppo.rollout.states[0], torch.tensor(state, dtype=torch.float32))
        assert torch.allclose(ppo.rollout.actions[0], torch.tensor(action, dtype=torch.float32))
        assert torch.allclose(ppo.rollout.raw_actions[0], torch.tensor(raw_action, dtype=torch.float32))
        assert torch.allclose(ppo.rollout.rewards[0], torch.tensor([reward], dtype=torch.float32))
        assert torch.allclose(ppo.rollout.dones[0], torch.tensor([0.0], dtype=torch.float32))

    def test_training_insufficient_data(self):
        """Test training with insufficient data."""
        state_dim = 20
        action_dim = 16
        N_t = 4
        K = 2
        P_max = 1.0
        device = torch.device('cpu')
        
        ppo = PPO(
            state_dim=state_dim,
            action_dim=action_dim,
            N_t=N_t,
            K=K,
            P_max=P_max,
            device=device,
            minibatch_size=64
        )
        
        # Add insufficient data
        for i in range(10):
            state = np.random.randn(state_dim)
            action = np.random.randn(action_dim)
            raw_action = np.random.randn(action_dim)
            reward = 1.0
            next_state = np.random.randn(state_dim)
            done = False
            
            ppo.store_transition(state, action, raw_action, reward, next_state, done)
        
        actor_loss, critic_loss, mean_reward = ppo.training()
        
        # Should return zeros when insufficient data
        assert actor_loss == 0.0
        assert critic_loss == 0.0
        assert mean_reward == 0.0

    def test_training_sufficient_data(self):
        """Test training with sufficient data."""
        state_dim = 20
        action_dim = 16
        N_t = 4
        K = 2
        P_max = 1.0
        device = torch.device('cpu')
        
        ppo = PPO(
            state_dim=state_dim,
            action_dim=action_dim,
            N_t=N_t,
            K=K,
            P_max=P_max,
            device=device,
            minibatch_size=8,
            ppo_epochs=2
        )
        
        # Add sufficient data
        for i in range(100):
            state = np.random.randn(state_dim)
            action = np.random.randn(action_dim)
            raw_action = np.random.randn(action_dim)
            reward = 1.0
            next_state = np.random.randn(state_dim)
            done = (i % 20 == 19)  # Terminal every 20 steps
            
            ppo.store_transition(state, action, raw_action, reward, next_state, done)
        
        actor_loss, critic_loss, mean_reward = ppo.training()
        
        assert isinstance(actor_loss, float)
        assert isinstance(critic_loss, float)
        assert isinstance(mean_reward, float)
        assert torch.isfinite(torch.tensor(actor_loss))
        assert torch.isfinite(torch.tensor(critic_loss))
        assert torch.isfinite(torch.tensor(mean_reward))

    def test_save_and_load_models(self):
        """Test model saving and loading."""
        state_dim = 20
        action_dim = 16
        N_t = 4
        K = 2
        P_max = 1.0
        device = torch.device('cpu')
        
        ppo = PPO(
            state_dim=state_dim,
            action_dim=action_dim,
            N_t=N_t,
            K=K,
            P_max=P_max,
            device=device
        )
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save models
            ppo.save_models(temp_dir)
            
            # Check files exist
            assert os.path.exists(os.path.join(temp_dir, "actor.pth"))
            assert os.path.exists(os.path.join(temp_dir, "critic.pth"))
            
            # Create new PPO instance and load models
            ppo_new = PPO(
                state_dim=state_dim,
                action_dim=action_dim,
                N_t=N_t,
                K=K,
                P_max=P_max,
                device=device
            )
            
            ppo_new.load_models(temp_dir)
            
            # Test that models produce identical outputs in eval mode (deterministic)
            state = np.random.randn(state_dim)
            action1, log_prob1, raw_action1 = ppo.select_action(state, eval_mode=True)
            action2, log_prob2, raw_action2 = ppo_new.select_action(state, eval_mode=True)
            
            # Should be identical in deterministic mode
            assert torch.allclose(action1, action2, atol=1e-6)
            assert torch.allclose(log_prob1, log_prob2, atol=1e-6)
            assert torch.allclose(raw_action1, raw_action2, atol=1e-6)
            
            # Test that the models can produce different outputs with different states
            state2 = np.random.randn(state_dim)
            action3, _, _ = ppo.select_action(state2, eval_mode=True)
            action4, _, _ = ppo_new.select_action(state2, eval_mode=True)
            
            # Should also be identical for different states
            assert torch.allclose(action3, action4, atol=1e-6)
            
            # And the actions should be different for different states
            assert not torch.allclose(action1, action3, atol=1e-3)

    def test_device_consistency(self):
        """Test that all tensors are on the correct device."""
        state_dim = 20
        action_dim = 16
        N_t = 4
        K = 2
        P_max = 1.0
        device = torch.device('cpu')
        
        ppo = PPO(
            state_dim=state_dim,
            action_dim=action_dim,
            N_t=N_t,
            K=K,
            P_max=P_max,
            device=device
        )
        
        state = np.random.randn(state_dim)
        action, log_prob, raw_action = ppo.select_action(state)
        
        assert action.device == device
        assert log_prob.device == device
        assert raw_action.device == device

    def test_power_constraint_satisfaction(self):
        """Test that actions satisfy power constraints."""
        state_dim = 20
        N_t = 4
        K = 2
        M = 4
        action_dim = calculate_action_dim(N_t, K, M)
        P_max = 1.0
        device = torch.device('cpu')
        
        ppo = PPO(
            state_dim=state_dim,
            action_dim=action_dim,
            N_t=N_t,
            K=K,
            P_max=P_max,
            device=device
        )
        
        # Test multiple actions
        for _ in range(10):
            state = np.random.randn(state_dim)
            action, _, _ = ppo.select_action(state)
            
            # Extract W matrix from actions
            W_raw_actions = action[:2 * N_t * K]
            W_real = W_raw_actions[:N_t * K].view(N_t, K)
            W_imag = W_raw_actions[N_t * K:].view(N_t, K)
            W = W_real + 1j * W_imag
            
            # Check power constraint
            trace = torch.einsum('ij,ji->', W, W.conj().T).real
            assert trace <= P_max + 1e-6

    def test_theta_constraint_satisfaction(self):
        """Test that Theta (phase shift) constraints are satisfied."""
        state_dim = 20
        N_t = 4
        K = 2
        M = 4
        action_dim = calculate_action_dim(N_t, K, M)
        device = torch.device('cpu')
        
        ppo = PPO(
            state_dim=state_dim,
            action_dim=action_dim,
            N_t=N_t,
            K=K,
            P_max=1.0,
            device=device
        )
        
        # Test multiple actions
        for _ in range(10):
            state = np.random.randn(state_dim)
            action, _, _ = ppo.select_action(state)
            
            # Extract Theta components
            theta_actions = action[2 * N_t * K:]
            theta_real = theta_actions[0::2]
            theta_imag = theta_actions[1::2]
            
            # Check unit modulus constraint
            magnitudes = torch.sqrt(theta_real**2 + theta_imag**2)
            assert torch.allclose(magnitudes, torch.ones_like(magnitudes), atol=1e-6)

    @pytest.mark.parametrize("state_dim,N_t,K,M", [
        (10, 2, 1, 2),
        (50, 4, 2, 4),
        (100, 8, 4, 8)
    ])
    def test_different_dimensions(self, state_dim, N_t, K, M):
        """Test PPO with different state and action dimensions."""
        action_dim = calculate_action_dim(N_t, K, M)
        ppo = PPO(
            state_dim=state_dim,
            action_dim=action_dim,
            N_t=N_t,
            K=K,
            P_max=1.0,
            device=torch.device('cpu')
        )
        
        state = np.random.randn(state_dim)
        action, log_prob, raw_action = ppo.select_action(state)
        
        assert action.shape == (action_dim,)
        assert log_prob.shape == (1,)
        assert raw_action.shape == (action_dim,)

    def test_rollout_buffer_integration(self):
        """Test integration with rollout buffer."""
        state_dim = 20
        action_dim = 16
        N_t = 4
        K = 2
        P_max = 1.0
        device = torch.device('cpu')
        
        ppo = PPO(
            state_dim=state_dim,
            action_dim=action_dim,
            N_t=N_t,
            K=K,
            P_max=P_max,
            device=device,
            rollout_size=50
        )
        
        # Add multiple transitions
        for i in range(30):
            state = np.random.randn(state_dim)
            action = np.random.randn(action_dim)
            raw_action = np.random.randn(action_dim)
            reward = float(i)
            next_state = np.random.randn(state_dim)
            done = (i % 10 == 9)
            
            ppo.store_transition(state, action, raw_action, reward, next_state, done)
        
        assert ppo.rollout.ptr == 30
        assert torch.allclose(ppo.rollout.rewards[:30], torch.tensor([[float(i)] for i in range(30)]))

    def test_gae_computation(self):
        """Test GAE computation in training."""
        state_dim = 20
        action_dim = 16
        N_t = 4
        K = 2
        P_max = 1.0
        device = torch.device('cpu')
        
        ppo = PPO(
            state_dim=state_dim,
            action_dim=action_dim,
            N_t=N_t,
            K=K,
            P_max=P_max,
            device=device,
            gae_lambda=0.95
        )
        
        # Add transitions with known rewards
        for i in range(20):
            state = np.random.randn(state_dim)
            action = np.random.randn(action_dim)
            raw_action = np.random.randn(action_dim)
            reward = 1.0 if i < 10 else 0.0  # Different rewards
            next_state = np.random.randn(state_dim)
            done = (i == 19)  # Terminal at the end
            
            ppo.store_transition(state, action, raw_action, reward, next_state, done)
        
        # Test GAE computation
        last_state = np.random.randn(state_dim)
        last_value = ppo.critic(torch.tensor(last_state, dtype=torch.float32, device=device).unsqueeze(0)).squeeze(0)
        
        ppo.rollout.compute_advantages(last_value, ppo.gamma, ppo.gae_lambda)
        
        assert hasattr(ppo.rollout, 'advantages')
        assert hasattr(ppo.rollout, 'returns')
        assert ppo.rollout.advantages.shape == (20, 1)
        assert ppo.rollout.returns.shape == (20, 1)
        assert torch.isfinite(ppo.rollout.advantages).all()
        assert torch.isfinite(ppo.rollout.returns).all()


class TestPPOIntegration:
    """Integration tests for complete PPO workflow."""

    def test_complete_training_cycle(self):
        """Test a complete training cycle."""
        state_dim = 20
        action_dim = 16
        N_t = 4
        K = 2
        P_max = 1.0
        device = torch.device('cpu')
        
        ppo = PPO(
            state_dim=state_dim,
            action_dim=action_dim,
            N_t=N_t,
            K=K,
            P_max=P_max,
            device=device,
            minibatch_size=8,
            ppo_epochs=2
        )
        
        # Simulate a complete episode
        state = np.random.randn(state_dim)
        for step in range(50):
            action, log_prob, raw_action = ppo.select_action(state)
            reward = np.random.randn()  # Random reward
            next_state = np.random.randn(state_dim)
            done = (step == 49)  # Terminal at the end
            
            ppo.store_transition(state, action, raw_action, reward, next_state, done)
            state = next_state
        
        # Perform training
        actor_loss, critic_loss, mean_reward = ppo.training()
        
        assert isinstance(actor_loss, float)
        assert isinstance(critic_loss, float)
        assert isinstance(mean_reward, float)
        assert torch.isfinite(torch.tensor(actor_loss))
        assert torch.isfinite(torch.tensor(critic_loss))
        assert torch.isfinite(torch.tensor(mean_reward))

    def test_multiple_episodes(self):
        """Test multiple episodes with training."""
        state_dim = 20
        action_dim = 16
        N_t = 4
        K = 2
        P_max = 1.0
        device = torch.device('cpu')
        
        ppo = PPO(
            state_dim=state_dim,
            action_dim=action_dim,
            N_t=N_t,
            K=K,
            P_max=P_max,
            device=device,
            minibatch_size=8,
            ppo_epochs=1
        )
        
        # Run multiple episodes
        for episode in range(3):
            state = np.random.randn(state_dim)
            for step in range(20):
                action, log_prob, raw_action = ppo.select_action(state)
                reward = np.random.randn()
                next_state = np.random.randn(state_dim)
                done = (step == 19)
                
                ppo.store_transition(state, action, raw_action, reward, next_state, done)
                state = next_state
                
                if done:
                    break
            
            # Train after each episode
            actor_loss, critic_loss, mean_reward = ppo.training()
            assert torch.isfinite(torch.tensor(actor_loss))
            assert torch.isfinite(torch.tensor(critic_loss))
            assert torch.isfinite(torch.tensor(mean_reward))

    def test_evaluation_mode(self):
        """Test evaluation mode behavior."""
        state_dim = 20
        action_dim = 16
        N_t = 4
        K = 2
        P_max = 1.0
        device = torch.device('cpu')
        
        ppo = PPO(
            state_dim=state_dim,
            action_dim=action_dim,
            N_t=N_t,
            K=K,
            P_max=P_max,
            device=device
        )
        
        state = np.random.randn(state_dim)
        
        # Test multiple evaluations - should be deterministic
        actions = []
        for _ in range(5):
            action, _, _ = ppo.select_action(state, eval_mode=True)
            actions.append(action)
        
        # All actions should be the same in eval mode
        for i in range(1, len(actions)):
            assert torch.allclose(actions[0], actions[i], atol=1e-6)

    def test_gradient_flow(self):
        """Test that gradients flow properly during training."""
        state_dim = 20
        action_dim = 16
        N_t = 4
        K = 2
        P_max = 1.0
        device = torch.device('cpu')
        
        ppo = PPO(
            state_dim=state_dim,
            action_dim=action_dim,
            N_t=N_t,
            K=K,
            P_max=P_max,
            device=device,
            minibatch_size=8,
            ppo_epochs=1
        )
        
        # Store some transitions
        for i in range(20):
            state = np.random.randn(state_dim)
            action = np.random.randn(action_dim)
            raw_action = np.random.randn(action_dim)
            reward = 1.0
            next_state = np.random.randn(state_dim)
            done = (i == 19)
            
            ppo.store_transition(state, action, raw_action, reward, next_state, done)
        
        # Get initial parameters
        initial_actor_params = [p.clone() for p in ppo.actor.parameters()]
        initial_critic_params = [p.clone() for p in ppo.critic.parameters()]
        
        # Train
        actor_loss, critic_loss, mean_reward = ppo.training()
        
        # Check that parameters have changed (gradients flowed)
        actor_changed = any(
            not torch.allclose(initial, current, atol=1e-6)
            for initial, current in zip(initial_actor_params, ppo.actor.parameters())
        )
        critic_changed = any(
            not torch.allclose(initial, current, atol=1e-6)
            for initial, current in zip(initial_critic_params, ppo.critic.parameters())
        )
        
        # At least one network should have updated parameters
        assert actor_changed or critic_changed


if __name__ == "__main__":
    pytest.main([__file__])
