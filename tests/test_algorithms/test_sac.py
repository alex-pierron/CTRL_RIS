"""
Comprehensive unit tests for SAC (Soft Actor-Critic) algorithm implementation.
Tests all components: ActorNetwork, CriticNetwork, and SAC main class.
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

from algorithms.sac import SAC, ActorNetwork, CriticNetwork


def calculate_action_dim(N_t, K, M):
    """Calculate the correct action dimension for RIS environment."""
    return 2 * M + 2 * N_t * K


class TestActorNetwork:
    """Test cases for ActorNetwork class."""

    def test_init_valid_parameters(self):
        """Test ActorNetwork initialization with valid parameters."""
        state_dim = 20
        N_t = 4
        K = 2
        M = 4  # Number of RIS elements
        P_max = 1.0
        action_dim = calculate_action_dim(N_t, K, M)
        actor_linear_layers = [128, 128, 128]
        
        actor = ActorNetwork(state_dim, action_dim, N_t, K, P_max, actor_linear_layers)
        
        assert actor.N_t == N_t
        assert actor.K == K
        assert actor.P_max == P_max
        assert len(actor.linear_layers) == len(actor_linear_layers)
        assert actor.mean_output.out_features == action_dim
        assert actor.log_std_output.out_features == action_dim
        assert actor.log_std_min == -10
        assert actor.log_std_max == 2

    def test_forward_pass(self):
        """Test forward pass of ActorNetwork."""
        state_dim = 20
        N_t = 4
        K = 2
        M = 4
        P_max = 1.0
        action_dim = calculate_action_dim(N_t, K, M)
        actor_linear_layers = [128, 128]
        
        actor = ActorNetwork(state_dim, action_dim, N_t, K, P_max, actor_linear_layers)
        
        # Test forward pass
        batch_size = 32
        state = torch.randn(batch_size, state_dim)
        mean, log_std = actor.forward(state)
        
        assert mean.shape == (batch_size, action_dim)
        assert log_std.shape == (batch_size, action_dim)
        assert torch.all(mean >= -1.0) and torch.all(mean <= 1.0)  # tanh output
        assert torch.all(log_std >= actor.log_std_min) and torch.all(log_std <= actor.log_std_max)

    def test_sample_action(self):
        """Test action sampling with reparameterization trick."""
        state_dim = 20
        N_t = 4
        K = 2
        M = 4
        P_max = 1.0
        action_dim = calculate_action_dim(N_t, K, M)
        actor_linear_layers = [128, 128]
        
        actor = ActorNetwork(state_dim, action_dim, N_t, K, P_max, actor_linear_layers)
        
        batch_size = 16
        state = torch.randn(batch_size, state_dim)
        action, log_prob, raw_action = actor.sample(state)
        
        assert action.shape == (batch_size, action_dim)
        assert log_prob.shape == (batch_size, 1)
        assert raw_action.shape == (batch_size, action_dim)
        assert torch.isfinite(log_prob).all()

    def test_get_action_deterministic(self):
        """Test deterministic action selection."""
        state_dim = 20
        N_t = 4
        K = 2
        M = 4
        P_max = 1.0
        action_dim = calculate_action_dim(N_t, K, M)
        actor_linear_layers = [128, 128]
        
        actor = ActorNetwork(state_dim, action_dim, N_t, K, P_max, actor_linear_layers)
        
        batch_size = 8
        state = torch.randn(batch_size, state_dim)
        action = actor.get_action(state)
        
        assert action.shape == (batch_size, action_dim)

    def test_forward_raw(self):
        """Test forward_raw method."""
        state_dim = 20
        N_t = 4
        K = 2
        M = 4
        P_max = 1.0
        action_dim = calculate_action_dim(N_t, K, M)
        actor_linear_layers = [128, 128]
        
        actor = ActorNetwork(state_dim, action_dim, N_t, K, P_max, actor_linear_layers)
        
        batch_size = 16
        state = torch.randn(batch_size, state_dim)
        raw_action = actor.forward_raw(state)
        
        assert raw_action.shape == (batch_size, action_dim)

    def test_actor_W_projection_operator(self):
        """Test W projection operator for power constraints."""
        state_dim = 20
        N_t = 4
        K = 2
        M = 4
        P_max = 1.0
        action_dim = calculate_action_dim(N_t, K, M)
        actor_linear_layers = [128, 128]
        
        actor = ActorNetwork(state_dim, action_dim, N_t, K, P_max, actor_linear_layers)
        
        batch_size = 8
        # Create test W matrices
        raw_W = torch.randn(batch_size, K, N_t, dtype=torch.complex64)
        
        projected_W = actor.actor_W_projection_operator(raw_W)
        
        assert projected_W.shape == raw_W.shape
        # Check that power constraints are satisfied
        traces = torch.einsum('bij,bji->b', projected_W, projected_W.conj().transpose(1, 2)).real
        assert torch.all(traces <= P_max + 1e-6)  # Allow small numerical errors

    def test_actor_process_raw_actions(self):
        """Test processing of raw actions into constrained actions."""
        state_dim = 20
        N_t = 4
        K = 2
        M = 4
        P_max = 1.0
        action_dim = calculate_action_dim(N_t, K, M)
        actor_linear_layers = [128, 128]
        
        actor = ActorNetwork(state_dim, action_dim, N_t, K, P_max, actor_linear_layers)
        
        batch_size = 8
        raw_actions = torch.randn(batch_size, action_dim)
        processed_actions = actor.actor_process_raw_actions(raw_actions)
        
        assert processed_actions.shape == raw_actions.shape


class TestCriticNetwork:
    """Test cases for CriticNetwork class."""

    def test_init_valid_parameters(self):
        """Test CriticNetwork initialization with valid parameters."""
        state_dim = 20
        action_dim = 12
        critic_linear_layers = [128, 128]
        
        critic = CriticNetwork(state_dim, action_dim, critic_linear_layers)
        
        assert len(critic.linear_layers) == len(critic_linear_layers)
        assert critic.output.out_features == 1

    def test_forward_pass(self):
        """Test forward pass of CriticNetwork."""
        state_dim = 20
        action_dim = 12
        critic_linear_layers = [128, 128]
        
        critic = CriticNetwork(state_dim, action_dim, critic_linear_layers)
        
        batch_size = 32
        state = torch.randn(batch_size, state_dim)
        action = torch.randn(batch_size, action_dim)
        q_value = critic.forward(state, action)
        
        assert q_value.shape == (batch_size, 1)

    def test_forward_with_different_batch_sizes(self):
        """Test forward pass with different batch sizes."""
        state_dim = 20
        action_dim = 12
        critic_linear_layers = [128, 128]
        
        critic = CriticNetwork(state_dim, action_dim, critic_linear_layers)
        
        for batch_size in [1, 16, 64, 128]:
            state = torch.randn(batch_size, state_dim)
            action = torch.randn(batch_size, action_dim)
            q_value = critic.forward(state, action)
            
            assert q_value.shape == (batch_size, 1)


class TestSAC:
    """Test cases for SAC main class."""

    def test_init_with_valid_parameters(self):
        """Test SAC initialization with valid parameters."""
        state_dim = 20
        N_t = 4
        K = 2
        M = 4
        P_max = 1.0
        action_dim = calculate_action_dim(N_t, K, M)
        
        sac = SAC(
            state_dim=state_dim,
            action_dim=action_dim,
            N_t=N_t,
            K=K,
            P_max=P_max,
            actor_linear_layers=[128, 128],
            critic_linear_layers=[128, 128],
            device=torch.device('cpu'),
            buffer_size=10000,
            seed=42
        )
        
        assert sac.state_dim == state_dim
        assert sac.action_dim == action_dim
        assert sac.N_t == N_t
        assert sac.K == K
        assert sac.P_max == P_max
        assert sac.device == torch.device('cpu')
        assert sac.total_it == 0
        assert not sac.use_per  # Default should be False

    def test_init_with_per(self):
        """Test SAC initialization with PER enabled."""
        state_dim = 20
        N_t = 4
        K = 2
        M = 4
        P_max = 1.0
        action_dim = calculate_action_dim(N_t, K, M)
        
        sac = SAC(
            state_dim=state_dim,
            action_dim=action_dim,
            N_t=N_t,
            K=K,
            P_max=P_max,
            use_per=True,
            per_alpha=0.6,
            per_beta_start=0.4,
            per_beta_frames=10000,
            per_epsilon=1e-6
        )
        
        assert sac.use_per
        assert sac.per_epsilon == 1e-6
        # Check that PER buffer was created
        assert hasattr(sac, 'replay_buffer')
        assert sac.replay_buffer is not None

    def test_init_with_automatic_entropy_tuning(self):
        """Test SAC initialization with automatic entropy tuning."""
        state_dim = 20
        N_t = 4
        K = 2
        M = 4
        P_max = 1.0
        action_dim = calculate_action_dim(N_t, K, M)
        
        sac = SAC(
            state_dim=state_dim,
            action_dim=action_dim,
            N_t=N_t,
            K=K,
            P_max=P_max,
            automatic_entropy_tuning=True,
            target_entropy=-action_dim
        )
        
        assert sac.automatic_entropy_tuning
        assert sac.target_entropy == -action_dim
        assert hasattr(sac, 'log_alpha')
        assert hasattr(sac, 'alpha_optimizer')

    def test_init_with_fixed_entropy(self):
        """Test SAC initialization with fixed entropy."""
        state_dim = 20
        N_t = 4
        K = 2
        M = 4
        P_max = 1.0
        action_dim = calculate_action_dim(N_t, K, M)
        alpha = 0.3
        
        sac = SAC(
            state_dim=state_dim,
            action_dim=action_dim,
            N_t=N_t,
            K=K,
            P_max=P_max,
            automatic_entropy_tuning=False,
            alpha=alpha
        )
        
        assert not sac.automatic_entropy_tuning
        assert sac.alpha == alpha

    def test_select_action_eval_mode(self):
        """Test action selection in evaluation mode."""
        state_dim = 20
        N_t = 4
        K = 2
        M = 4
        P_max = 1.0
        action_dim = calculate_action_dim(N_t, K, M)
        
        sac = SAC(
            state_dim=state_dim,
            action_dim=action_dim,
            N_t=N_t,
            K=K,
            P_max=P_max,
            device=torch.device('cpu')
        )
        
        state = np.random.randn(state_dim)
        action = sac.select_action(state, eval_mode=True)
        
        assert action.shape == (action_dim,)
        assert isinstance(action, torch.Tensor)

    def test_select_action_training_mode(self):
        """Test action selection in training mode."""
        state_dim = 20
        N_t = 4
        K = 2
        M = 4
        P_max = 1.0
        action_dim = calculate_action_dim(N_t, K, M)
        
        sac = SAC(
            state_dim=state_dim,
            action_dim=action_dim,
            N_t=N_t,
            K=K,
            P_max=P_max,
            device=torch.device('cpu')
        )
        
        state = np.random.randn(state_dim)
        action = sac.select_action(state, eval_mode=False)
        
        assert action.shape == (action_dim,)
        assert isinstance(action, torch.Tensor)

    def test_select_noised_action(self):
        """Test noised action selection."""
        state_dim = 20
        N_t = 4
        K = 2
        M = 4
        P_max = 1.0
        action_dim = calculate_action_dim(N_t, K, M)
        
        sac = SAC(
            state_dim=state_dim,
            action_dim=action_dim,
            N_t=N_t,
            K=K,
            P_max=P_max,
            device=torch.device('cpu')
        )
        
        state = np.random.randn(state_dim)
        action, noised_action = sac.select_noised_action(state)
        
        assert action.shape == (action_dim,)
        assert noised_action.shape == (action_dim,)
        assert torch.allclose(action, noised_action)  # For SAC, they should be the same

    def test_store_transition(self):
        """Test storing transitions in replay buffer."""
        state_dim = 20
        N_t = 4
        K = 2
        M = 4
        P_max = 1.0
        action_dim = calculate_action_dim(N_t, K, M)
        
        sac = SAC(
            state_dim=state_dim,
            action_dim=action_dim,
            N_t=N_t,
            K=K,
            P_max=P_max,
            buffer_size=1000
        )
        
        # Store some transitions
        for _ in range(10):
            state = np.random.randn(state_dim)
            action = np.random.randn(action_dim)
            reward = np.random.randn()
            next_state = np.random.randn(state_dim)
            
            sac.store_transition(state, action, reward, next_state)
        
        assert sac.replay_buffer.size == 10

    def test_training_step(self):
        """Test training step with sufficient buffer."""
        state_dim = 20
        N_t = 4
        K = 2
        M = 4
        P_max = 1.0
        action_dim = calculate_action_dim(N_t, K, M)
        
        sac = SAC(
            state_dim=state_dim,
            action_dim=action_dim,
            N_t=N_t,
            K=K,
            P_max=P_max,
            buffer_size=1000,
            device=torch.device('cpu')
        )
        
        # Fill buffer with some data
        for _ in range(100):
            state = np.random.randn(state_dim)
            action = np.random.randn(action_dim)
            reward = np.random.randn()
            next_state = np.random.randn(state_dim)
            sac.store_transition(state, action, reward, next_state)
        
        # Perform training step
        batch_size = 32
        losses = sac.training(batch_size)
        
        assert len(losses) == 3  # actor_loss, critic1_loss, rewards
        assert sac.total_it == 1

    def test_training_step_with_per(self):
        """Test training step with PER enabled."""
        state_dim = 20
        N_t = 4
        K = 2
        M = 4
        P_max = 1.0
        action_dim = calculate_action_dim(N_t, K, M)
        
        sac = SAC(
            state_dim=state_dim,
            action_dim=action_dim,
            N_t=N_t,
            K=K,
            P_max=P_max,
            buffer_size=1000,
            use_per=True,
            device=torch.device('cpu')
        )
        
        # Fill buffer with some data
        for _ in range(100):
            state = np.random.randn(state_dim)
            action = np.random.randn(action_dim)
            reward = np.random.randn()
            next_state = np.random.randn(state_dim)
            sac.store_transition(state, action, reward, next_state)
        
        # Perform training step
        batch_size = 32
        losses = sac.training(batch_size)
        
        assert len(losses) == 3
        assert sac.total_it == 1

    def test_update_target_networks(self):
        """Test target network updates."""
        state_dim = 20
        N_t = 4
        K = 2
        M = 4
        P_max = 1.0
        action_dim = calculate_action_dim(N_t, K, M)
        
        sac = SAC(
            state_dim=state_dim,
            action_dim=action_dim,
            N_t=N_t,
            K=K,
            P_max=P_max,
            device=torch.device('cpu')
        )
        
        # First, modify the critic parameters to ensure they're different from targets
        with torch.no_grad():
            for param in sac.critic1.parameters():
                param.data += torch.randn_like(param.data) * 0.1
            for param in sac.critic2.parameters():
                param.data += torch.randn_like(param.data) * 0.1
        
        # Store original target parameters
        original_params1 = [p.clone() for p in sac.target_critic1.parameters()]
        original_params2 = [p.clone() for p in sac.target_critic2.parameters()]
        
        # Update target networks
        sac.update_target_networks(update_target_critics=True)
        
        # Check that parameters have changed (soft update with small tau)
        # With tau=0.005, changes should be small but detectable
        for orig, new in zip(original_params1, sac.target_critic1.parameters()):
            # Check that at least some parameters have changed (allowing for numerical precision)
            assert not torch.allclose(orig, new, atol=1e-8)

        for orig, new in zip(original_params2, sac.target_critic2.parameters()):
            assert not torch.allclose(orig, new, atol=1e-8)

    def test_get_buffer_info(self):
        """Test buffer information retrieval."""
        state_dim = 20
        N_t = 4
        K = 2
        M = 4
        P_max = 1.0
        action_dim = calculate_action_dim(N_t, K, M)
        
        sac = SAC(
            state_dim=state_dim,
            action_dim=action_dim,
            N_t=N_t,
            K=K,
            P_max=P_max,
            buffer_size=1000,
            use_per=True
        )
        
        # Add some data
        for _ in range(50):
            state = np.random.randn(state_dim)
            action = np.random.randn(action_dim)
            reward = np.random.randn()
            next_state = np.random.randn(state_dim)
            sac.store_transition(state, action, reward, next_state)
        
        info = sac.get_buffer_info()
        
        assert 'buffer_type' in info
        assert 'buffer_size' in info
        assert 'buffer_capacity' in info
        assert 'buffer_filled' in info
        assert info['buffer_type'] == 'PER'
        assert info['buffer_size'] == 50

    def test_save_models(self, temp_dir):
        """Test model saving functionality."""
        state_dim = 20
        N_t = 4
        K = 2
        M = 4
        P_max = 1.0
        action_dim = calculate_action_dim(N_t, K, M)
        
        sac = SAC(
            state_dim=state_dim,
            action_dim=action_dim,
            N_t=N_t,
            K=K,
            P_max=P_max,
            device=torch.device('cpu')
        )
        
        # Save models
        sac.save_models(temp_dir)
        
        # Check that files exist
        expected_files = [
            'actor.pth', 'critic1.pth', 'critic2.pth',
            'target_critic1.pth', 'target_critic2.pth',
            'log_alpha.pth', 'config.pth'
        ]
        
        for filename in expected_files:
            filepath = os.path.join(temp_dir, filename)
            assert os.path.exists(filepath)

    def test_load_models(self, temp_dir):
        """Test model loading functionality."""
        state_dim = 20
        N_t = 4
        K = 2
        M = 4
        P_max = 1.0
        action_dim = calculate_action_dim(N_t, K, M)
        
        # Create and save models
        sac1 = SAC(
            state_dim=state_dim,
            action_dim=action_dim,
            N_t=N_t,
            K=K,
            P_max=P_max,
            device=torch.device('cpu')
        )
        sac1.save_models(temp_dir)
        
        # Create new SAC instance and load models
        sac2 = SAC(
            state_dim=state_dim,
            action_dim=action_dim,
            N_t=N_t,
            K=K,
            P_max=P_max,
            device=torch.device('cpu')
        )
        sac2.load_models(temp_dir)
        
        # Check that models are loaded correctly
        assert sac2.actor is not None
        assert sac2.critic1 is not None
        assert sac2.critic2 is not None
        assert sac2.target_critic1 is not None
        assert sac2.target_critic2 is not None

    def test_different_optimizers(self):
        """Test SAC with different optimizers."""
        state_dim = 20
        N_t = 4
        K = 2
        M = 4
        P_max = 1.0
        action_dim = calculate_action_dim(N_t, K, M)
        
        optimizers = ['adam', 'adamw', 'rmsprop', 'sgd']
        
        for optimizer in optimizers:
            sac = SAC(
                state_dim=state_dim,
                action_dim=action_dim,
                N_t=N_t,
                K=K,
                P_max=P_max,
                optimizer=optimizer,
                device=torch.device('cpu')
            )
            
            assert sac.actor_optimizer is not None
            assert sac.critic1_optimizer is not None
            assert sac.critic2_optimizer is not None

    def test_frequency_updates(self):
        """Test different update frequencies."""
        state_dim = 20
        N_t = 4
        K = 2
        M = 4
        P_max = 1.0
        action_dim = calculate_action_dim(N_t, K, M)
        
        sac = SAC(
            state_dim=state_dim,
            action_dim=action_dim,
            N_t=N_t,
            K=K,
            P_max=P_max,
            actor_frequency_update=2,
            critic_frequency_update=3,
            device=torch.device('cpu')
        )
        
        assert sac.actor_frequency_update == 2
        assert sac.critic_frequency_update == 3

    def test_gpu_device_handling(self):
        """Test GPU device handling."""
        state_dim = 20
        N_t = 4
        K = 2
        M = 4
        P_max = 1.0
        action_dim = calculate_action_dim(N_t, K, M)
        
        # Test with CPU device (should work regardless of CUDA availability)
        sac = SAC(
            state_dim=state_dim,
            action_dim=action_dim,
            N_t=N_t,
            K=K,
            P_max=P_max,
            device=torch.device('cpu')
        )
        
        # Check that the device handling works correctly
        assert sac.device == torch.device('cpu')
        assert not sac.gpu_used
        assert sac.scaler is None  # No scaler for CPU
        
        # Test that the SAC was initialized correctly
        assert sac.actor is not None
        assert sac.critic1 is not None
        assert sac.critic2 is not None

    def test_loss_scaling(self):
        """Test loss scaling functionality."""
        state_dim = 20
        N_t = 4
        K = 2
        M = 4
        P_max = 1.0
        action_dim = calculate_action_dim(N_t, K, M)
        
        with patch('torch.cuda.is_available', return_value=True):
            sac = SAC(
                state_dim=state_dim,
                action_dim=action_dim,
                N_t=N_t,
                K=K,
                P_max=P_max,
                using_loss_scaling=True,
                device=torch.device('cuda')
            )
            
            assert sac.using_loss_scaling
            assert sac.scaler is not None

    def test_calculate_td_errors(self):
        """Test TD error calculation for PER."""
        state_dim = 20
        N_t = 4
        K = 2
        M = 4
        P_max = 1.0
        action_dim = calculate_action_dim(N_t, K, M)
        
        sac = SAC(
            state_dim=state_dim,
            action_dim=action_dim,
            N_t=N_t,
            K=K,
            P_max=P_max,
            device=torch.device('cpu')
        )
        
        batch_size = 16
        states = torch.randn(batch_size, state_dim)
        actions = torch.randn(batch_size, action_dim)
        rewards = torch.randn(batch_size, 1)
        next_states = torch.randn(batch_size, state_dim)
        q1_values = torch.randn(batch_size, 1)
        q2_values = torch.randn(batch_size, 1)
        
        td_errors = sac._calculate_td_errors(states, actions, rewards, next_states, q1_values, q2_values)
        
        assert td_errors.shape == (batch_size,)
        assert np.all(td_errors >= 0)  # TD errors should be non-negative

    def test_sample_from_buffer_standard(self):
        """Test sampling from standard buffer."""
        state_dim = 20
        N_t = 4
        K = 2
        M = 4
        P_max = 1.0
        action_dim = calculate_action_dim(N_t, K, M)
        
        sac = SAC(
            state_dim=state_dim,
            action_dim=action_dim,
            N_t=N_t,
            K=K,
            P_max=P_max,
            use_per=False,
            device=torch.device('cpu')
        )
        
        # Add some data to buffer
        for _ in range(50):
            state = np.random.randn(state_dim)
            action = np.random.randn(action_dim)
            reward = np.random.randn()
            next_state = np.random.randn(state_dim)
            sac.store_transition(state, action, reward, next_state)
        
        batch_size = 16
        result = sac._sample_from_buffer(batch_size)
        
        assert len(result) == 6  # states, actions, rewards, next_states, weights, indices
        states, actions, rewards, next_states, weights, indices = result
        
        assert states.shape == (batch_size, state_dim)
        assert actions.shape == (batch_size, action_dim)
        assert rewards.shape == (batch_size, 1)
        assert next_states.shape == (batch_size, state_dim)
        assert weights.shape == (batch_size,)
        assert len(indices) == batch_size

    def test_sample_from_buffer_per(self):
        """Test sampling from PER buffer."""
        state_dim = 20
        N_t = 4
        K = 2
        M = 4
        P_max = 1.0
        action_dim = calculate_action_dim(N_t, K, M)
        
        sac = SAC(
            state_dim=state_dim,
            action_dim=action_dim,
            N_t=N_t,
            K=K,
            P_max=P_max,
            use_per=True,
            device=torch.device('cpu')
        )
        
        # Add some data to buffer
        for _ in range(50):
            state = np.random.randn(state_dim)
            action = np.random.randn(action_dim)
            reward = np.random.randn()
            next_state = np.random.randn(state_dim)
            sac.store_transition(state, action, reward, next_state)
        
        batch_size = 16
        result = sac._sample_from_buffer(batch_size)
        
        assert len(result) == 6
        states, actions, rewards, next_states, weights, indices = result
        
        assert states.shape == (batch_size, state_dim)
        assert actions.shape == (batch_size, action_dim)
        assert rewards.shape == (batch_size, 1)
        assert next_states.shape == (batch_size, state_dim)
        assert weights.shape == (batch_size,)
        assert len(indices) == batch_size
        assert torch.all(weights > 0)  # PER weights should be positive


class TestSACIntegration:
    """Integration tests for SAC algorithm."""

    def test_full_training_cycle(self):
        """Test a complete training cycle."""
        state_dim = 20
        N_t = 4
        K = 2
        M = 4
        P_max = 1.0
        action_dim = calculate_action_dim(N_t, K, M)
        
        sac = SAC(
            state_dim=state_dim,
            action_dim=action_dim,
            N_t=N_t,
            K=K,
            P_max=P_max,
            buffer_size=1000,
            device=torch.device('cpu')
        )
        
        # Simulate environment interaction
        for episode in range(5):
            state = np.random.randn(state_dim)
            for step in range(10):
                action = sac.select_action(state)
                reward = np.random.randn()
                next_state = np.random.randn(state_dim)
                
                sac.store_transition(state, action, reward, next_state)
                state = next_state
                
                # Train every few steps
                if sac.replay_buffer.size >= 32:
                    sac.training(32)
        
        assert sac.total_it > 0
        assert sac.replay_buffer.size > 0

    def test_per_integration(self):
        """Test PER integration with SAC."""
        state_dim = 20
        N_t = 4
        K = 2
        M = 4
        P_max = 1.0
        action_dim = calculate_action_dim(N_t, K, M)
        
        sac = SAC(
            state_dim=state_dim,
            action_dim=action_dim,
            N_t=N_t,
            K=K,
            P_max=P_max,
            buffer_size=1000,
            use_per=True,
            per_alpha=0.6,
            per_beta_start=0.4,
            device=torch.device('cpu')
        )
        
        # Fill buffer and train
        for _ in range(100):
            state = np.random.randn(state_dim)
            action = sac.select_action(state)
            reward = np.random.randn()
            next_state = np.random.randn(state_dim)
            sac.store_transition(state, action, reward, next_state)
        
        # Train with PER
        for _ in range(10):
            sac.training(32)
        
        assert sac.total_it == 10
        assert sac.use_per

    def test_entropy_tuning_integration(self):
        """Test automatic entropy tuning integration."""
        state_dim = 20
        N_t = 4
        K = 2
        M = 4
        P_max = 1.0
        action_dim = calculate_action_dim(N_t, K, M)
        
        sac = SAC(
            state_dim=state_dim,
            action_dim=action_dim,
            N_t=N_t,
            K=K,
            P_max=P_max,
            automatic_entropy_tuning=True,
            target_entropy=-action_dim,
            device=torch.device('cpu')
        )
        
        # Fill buffer and train
        for _ in range(100):
            state = np.random.randn(state_dim)
            action = sac.select_action(state)
            reward = np.random.randn()
            next_state = np.random.randn(state_dim)
            sac.store_transition(state, action, reward, next_state)
        
        # Train multiple steps
        for _ in range(5):
            sac.training(32)
        
        # Check that alpha has been updated
        assert sac.alpha is not None
        assert hasattr(sac, 'log_alpha')


if __name__ == '__main__':
    pytest.main([__file__])
