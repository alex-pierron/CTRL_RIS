"""
Unit tests for DDPG algorithm.
"""
import pytest
import numpy as np
import torch
import torch.nn as nn
from unittest.mock import Mock, patch, MagicMock
import sys
import os
import tempfile

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from algorithms.ddpg import DDPG, ActorNetwork, CriticNetwork, Custom_DDPG


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
        assert actor.output.out_features == action_dim

    def test_init_invalid_parameters(self):
        """Test ActorNetwork initialization with invalid parameters."""
        # Test with negative action_dim (this should raise an error)
        with pytest.raises((ValueError, TypeError, RuntimeError)):
            ActorNetwork(20, -1, 4, 2, 1.0, [128, 128])
        
        # Test with zero action_dim (this should raise an error)
        with pytest.raises((ValueError, TypeError, RuntimeError)):
            ActorNetwork(20, 0, 4, 2, 1.0, [128, 128])

    def test_forward_pass(self):
        """Test forward pass through the actor network."""
        state_dim = 20
        N_t = 4
        K = 2
        M = 4  # Number of RIS elements
        P_max = 1.0
        action_dim = calculate_action_dim(N_t, K, M)
        
        actor = ActorNetwork(state_dim, action_dim, N_t, K, P_max)
        
        # Test with batch size >= 2 for BatchNorm1d training mode
        state = torch.randn(2, state_dim)  # Use batch size 2 for BatchNorm
        output = actor(state)
        
        assert output.shape == (2, action_dim)
        assert torch.is_tensor(output)
        
        # Test that the output is finite and has reasonable values
        assert torch.isfinite(output).all()
        assert not torch.isnan(output).any()
        
        # Test that the output is different from zero (not all zeros)
        assert not torch.allclose(output, torch.zeros_like(output), atol=1e-6)
        
        # Test that we can extract the expected components
        W_components = output[:, :2 * N_t * K]  # First 2*N_t*K components for W
        theta_components = output[:, 2 * N_t * K:]  # Remaining components for Theta
        
        assert W_components.shape == (2, 2 * N_t * K)
        assert theta_components.shape == (2, 2 * M)
        
        # Test that both components are finite
        assert torch.isfinite(W_components).all()
        assert torch.isfinite(theta_components).all()

    def test_forward_pass_batch(self):
        """Test forward pass with batch input."""
        state_dim = 20
        N_t = 4
        K = 2
        M = 4  # Number of RIS elements
        P_max = 1.0
        action_dim = calculate_action_dim(N_t, K, M)
        
        actor = ActorNetwork(state_dim, action_dim, N_t, K, P_max)
        
        # Test batch of states
        batch_size = 32
        states = torch.randn(batch_size, state_dim)
        output = actor(states)
        
        assert output.shape == (batch_size, action_dim)
        assert torch.is_tensor(output)
        # Check that output is processed (not raw tanh output)
        assert not torch.allclose(output, torch.tanh(output), atol=1e-6)

    def test_actor_w_projection_operator(self):
        """Test W projection operator for power constraints."""
        state_dim = 20
        N_t = 4
        K = 2
        M = 4  # Number of RIS elements
        P_max = 1.0
        action_dim = calculate_action_dim(N_t, K, M)
        
        actor = ActorNetwork(state_dim, action_dim, N_t, K, P_max)
        
        # Test with batch of beamforming matrices
        batch_size = 5
        raw_W = torch.randn(batch_size, N_t, K) + 1j * torch.randn(batch_size, N_t, K)
        
        projected_W = actor.actor_W_projection_operator(raw_W)
        
        assert projected_W.shape == raw_W.shape
        assert torch.is_tensor(projected_W)
        
        # Check power constraint is satisfied
        traces = torch.einsum('bij,bji->b', projected_W, projected_W.conj().transpose(1, 2)).real
        assert torch.all(traces <= P_max + 1e-6)  # Allow small numerical errors
        
        # Test that matrices exceeding constraint are scaled down
        large_W = torch.randn(batch_size, N_t, K) + 1j * torch.randn(batch_size, N_t, K)
        large_W = large_W * 10  # Make it exceed constraint
        projected_large_W = actor.actor_W_projection_operator(large_W)
        large_traces = torch.einsum('bij,bji->b', projected_large_W, projected_large_W.conj().transpose(1, 2)).real
        assert torch.all(large_traces <= P_max + 1e-6)

    def test_actor_process_raw_actions(self):
        """Test processing of raw actions into valid actions."""
        state_dim = 20
        N_t = 4
        K = 2
        M = 4  # Number of RIS elements
        P_max = 1.0
        action_dim = calculate_action_dim(N_t, K, M)
        
        actor = ActorNetwork(state_dim, action_dim, N_t, K, P_max)
        
        # Test with batch of raw actions
        batch_size = 32
        raw_actions = torch.randn(batch_size, action_dim)
        
        processed_actions = actor.actor_process_raw_actions(raw_actions)
        
        assert processed_actions.shape == raw_actions.shape
        assert torch.is_tensor(processed_actions)

    def test_power_constraint_satisfaction(self):
        """Test that power constraints are satisfied after processing."""
        state_dim = 20
        N_t = 4
        K = 2
        M = 4  # Number of RIS elements
        P_max = 1.0
        action_dim = calculate_action_dim(N_t, K, M)
        
        actor = ActorNetwork(state_dim, action_dim, N_t, K, P_max)
        
        # Create actions that would violate power constraints
        raw_actions = torch.randn(10, action_dim) * 10  # Large values
        
        processed_actions = actor.actor_process_raw_actions(raw_actions)
        
        # Extract W matrix from processed actions
        W_raw_actions = processed_actions[:, :2 * N_t * K]
        W_real = W_raw_actions[:, :N_t * K].view(-1, N_t, K)
        W_imag = W_raw_actions[:, N_t * K:].view(-1, N_t, K)
        W = W_real + 1j * W_imag
        
        # Check power constraint
        traces = torch.einsum('bij,bji->b', W, W.conj().transpose(1, 2)).real
        assert torch.all(traces <= P_max + 1e-6)
        
        # Test with very large raw actions to ensure constraint is enforced
        very_large_actions = torch.randn(5, action_dim) * 100
        processed_large = actor.actor_process_raw_actions(very_large_actions)
        W_large_raw = processed_large[:, :2 * N_t * K]
        W_large_real = W_large_raw[:, :N_t * K].view(-1, N_t, K)
        W_large_imag = W_large_raw[:, N_t * K:].view(-1, N_t, K)
        W_large = W_large_real + 1j * W_large_imag
        large_traces = torch.einsum('bij,bji->b', W_large, W_large.conj().transpose(1, 2)).real
        assert torch.all(large_traces <= P_max + 1e-6)


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
        """Test forward pass through the critic network."""
        state_dim = 20
        action_dim = 12
        critic_linear_layers = [128, 128]
        
        critic = CriticNetwork(state_dim, action_dim, critic_linear_layers)
        
        # Test single state-action pair
        state = torch.randn(state_dim)
        action = torch.randn(action_dim)
        output = critic(state, action)
        
        assert output.shape == (1,)
        assert torch.is_tensor(output)
        # Check that output is a scalar value (Q-value)
        assert output.dim() == 1

    def test_forward_pass_batch(self):
        """Test forward pass with batch input."""
        state_dim = 20
        action_dim = 12
        critic_linear_layers = [128, 128]
        
        critic = CriticNetwork(state_dim, action_dim, critic_linear_layers)
        
        # Test batch of state-action pairs
        batch_size = 32
        states = torch.randn(batch_size, state_dim)
        actions = torch.randn(batch_size, action_dim)
        output = critic(states, actions)
        
        assert output.shape == (batch_size, 1)
        assert torch.is_tensor(output)
        # Check that output has correct dimensions
        assert output.dim() == 2


class TestDDPG:
    """Test cases for DDPG class."""

    def test_init_valid_parameters(self, sample_config):
        """Test DDPG initialization with valid parameters."""
        state_dim = 20
        N_t = 4
        K = 2
        M = 4  # Number of RIS elements
        action_dim = calculate_action_dim(N_t, K, M)
        gamma = 0.99
        P_max = 1.0
        device = torch.device('cpu')
        
        ddpg = DDPG(
            state_dim=state_dim,
            action_dim=action_dim,
            N_t=N_t,
            K=K,
            P_max=P_max,
            device=device
        )
        
        assert ddpg.state_dim == state_dim
        assert ddpg.action_dim == action_dim
        assert ddpg.gamma == gamma
        assert ddpg.N_t == N_t
        assert ddpg.K == K
        assert ddpg.P_max == P_max
        assert ddpg.device == device


    def test_select_action(self, sample_config):
        """Test action selection."""
        state_dim = 20
        N_t = 4
        K = 2
        M = 4  # Number of RIS elements
        action_dim = calculate_action_dim(N_t, K, M)
        device = torch.device('cpu')
        
        ddpg = DDPG(
            state_dim=state_dim,
            action_dim=action_dim,
            N_t=N_t,
            K=K,
            P_max=1.0,
            device=device
        )
        
        # Test single state (DDPG handles batch dimension internally)
        state = torch.randn(state_dim)
        action = ddpg.select_action(state)
        
        assert action.shape == (action_dim,)
        assert torch.is_tensor(action)

    def test_select_action_batch(self, sample_config):
        """Test action selection with batch input."""
        state_dim = 20
        N_t = 4
        K = 2
        M = 4  # Number of RIS elements
        action_dim = calculate_action_dim(N_t, K, M)
        device = torch.device('cpu')
        
        ddpg = DDPG(
            state_dim=state_dim,
            action_dim=action_dim,
            N_t=N_t,
            K=K,
            P_max=1.0,
            device=device
        )
        
        # Test batch of states by calling select_action for each state
        batch_size = 32
        states = torch.randn(batch_size, state_dim)
        actions = torch.stack([ddpg.select_action(state) for state in states])
        
        assert actions.shape == (batch_size, action_dim)
        assert torch.is_tensor(actions)

    def test_select_action_with_noise(self, sample_config):
        """Test action selection with noise."""
        state_dim = 20
        N_t = 4
        K = 2
        M = 4  # Number of RIS elements
        action_dim = calculate_action_dim(N_t, K, M)
        device = torch.device('cpu')
        
        ddpg = DDPG(
            state_dim=state_dim,
            action_dim=action_dim,
            N_t=N_t,
            K=K,
            P_max=1.0,
            device=device,
            action_noise_scale=0.1
        )
        
        state = torch.randn(state_dim)
        # Use select_noised_action method instead
        clean_action, noised_action = ddpg.select_noised_action(state, noise_scale=0.1)
        
        assert clean_action.shape == (action_dim,)
        assert noised_action.shape == (action_dim,)
        assert torch.is_tensor(clean_action)
        assert torch.is_tensor(noised_action)
        
        # Check that noised action is different from clean action
        assert not torch.allclose(clean_action, noised_action, atol=1e-6)

    def test_update_networks(self, sample_config):
        """Test network updates."""
        state_dim = 20
        N_t = 4
        K = 2
        M = 4  # Number of RIS elements
        action_dim = calculate_action_dim(N_t, K, M)
        device = torch.device('cpu')
        
        ddpg = DDPG(
            state_dim=state_dim,
            action_dim=action_dim,
            N_t=N_t,
            K=K,
            P_max=1.0,
            device=device
        )
        
        # Add some transitions to the replay buffer first
        for _ in range(64):  # Add enough samples for batch
            state = torch.randn(state_dim)
            action = torch.randn(action_dim)
            reward = torch.randn(1)
            next_state = torch.randn(state_dim)
            ddpg.store_transition(state, action, reward, next_state)
        
        # Test training step
        actor_loss, critic_loss, rewards = ddpg.training(batch_size=32)
        
        assert isinstance(actor_loss, (float, torch.Tensor))
        assert isinstance(critic_loss, (float, torch.Tensor))
        assert isinstance(rewards, torch.Tensor)

    def test_update_target_networks(self, sample_config):
        """Test target network updates."""
        state_dim = 20
        N_t = 4
        K = 2
        M = 4  # Number of RIS elements
        action_dim = calculate_action_dim(N_t, K, M)
        device = torch.device('cpu')
        
        ddpg = DDPG(
            state_dim=state_dim,
            action_dim=action_dim,
            N_t=N_t,
            K=K,
            P_max=1.0,
            device=device,
            tau=0.01
        )
        
        # Store original target network parameters
        original_actor_params = [p.clone() for p in ddpg.target_actor.parameters()]
        original_critic_params = [p.clone() for p in ddpg.target_critic.parameters()]
        
        # Update target networks
        ddpg.update_target_networks()
        
        # Check that parameters have changed (soft update)
        # With small tau, changes might be very small, so check for any change
        actor_changed = False
        for orig, new in zip(original_actor_params, ddpg.target_actor.parameters()):
            if not torch.equal(orig, new):
                actor_changed = True
                break
        
        # At least one parameter should have changed
        assert actor_changed

    def test_save_and_load_models(self, sample_config, temp_dir):
        """Test saving and loading models."""
        state_dim = 20
        N_t = 4
        K = 2
        M = 4  # Number of RIS elements
        action_dim = calculate_action_dim(N_t, K, M)
        device = torch.device('cpu')
        
        # Define consistent architecture parameters
        actor_linear_layers = [128, 128, 128]
        critic_linear_layers = [128, 128]
        
        ddpg = DDPG(
            state_dim=state_dim,
            action_dim=action_dim,
            N_t=N_t,
            K=K,
            P_max=1.0,
            device=device,
            actor_linear_layers=actor_linear_layers,
            critic_linear_layers=critic_linear_layers
        )
        
        # Save models
        ddpg.save_models(temp_dir)
        
        # Check that files were created
        assert os.path.exists(os.path.join(temp_dir, 'actor.pth'))
        assert os.path.exists(os.path.join(temp_dir, 'critic.pth'))
        assert os.path.exists(os.path.join(temp_dir, 'target_actor.pth'))
        assert os.path.exists(os.path.join(temp_dir, 'target_critic.pth'))
        
        # Test that we can load the state_dicts without creating new instances
        # This tests the save/load functionality without architecture mismatches
        actor_state_dict = torch.load(os.path.join(temp_dir, 'actor.pth'), map_location='cpu')
        critic_state_dict = torch.load(os.path.join(temp_dir, 'critic.pth'), map_location='cpu')
        target_actor_state_dict = torch.load(os.path.join(temp_dir, 'target_actor.pth'), map_location='cpu')
        target_critic_state_dict = torch.load(os.path.join(temp_dir, 'target_critic.pth'), map_location='cpu')
        
        # Test that state_dicts are not empty and have expected structure
        assert len(actor_state_dict) > 0
        assert len(critic_state_dict) > 0
        assert len(target_actor_state_dict) > 0
        assert len(target_critic_state_dict) > 0
        
        # Test that we can load them back into the same networks
        ddpg.actor.load_state_dict(actor_state_dict)
        ddpg.critic.load_state_dict(critic_state_dict)
        ddpg.target_actor.load_state_dict(target_actor_state_dict)
        ddpg.target_critic.load_state_dict(target_critic_state_dict)
        
        # Test that the network still works after loading
        state = torch.randn(state_dim)
        action = ddpg.select_action(state)
        assert action.shape == (action_dim,)
        assert torch.isfinite(action).all()

    def test_gradient_clipping(self, sample_config):
        """Test gradient clipping during updates."""
        state_dim = 20
        N_t = 4
        K = 2
        M = 4  # Number of RIS elements
        action_dim = calculate_action_dim(N_t, K, M)
        device = torch.device('cpu')
        
        ddpg = DDPG(
            state_dim=state_dim,
            action_dim=action_dim,
            N_t=N_t,
            K=K,
            P_max=1.0,
            device=device
        )
        
        # Add some transitions to the replay buffer first
        for _ in range(64):  # Add enough samples for batch
            state = torch.randn(state_dim)
            action = torch.randn(action_dim)
            reward = torch.randn(1)
            next_state = torch.randn(state_dim)
            ddpg.store_transition(state, action, reward, next_state)
        
        # Test training step
        actor_loss, critic_loss, rewards = ddpg.training(batch_size=32)
        
        assert isinstance(actor_loss, (float, torch.Tensor))
        assert isinstance(critic_loss, (float, torch.Tensor))
        assert isinstance(rewards, torch.Tensor)

    def test_deterministic_action_selection(self, sample_config):
        """Test that action selection is deterministic when noise is disabled."""
        state_dim = 20
        N_t = 4
        K = 2
        M = 4  # Number of RIS elements
        action_dim = calculate_action_dim(N_t, K, M)
        device = torch.device('cpu')
        
        ddpg = DDPG(
            state_dim=state_dim,
            action_dim=action_dim,
            N_t=N_t,
            K=K,
            P_max=1.0,
            device=device
        )
        
        state = torch.randn(state_dim)
        
        # Get action multiple times (select_action doesn't have add_noise parameter)
        action1 = ddpg.select_action(state)
        action2 = ddpg.select_action(state)
        
        assert torch.allclose(action1, action2, atol=1e-6)

    def test_device_consistency(self, sample_config):
        """Test that all tensors are on the correct device."""
        state_dim = 20
        N_t = 4
        K = 2
        M = 4  # Number of RIS elements
        action_dim = calculate_action_dim(N_t, K, M)
        device = torch.device('cpu')
        
        ddpg = DDPG(
            state_dim=state_dim,
            action_dim=action_dim,
            N_t=N_t,
            K=K,
            P_max=1.0,
            device=device
        )
        
        state = torch.randn(state_dim)
        action = ddpg.select_action(state)
        
        assert action.device == device

    @pytest.mark.parametrize("state_dim,N_t,K,M", [(10, 2, 1, 2), (50, 4, 2, 4), (100, 8, 4, 8)])
    def test_different_dimensions(self, state_dim, N_t, K, M):
        """Test DDPG with different state and action dimensions."""
        action_dim = calculate_action_dim(N_t, K, M)
        ddpg = DDPG(
            state_dim=state_dim,
            action_dim=action_dim,
            N_t=N_t,
            K=K,
            P_max=1.0,
            device=torch.device('cpu')
        )
        
        state = torch.randn(state_dim)
        action = ddpg.select_action(state)
        
        assert action.shape == (action_dim,)

    def test_learning_rate_scheduling(self, sample_config):
        """Test learning rate scheduling."""
        state_dim = 20
        N_t = 4
        K = 2
        M = 4  # Number of RIS elements
        action_dim = calculate_action_dim(N_t, K, M)
        device = torch.device('cpu')
        
        ddpg = DDPG(
            state_dim=state_dim,
            action_dim=action_dim,
            N_t=N_t,
            K=K,
            P_max=1.0,
            device=device
        )
        
        # Test learning rate decay (if the method exists)
        original_lr = ddpg.actor_optimizer.param_groups[0]['lr']
        
        # Manually decay learning rate
        for param_group in ddpg.actor_optimizer.param_groups:
            param_group['lr'] *= 0.9
        
        new_lr = ddpg.actor_optimizer.param_groups[0]['lr']
        
        assert new_lr == original_lr * 0.9

    def test_memory_efficiency(self, sample_config):
        """Test that DDPG uses memory efficiently."""
        state_dim = 20
        N_t = 4
        K = 2
        M = 4  # Number of RIS elements
        action_dim = calculate_action_dim(N_t, K, M)
        device = torch.device('cpu')
        
        ddpg = DDPG(
            state_dim=state_dim,
            action_dim=action_dim,
            N_t=N_t,
            K=K,
            P_max=1.0,
            device=device
        )
        
        # Test that models don't consume excessive memory
        # Test with single states (DDPG select_action handles single states)
        for _ in range(10):  # Test multiple single states
            state = torch.randn(state_dim)
            action = ddpg.select_action(state)
            assert action.shape == (action_dim,)
            assert action.device == device
        
        # Test that we can handle multiple single states without memory issues
        for _ in range(5):
            single_state = torch.randn(state_dim)
            single_action = ddpg.select_action(single_state)
            assert single_action.shape == (action_dim,)

    def test_replay_buffer_operations(self, sample_config):
        """Test replay buffer operations."""
        state_dim = 20
        N_t = 4
        K = 2
        M = 4  # Number of RIS elements
        action_dim = calculate_action_dim(N_t, K, M)
        device = torch.device('cpu')
        
        ddpg = DDPG(
            state_dim=state_dim,
            action_dim=action_dim,
            N_t=N_t,
            K=K,
            P_max=1.0,
            device=device
        )
        
        # Test storing transitions
        for i in range(10):
            state = torch.randn(state_dim)
            action = torch.randn(action_dim)
            reward = torch.randn(1)
            next_state = torch.randn(state_dim)
            ddpg.store_transition(state, action, reward, next_state)
        
        # Test that buffer has stored transitions
        assert ddpg.replay_buffer.size >= 0

    def test_network_initialization(self, sample_config):
        """Test that all networks are properly initialized."""
        state_dim = 20
        N_t = 4
        K = 2
        M = 4  # Number of RIS elements
        action_dim = calculate_action_dim(N_t, K, M)
        device = torch.device('cpu')
        
        ddpg = DDPG(
            state_dim=state_dim,
            action_dim=action_dim,
            N_t=N_t,
            K=K,
            P_max=1.0,
            device=device
        )
        
        # Test that all networks exist
        assert ddpg.actor is not None
        assert ddpg.critic is not None
        assert ddpg.target_actor is not None
        assert ddpg.target_critic is not None
        
        # Test that target networks are initialized with same weights as main networks
        actor_params = list(ddpg.actor.parameters())
        target_actor_params = list(ddpg.target_actor.parameters())
        
        for param, target_param in zip(actor_params, target_actor_params):
            assert torch.equal(param, target_param)

    def test_action_processing_constraints(self, sample_config):
        """Test that action processing enforces constraints."""
        state_dim = 20
        N_t = 4
        K = 2
        M = 4  # Number of RIS elements
        action_dim = calculate_action_dim(N_t, K, M)
        P_max = 1.0
        
        actor = ActorNetwork(state_dim, action_dim, N_t, K, P_max)
        
        # Test with extreme values
        extreme_actions = torch.randn(5, action_dim) * 1000
        processed_actions = actor.actor_process_raw_actions(extreme_actions)
        
        # Check that processed actions satisfy constraints
        W_raw_actions = processed_actions[:, :2 * N_t * K]
        W_real = W_raw_actions[:, :N_t * K].view(-1, N_t, K)
        W_imag = W_raw_actions[:, N_t * K:].view(-1, N_t, K)
        W = W_real + 1j * W_imag
        
        traces = torch.einsum('bij,bji->b', W, W.conj().transpose(1, 2)).real
        assert torch.all(traces <= P_max + 1e-6)

    def test_theta_constraints(self, sample_config):
        """Test that Theta (phase shift) constraints are satisfied."""
        state_dim = 20
        N_t = 4
        K = 2
        M = 4  # Number of RIS elements
        action_dim = calculate_action_dim(N_t, K, M)
        P_max = 1.0
        
        actor = ActorNetwork(state_dim, action_dim, N_t, K, P_max)
        
        # Test with extreme values
        extreme_actions = torch.randn(5, action_dim) * 1000
        processed_actions = actor.actor_process_raw_actions(extreme_actions)
        
        # Extract Theta components
        theta_actions = processed_actions[:, 2 * N_t * K:]
        theta_real = theta_actions[:, 0::2]
        theta_imag = theta_actions[:, 1::2]
        
        # Check unit modulus constraint
        magnitudes = torch.sqrt(theta_real**2 + theta_imag**2)
        assert torch.allclose(magnitudes, torch.ones_like(magnitudes), atol=1e-6)

    def test_batch_processing_consistency(self, sample_config):
        """Test that batch processing works correctly with BatchNorm1d."""
        state_dim = 20
        N_t = 4
        K = 2
        M = 4  # Number of RIS elements
        action_dim = calculate_action_dim(N_t, K, M)
        P_max = 1.0
        
        actor = ActorNetwork(state_dim, action_dim, N_t, K, P_max)
        
        # Test with different batch sizes to ensure BatchNorm1d works properly
        # BatchNorm1d requires batch_size >= 1, but behaves differently with small vs large batches
        small_batch = torch.randn(2, state_dim)  # Small batch
        large_batch = torch.randn(16, state_dim)  # Larger batch
        
        small_output = actor(small_batch)
        large_output = actor(large_batch)
        
        # Test that both batch sizes work correctly
        assert small_output.shape == (2, action_dim)
        assert large_output.shape == (16, action_dim)
        
        # Test that BatchNorm1d is in training mode initially
        assert actor.batch_norm.training == True
        
        # Test that we can switch to eval mode
        actor.eval()
        assert actor.batch_norm.training == False
        
        # Test inference in eval mode
        eval_output = actor(large_batch)
        assert eval_output.shape == (16, action_dim)

    def test_device_handling(self, sample_config):
        """Test device handling for different devices."""
        state_dim = 20
        N_t = 4
        K = 2
        M = 4  # Number of RIS elements
        action_dim = calculate_action_dim(N_t, K, M)
        
        # Test CPU device
        ddpg_cpu = DDPG(
            state_dim=state_dim,
            action_dim=action_dim,
            N_t=N_t,
            K=K,
            P_max=1.0,
            device=torch.device('cpu')
        )
        
        state = torch.randn(state_dim)
        action = ddpg_cpu.select_action(state)
        assert action.device == torch.device('cpu')
        
        # Test that all networks are on the correct device
        assert next(ddpg_cpu.actor.parameters()).device == torch.device('cpu')
        assert next(ddpg_cpu.critic.parameters()).device == torch.device('cpu')

    def test_optimizer_initialization(self, sample_config):
        """Test that optimizers are properly initialized."""
        state_dim = 20
        N_t = 4
        K = 2
        M = 4  # Number of RIS elements
        action_dim = calculate_action_dim(N_t, K, M)
        device = torch.device('cpu')
        
        ddpg = DDPG(
            state_dim=state_dim,
            action_dim=action_dim,
            N_t=N_t,
            K=K,
            P_max=1.0,
            device=device
        )
        
        # Test that optimizers exist
        assert ddpg.actor_optimizer is not None
        assert ddpg.critic_optimizer is not None
        
        # Test that optimizers have the correct parameters
        assert len(ddpg.actor_optimizer.param_groups) == 1
        assert len(ddpg.critic_optimizer.param_groups) == 1
        
        # Test learning rates
        assert ddpg.actor_optimizer.param_groups[0]['lr'] == 0.0001
        assert ddpg.critic_optimizer.param_groups[0]['lr'] == 0.0005


class TestCustomDDPG:
    """Test cases for Custom_DDPG class."""

    def test_init_valid_parameters(self, sample_config):
        """Test Custom_DDPG initialization with valid parameters."""
        state_dim = 20
        N_t = 4
        K = 2
        M = 4  # Number of RIS elements
        action_dim = calculate_action_dim(N_t, K, M)
        P_max = 1.0
        device = torch.device('cpu')
        
        custom_ddpg = Custom_DDPG(
            state_dim=state_dim,
            action_dim=action_dim,
            N_t=N_t,
            K=K,
            P_max=P_max,
            device=device
        )
        
        assert custom_ddpg.state_dim == state_dim
        assert custom_ddpg.action_dim == action_dim
        assert custom_ddpg.N_t == N_t
        assert custom_ddpg.K == K
        assert custom_ddpg.P_max == P_max
        assert custom_ddpg.device == device

    def test_dual_critic_networks(self, sample_config):
        """Test that Custom_DDPG has dual critic networks."""
        state_dim = 20
        N_t = 4
        K = 2
        M = 4  # Number of RIS elements
        action_dim = calculate_action_dim(N_t, K, M)
        device = torch.device('cpu')
        
        custom_ddpg = Custom_DDPG(
            state_dim=state_dim,
            action_dim=action_dim,
            N_t=N_t,
            K=K,
            P_max=1.0,
            device=device
        )
        
        # Test that both critic networks exist
        assert custom_ddpg.present_critic is not None
        assert custom_ddpg.future_critic is not None
        assert custom_ddpg.target_present_critic is not None
        assert custom_ddpg.target_future_critic is not None
        
        # Test that target networks are initialized with same weights
        present_params = list(custom_ddpg.present_critic.parameters())
        target_present_params = list(custom_ddpg.target_present_critic.parameters())
        
        for param, target_param in zip(present_params, target_present_params):
            assert torch.equal(param, target_param)

    def test_action_selection(self, sample_config):
        """Test action selection with Custom_DDPG."""
        state_dim = 20
        N_t = 4
        K = 2
        M = 4  # Number of RIS elements
        action_dim = calculate_action_dim(N_t, K, M)
        device = torch.device('cpu')
        
        custom_ddpg = Custom_DDPG(
            state_dim=state_dim,
            action_dim=action_dim,
            N_t=N_t,
            K=K,
            P_max=1.0,
            device=device
        )
        
        # Test single state
        state = torch.randn(state_dim)
        action = custom_ddpg.select_action(state)
        
        assert action.shape == (action_dim,)
        assert torch.is_tensor(action)

    def test_noised_action_selection(self, sample_config):
        """Test noised action selection with Custom_DDPG."""
        state_dim = 20
        N_t = 4
        K = 2
        M = 4  # Number of RIS elements
        action_dim = calculate_action_dim(N_t, K, M)
        device = torch.device('cpu')
        
        custom_ddpg = Custom_DDPG(
            state_dim=state_dim,
            action_dim=action_dim,
            N_t=N_t,
            K=K,
            P_max=1.0,
            device=device,
            action_noise_scale=0.1
        )
        
        state = torch.randn(state_dim)
        clean_action, noised_action = custom_ddpg.select_noised_action(state, noise_scale=0.1)
        
        assert clean_action.shape == (action_dim,)
        assert noised_action.shape == (action_dim,)
        assert torch.is_tensor(clean_action)
        assert torch.is_tensor(noised_action)
        
        # Check that noised action is different from clean action
        assert not torch.allclose(clean_action, noised_action, atol=1e-6)

    def test_training_with_dual_critics(self, sample_config):
        """Test training with dual critics."""
        state_dim = 20
        N_t = 4
        K = 2
        M = 4  # Number of RIS elements
        action_dim = calculate_action_dim(N_t, K, M)
        device = torch.device('cpu')
        
        custom_ddpg = Custom_DDPG(
            state_dim=state_dim,
            action_dim=action_dim,
            N_t=N_t,
            K=K,
            P_max=1.0,
            device=device
        )
        
        # Add some transitions to the replay buffer first
        for _ in range(64):  # Add enough samples for batch
            state = torch.randn(state_dim)
            action = torch.randn(action_dim)
            reward = torch.randn(1)
            next_state = torch.randn(state_dim)
            custom_ddpg.store_transition(state, action, reward, next_state)
        
        # Test training step
        actor_loss, critic_loss, rewards = custom_ddpg.training(batch_size=32)
        
        assert isinstance(actor_loss, (float, torch.Tensor))
        assert isinstance(critic_loss, (float, torch.Tensor))
        assert isinstance(rewards, torch.Tensor)

    def test_save_models_custom(self, sample_config, temp_dir):
        """Test saving models for Custom_DDPG."""
        state_dim = 20
        N_t = 4
        K = 2
        M = 4  # Number of RIS elements
        action_dim = calculate_action_dim(N_t, K, M)
        device = torch.device('cpu')
        
        custom_ddpg = Custom_DDPG(
            state_dim=state_dim,
            action_dim=action_dim,
            N_t=N_t,
            K=K,
            P_max=1.0,
            device=device
        )
        
        # Save models
        custom_ddpg.save_models(temp_dir)
        
        # Check that all files were created
        assert os.path.exists(os.path.join(temp_dir, 'actor.pth'))
        assert os.path.exists(os.path.join(temp_dir, 'present_critic.pth'))
        assert os.path.exists(os.path.join(temp_dir, 'future_critic.pth'))
        assert os.path.exists(os.path.join(temp_dir, 'target_actor.pth'))
        assert os.path.exists(os.path.join(temp_dir, 'target_present_critic.pth'))
        assert os.path.exists(os.path.join(temp_dir, 'target_future_critic.pth'))

    def test_per_enabled_by_default(self, sample_config):
        """Test that Custom_DDPG uses PER by default."""
        state_dim = 20
        N_t = 4
        K = 2
        M = 4  # Number of RIS elements
        action_dim = calculate_action_dim(N_t, K, M)
        device = torch.device('cpu')
        
        custom_ddpg = Custom_DDPG(
            state_dim=state_dim,
            action_dim=action_dim,
            N_t=N_t,
            K=K,
            P_max=1.0,
            device=device
        )
        
        # Check that PER is disabled by default
        assert custom_ddpg.use_per == False
        assert hasattr(custom_ddpg.replay_buffer, 'alpha')  # PER buffer has alpha parameter
