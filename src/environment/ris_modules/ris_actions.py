"""
Action processing and matrix computations for RIS Duplex environment.

This module handles:
- Processing raw actor actions into Theta and W matrices
- Converting raw tanh output to constrained actions (unit modulus for Theta, power for W)
- Computing cached matrix products (Theta_Phi, WWH, etc.)
- Updating decoding matrices
"""
import numpy as np
import torch


def process_raw_actions_torch(raw_actions, N_t, K, P_max, device=None, w_action_mapping="projection"):
    """
    Transform raw model output (tanh) into constrained actions: unit-modulus Theta
    and power-projected W. Works for batch processing.
    
    This function is intended to be called outside the neural network forward pass,
    so that gradient computation is not affected by the projection operators.
    It can be used by both algorithms (during training/selection) and the environment.

    Args:
        raw_actions (torch.Tensor): Raw output of the model (tanh). Shape: (batch_size, action_dim).
        N_t (int): Number of transmit antennas.
        K (int): Number of users.
        P_max (float): Maximum power constraint at the Base Station.
        device (torch.device, optional): Device for tensors. Defaults to raw_actions.device.

    Returns:
        torch.Tensor: Processed actions of the same shape as raw_actions.
    """
    if device is None:
        device = raw_actions.device
    tensored_P_max = torch.tensor(P_max, dtype=raw_actions.dtype, device=device)
    
    batch_size = raw_actions.shape[0]
    actions = torch.zeros_like(raw_actions)

    # W-related and Theta-related components
    W_raw_actions = raw_actions[:, :2 * N_t * K]
    theta_actions = raw_actions[:, 2 * N_t * K:]

    # Process Theta: normalize to unit modulus
    theta_real = theta_actions[:, 0::2]
    theta_imag = theta_actions[:, 1::2]
    magnitudes = torch.sqrt(theta_real**2 + theta_imag**2)
    magnitudes = torch.where(magnitudes == 0, torch.ones_like(magnitudes), magnitudes)
    normalized_theta_real = theta_real / magnitudes
    normalized_theta_imag = theta_imag / magnitudes
    actions[:, 2 * N_t * K::2] = normalized_theta_real
    actions[:, 2 * N_t * K + 1::2] = normalized_theta_imag

    # Process W according to the configured mapping.
    W_raw_actions = W_raw_actions.reshape(batch_size, K, 2 * N_t)
    W_real = W_raw_actions[:, :, 0::2]
    W_imag = W_raw_actions[:, :, 1::2]
    raw_W = W_real + 1j * W_imag

    if w_action_mapping == "direction_power":
        W = _w_direction_power_operator(raw_W, tensored_P_max)
    else:
        W = _w_projection_operator(raw_W, tensored_P_max)
    flattened_real = W.real.flatten(start_dim=1)
    flattened_imag = W.imag.flatten(start_dim=1)
    actions[:, :2 * N_t * K:2] = flattened_real
    actions[:, 1:2 * N_t * K:2] = flattened_imag

    return actions


def _w_projection_operator(raw_W, tensored_P_max):
    """
    Project a batch of beamforming matrices W onto the power constraint set.

    Args:
        raw_W (torch.Tensor): Batch of raw beamforming matrices, shape (batch_size, K, N_t).
        tensored_P_max (torch.Tensor): Scalar maximum power (on same device/dtype as raw_W).

    Returns:
        torch.Tensor: Projected beamforming matrices.
    """
    frobenius_norms = torch.linalg.norm(raw_W, dim=(1, 2), ord='fro')
    traces = torch.einsum('bij,bji->b', raw_W, raw_W.conj().transpose(1, 2)).real
    exceed_mask = traces > tensored_P_max
    scaling_factors = torch.where(
        exceed_mask,
        (tensored_P_max.sqrt() / frobenius_norms),
        torch.ones_like(frobenius_norms)
    )
    scaling_factors = scaling_factors.view(-1, 1, 1)
    return raw_W * scaling_factors


def _w_direction_power_operator(raw_W, tensored_P_max):
    """Map raw W to direction+power with explicit amplitude control.

    For each user k, the first complex coefficient controls power through magnitude,
    while the full vector controls direction. The resulting per-user beam has norm
    sqrt(power_k), and total power is then globally clipped to P_max if needed.
    """
    eps = 1e-8
    amplitudes = torch.abs(raw_W[:, :, :1]).real
    amplitudes = torch.tanh(amplitudes)
    power_per_user = (amplitudes + 1.0) * 0.5 * tensored_P_max / max(1, raw_W.shape[1])

    norms = torch.linalg.norm(raw_W, dim=2, keepdim=True).clamp_min(eps)
    directions = raw_W / norms
    W = directions * torch.sqrt(power_per_user.clamp_min(0.0))
    return _w_projection_operator(W, tensored_P_max)


def process_raw_actions_numpy(raw_actions, N_t, K, P_max, w_action_mapping="projection"):
    """
    Numpy version of process_raw_actions_torch for use when inputs are numpy arrays
    (e.g. in the environment when receiving from external sources).

    Args:
        raw_actions (np.ndarray): Raw actions, shape (action_dim,) or (batch_size, action_dim).
        N_t, K, P_max: Same as process_raw_actions_torch.

    Returns:
        np.ndarray: Processed actions, same shape as input.
    """
    was_1d = (raw_actions.ndim == 1)
    if was_1d:
        raw_actions = raw_actions.reshape(1, -1)
    tensor = torch.from_numpy(np.asarray(raw_actions, dtype=np.float32))
    processed = process_raw_actions_torch(
        tensor, N_t, K, P_max, w_action_mapping=w_action_mapping
    )
    result = processed.numpy()
    return result.squeeze(0) if was_1d else result


class ActionProcessor:
    """Handles action processing and matrix computations."""
    
    @staticmethod
    def process_raw_actions(actor_actions, N_t, K, M):
        """Convert raw actor output tensor into `Theta` and `W`.

        Args:
            actor_actions: Raw output of the model for actions to take.
            N_t: Number of BS transmit antennas
            K: Number of users
            M: Number of RIS elements

        Returns:
            tuple: (Theta, W) matrices
        """
        current_actions = actor_actions.numpy() if hasattr(actor_actions, 'numpy') else actor_actions
        
        # Update Theta
        normalized_theta_real = current_actions[2 * N_t * K::2] 
        normalized_theta_imag = current_actions[2 * N_t * K + 1::2] 
        Theta = np.diag(normalized_theta_real + 1j * normalized_theta_imag)

        # Update W
        W_flattened_real = current_actions[:2 * N_t * K:2]
        W_flattened_imag = current_actions[1:2 * N_t * K:2]
        temporary_W = W_flattened_real + 1j * W_flattened_imag
        
        # Reshape using column-major ordering
        W = temporary_W.reshape(K, N_t).T
        
        return Theta, W
    
    @staticmethod
    def compute_theta_phi(Theta, Phi):
        """Update cached products of `Theta` and `Phi` used across formulas.
        
        Args:
            Theta: RIS phase shift matrix
            Phi: Phase noise matrix
            
        Returns:
            tuple: (Theta_Phi, Phi_H_Theta_H) matrices
        """
        diagonal_elements = np.diag(Theta @ Phi) 
        Theta_Phi = np.diag(diagonal_elements / np.abs(diagonal_elements)) 
        # NOTE: Phi_H_Theta_H must be the Hermitian conjugate of the *normalized*
        # Theta_Phi, not of the raw Phi @ Theta. Otherwise the distortion terms
        # (when re-enabled) will be inconsistent with Theta_Phi.
        Phi_H_Theta_H = Theta_Phi.conj().T
        return Theta_Phi, Phi_H_Theta_H

    @staticmethod
    def compute_WWH(W):
        """Update cached `W @ W^H` and its diagonal; reused in SINR terms.
        
        Args:
            W: Beamforming matrix
            
        Returns:
            tuple: (WWH, diag_matrix_WWH) matrices
        """
        WWH = W @ W.conj().T
        diag_matrix_WWH = np.diag(np.diag(WWH)).real
        return WWH, diag_matrix_WWH
    
    @staticmethod
    def cache_matrix_products(Theta_Phi, W, channel_matrices, uplink_used, eavesdropper_active, 
                             num_eavesdroppers, N_r, K, M, last_theta_phi=None, last_W=None):
        """Cache expensive matrix products used in state building.
        
        Args:
            Theta_Phi: Product of Theta and Phi
            W: Beamforming matrix
            channel_matrices: Dictionary of channel matrices
            uplink_used: Whether uplink is used
            eavesdropper_active: Whether eavesdroppers are active
            num_eavesdroppers: Number of eavesdroppers
            N_r: Number of BS receive antennas
            K: Number of users
            M: Number of RIS elements
            last_theta_phi: Previous Theta_Phi for change detection
            last_W: Previous W for change detection
            
        Returns:
            dict: Cached matrix products and updated cache markers
        """
        # Check if matrices have changed
        theta_phi_changed = (last_theta_phi is None or 
                            not np.array_equal(last_theta_phi, Theta_Phi))
        W_changed = (last_W is None or 
                    not np.array_equal(last_W, W))
        
        cached_products = {}
        
        if theta_phi_changed or W_changed:
            # Cache BS-RIS-LU channel products
            h_d = channel_matrices["H_RIS_Users"].squeeze(axis=1)  # Shape: (K,M)
            cached_products["G_1D"] = h_d @ Theta_Phi @ channel_matrices["H_BS_RIS"] @ W
            
            # Cache LU-RIS-BS channel products  
            if uplink_used:
                h_u = channel_matrices["H_Users_RIS"].squeeze(axis=2).T  # Shape: (M, K)
                cached_products["LU_BS_RIS"] = channel_matrices["H_RIS_BS"] @ Theta_Phi @ h_u
            else:
                cached_products["LU_BS_RIS"] = np.zeros((N_r, K))
            
            # Cache eavesdropper products if active
            if eavesdropper_active:
                G_2D = channel_matrices["H_RIS_Eaves_downlink"].squeeze(axis=1)  # Shape: (L, M)
                cached_products["BS_RIS_EAVES"] = G_2D @ Theta_Phi @ channel_matrices["H_BS_RIS"] @ W
                
                if uplink_used:
                    g_u = channel_matrices["H_RIS_Eaves_uplink"].squeeze(axis=1)  # Shape: (L, M)
                    cached_products["LU_BS_EAVES"] = g_u @ Theta_Phi @ h_u
                else:
                    cached_products["LU_BS_EAVES"] = np.zeros((num_eavesdroppers, K))
        
        # Always return cache markers (even if unchanged, for consistency)
        cached_products["_cache_updated"] = theta_phi_changed or W_changed
        cached_products["_last_theta_phi"] = Theta_Phi.copy() if theta_phi_changed or W_changed else last_theta_phi
        cached_products["_last_W"] = W.copy() if theta_phi_changed or W_changed else last_W
        
        return cached_products

