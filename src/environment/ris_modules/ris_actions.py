"""
Action processing and matrix computations for RIS Duplex environment.

This module handles:
- Processing raw actor actions into Theta and W matrices
- Computing cached matrix products (Theta_Phi, WWH, etc.)
- Updating decoding matrices
"""
import numpy as np


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
        Phi_H_Theta_H = Phi.conj().T @ Theta.conj().T
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

