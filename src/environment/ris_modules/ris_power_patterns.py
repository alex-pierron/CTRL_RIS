"""
Power pattern computations for RIS Duplex environment.

This module handles computation of power patterns for visualization and analysis.
"""
import numpy as np
from copy import deepcopy


class PowerPatternComputer:
    """Handles power pattern computations."""
    
    @staticmethod
    def RIS_W_compute_power_pattern(W, K, N_t, lambda_h, d_h, angles, debugging=False):
        """
        Compute BS array power patterns.

        Args:
            W: Beamforming matrix of shape (N_t, K)
            K: Number of users
            N_t: Number of BS transmit antennas
            lambda_h: Wavelength parameter
            d_h: Inter-element spacing array
            angles: Array of angles for pattern computation
            debugging: Whether in debugging mode

        Returns:
            numpy.ndarray: Power patterns of shape (K, 360)
        """
        power_patterns = np.zeros((K, 360))
        for k in range(K):
            Sended_Signal_Downlink_user_k = W[:, k]
            E_total = np.zeros_like(angles, dtype=complex)
            for i, theta in enumerate(angles):
                spatial_phase = (2 * np.pi / lambda_h) * d_h[0] * np.arange(N_t) * np.sin(-theta)
                E_total[i] = np.sum(Sended_Signal_Downlink_user_k * np.exp(1j * spatial_phase))
            power_patterns[k] = np.abs(E_total) ** 2
        return power_patterns

    @staticmethod
    def RIS_downlink_compute_power_patterns(Theta_Phi, BS_RIS_channel, W, K, M, 
                                          lambda_h, d_h, angles, debugging=False):
        """
        Compute RIS downlink power patterns.

        Args:
            Theta_Phi: Product of Theta and Phi matrices
            BS_RIS_channel: BS to RIS channel matrix
            W: Beamforming matrix of shape (N_t, K)
            K: Number of users
            M: Number of RIS elements
            lambda_h: Wavelength parameter
            d_h: Inter-element spacing array
            angles: Array of angles for pattern computation
            debugging: Whether in debugging mode

        Returns:
            numpy.ndarray: Power patterns of shape (K, 360)
        """
        power_patterns = np.zeros((K, 360))
        for k in range(K):
            Reflected_Signal_Downlink_user_k = Theta_Phi @ BS_RIS_channel @ W[:, k]
            E_total = np.zeros_like(angles, dtype=complex)
            for i, theta in enumerate(angles):
                spatial_phase = (2 * np.pi / lambda_h) * d_h[1] * np.arange(M) * np.sin(-theta)
                E_total[i] = np.sum(Reflected_Signal_Downlink_user_k * np.exp(1j * spatial_phase))
            power_patterns[k] = deepcopy(np.abs(E_total) ** 2)
        return power_patterns

    @staticmethod
    def RIS_uplink_compute_power_patterns(Theta_Phi, User_RIS_channels, P_users, K, M,
                                        lambda_h, d_h, angles):
        """
        Compute RIS uplink power patterns.

        Args:
            Theta_Phi: Product of Theta and Phi matrices
            User_RIS_channels: User to RIS channel matrices of shape (K, M, 1)
            P_users: User transmit powers array of shape (K,)
            K: Number of users
            M: Number of RIS elements
            lambda_h: Wavelength parameter
            d_h: Inter-element spacing array
            angles: Array of angles for pattern computation

        Returns:
            numpy.ndarray: Power patterns of shape (K, 360)
        """
        power_patterns = np.zeros((K, 360))
        for k in range(K):
            Reflected_Signal_Uplink_user_k = (
                np.sqrt(P_users[k]) * Theta_Phi @ User_RIS_channels[k]
            )[:, 0]
            E_total = np.zeros_like(angles, dtype=complex)
            for i, theta in enumerate(angles):
                spatial_phase = (2 * np.pi / lambda_h) * d_h[1] * np.arange(M) * np.sin(-theta)
                E_total[i] = np.sum(Reflected_Signal_Uplink_user_k * np.exp(1j * spatial_phase))
            power_patterns[k] = np.abs(E_total) ** 2
        return power_patterns

