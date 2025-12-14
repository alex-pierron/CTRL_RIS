"""
Metrics tracking and information retrieval for RIS Duplex environment.

This module handles:
- User and eavesdropper information tracking
- Communication rates computation
- Signal strength metrics
"""
import numpy as np
from ..physics.signal_processing import watts_to_dbm


class MetricsTracker:
    """Handles metrics tracking and information retrieval."""
    
    @staticmethod
    def get_user_communication_rates(sinr_downlink_users, sinr_uplink_users, K, uplink_used):
        """Compute downlink, uplink, and total communication rates for all users.
        
        Args:
            sinr_downlink_users: Array of downlink SINR values per user
            sinr_uplink_users: Array of uplink SINR values per user  
            K: Number of users
            uplink_used: Whether uplink is used
            
        Returns:
            dict: Dictionary with keys 'downlink', 'uplink', 'total', each containing
                  numpy arrays of shape (K,) with rates in bits/s/Hz per user.
        """
        downlink_rates = np.array([np.log2(1 + sinr_downlink_users[k]) for k in range(K)])
        if uplink_used:
            uplink_rates = np.array([np.log2(1 + sinr_uplink_users[k]) for k in range(K)])
        else:
            uplink_rates = np.zeros(K)
        total_rates = downlink_rates + uplink_rates
        
        return {
            'downlink': downlink_rates,
            'uplink': uplink_rates,
            'total': total_rates
        }
    
    @staticmethod
    def get_eavesdropper_communication_rates(SINR_E_d_k_l, SINR_E_u_k_l, K, num_eavesdroppers, 
                                            eavesdropper_active, uplink_used):
        """Compute downlink, uplink, and total communication rates for all eavesdroppers.
        
        Args:
            SINR_E_d_k_l: Function to get downlink SINR at eavesdropper l for user k
            SINR_E_u_k_l: Function to get uplink SINR at eavesdropper l for user k
            K: Number of users
            num_eavesdroppers: Number of eavesdroppers
            eavesdropper_active: Whether eavesdroppers are active
            uplink_used: Whether uplink is used
            
        Returns:
            dict: Dictionary with keys 'downlink', 'uplink', 'total', each containing
                  numpy arrays of shape (K, L) with rates in bits/s/Hz per user-eavesdropper pair.
                  Also includes 'max_per_user' with shape (K,) for maximum rates across eavesdroppers.
        """
        if not eavesdropper_active:
            return {
                'downlink': np.zeros((K, 0)),
                'uplink': np.zeros((K, 0)),
                'total': np.zeros((K, 0)),
                'max_per_user': np.zeros(K)
            }
        
        downlink_rates = np.zeros((K, num_eavesdroppers))
        uplink_rates = np.zeros((K, num_eavesdroppers))
        
        for k in range(K):
            for l in range(num_eavesdroppers):
                downlink_rates[k, l] = np.log2(1 + SINR_E_d_k_l(k, l))
                if uplink_used:
                    uplink_rates[k, l] = np.log2(1 + SINR_E_u_k_l(k, l))
        
        total_rates = downlink_rates + uplink_rates
        max_per_user = np.max(total_rates, axis=1)  # Maximum across eavesdroppers for each user
        
        return {
            'downlink': downlink_rates,
            'uplink': uplink_rates,
            'total': total_rates,
            'max_per_user': max_per_user
        }
    
    @staticmethod
    def get_user_signal_strengths(user_info, K):
        """Get minimum and maximum signal strengths for all users.
        
        Args:
            user_info: Dictionary containing user information
            K: Number of users
            
        Returns:
            dict: Dictionary with keys 'min_downlink', 'max_downlink', 'min_uplink', 'max_uplink',
                  each containing numpy arrays of shape (K,) with signal strengths in dBm.
        """
        min_downlink = np.array([
            watts_to_dbm(user_info[k]['downlink']['min_signal_watts']) 
            if user_info[k]['downlink']['min_signal_watts'] != np.inf 
            else -np.inf for k in range(K)
        ])
        max_downlink = np.array([
            watts_to_dbm(user_info[k]['downlink']['max_signal_watts']) 
            if user_info[k]['downlink']['max_signal_watts'] != -np.inf 
            else -np.inf for k in range(K)
        ])
        
        min_uplink = np.array([
            watts_to_dbm(user_info[k]['uplink']['min_signal_watts']) 
            if user_info[k]['uplink']['min_signal_watts'] != np.inf 
            else -np.inf for k in range(K)
        ])
        max_uplink = np.array([
            watts_to_dbm(user_info[k]['uplink']['max_signal_watts']) 
            if user_info[k]['uplink']['max_signal_watts'] != -np.inf 
            else -np.inf for k in range(K)
        ])
        
        return {
            'min_downlink': min_downlink,
            'max_downlink': max_downlink,
            'min_uplink': min_uplink,
            'max_uplink': max_uplink
        }
    
    @staticmethod
    def get_eavesdropper_signal_strengths(eavesdropper_info, K, num_eavesdroppers, 
                                         eavesdropper_active, uplink_used):
        """Get minimum and maximum signal strengths for all eavesdroppers.
        
        Args:
            eavesdropper_info: Dictionary containing eavesdropper information
            K: Number of users
            num_eavesdroppers: Number of eavesdroppers
            eavesdropper_active: Whether eavesdroppers are active
            uplink_used: Whether uplink is used
            
        Returns:
            dict: Dictionary with keys 'min_downlink', 'max_downlink', 'min_uplink', 'max_uplink',
                  each containing numpy arrays of shape (K, L) with signal strengths in dBm.
                  Also includes 'min_across_eaves', 'max_across_eaves' with shape (K,) for min/max across eavesdroppers.
        """
        if not eavesdropper_active:
            return {
                'min_downlink': np.zeros((K, 0)),
                'max_downlink': np.zeros((K, 0)),
                'min_uplink': np.zeros((K, 0)),
                'max_uplink': np.zeros((K, 0)),
                'min_across_eaves': np.zeros(K),
                'max_across_eaves': np.zeros(K)
            }
        
        min_downlink = np.zeros((K, num_eavesdroppers))
        max_downlink = np.zeros((K, num_eavesdroppers))
        min_uplink = np.zeros((K, num_eavesdroppers))
        max_uplink = np.zeros((K, num_eavesdroppers))
        
        for k in range(K):
            for l in range(num_eavesdroppers):
                min_dl = eavesdropper_info[l]['downlink']['min_signal_watts'][k]
                max_dl = eavesdropper_info[l]['downlink']['max_signal_watts'][k]
                min_downlink[k, l] = watts_to_dbm(min_dl) if min_dl != np.inf else -np.inf
                max_downlink[k, l] = watts_to_dbm(max_dl) if max_dl != -np.inf else -np.inf
                
                min_ul = eavesdropper_info[l]['uplink']['min_signal_watts'][k]
                max_ul = eavesdropper_info[l]['uplink']['max_signal_watts'][k]
                min_uplink[k, l] = watts_to_dbm(min_ul) if min_ul != np.inf else -np.inf
                max_uplink[k, l] = watts_to_dbm(max_ul) if max_ul != -np.inf else -np.inf
        
        # Min/max across all eavesdroppers for each user
        if uplink_used:
            combined_min = min_downlink + min_uplink
            combined_max = max_downlink + max_uplink
        else:
            combined_min = min_downlink
            combined_max = max_downlink
        
        min_across_eaves = np.min(combined_min, axis=1)
        max_across_eaves = np.max(combined_max, axis=1)
        
        return {
            'min_downlink': min_downlink,
            'max_downlink': max_downlink,
            'min_uplink': min_uplink,
            'max_uplink': max_uplink,
            'min_across_eaves': min_across_eaves,
            'max_across_eaves': max_across_eaves
        }

