import numpy as np
from copy import deepcopy
import os 
import sys

def independant_R_sec_k(k: int, L: int, SINR_B_k, SINR_S_k,
                        SINR_E_d_k_l, SINR_E_u_k_l,
            eavesdropper_active: bool, verbose: bool = False):
    downlink_reward_k = np.log2(1 + SINR_B_k[k])
    uplink_reward_k = np.log2(1 + SINR_S_k[k])

    if eavesdropper_active:
        eavesdroppers_downlink_rewards = np.array([np.log2(1 + SINR_E_d_k_l(k, l)) for l in range(L)])
        eavesdroppers_uplink_rewards = np.array([np.log2(1 + SINR_E_u_k_l(k, l)) for l in range(L)])
        eavesdroppers_rewards_all =  eavesdroppers_downlink_rewards + eavesdroppers_uplink_rewards 
    else:
        eavesdroppers_rewards_all = 0

    potential_reward = downlink_reward_k + uplink_reward_k - np.max(eavesdroppers_rewards_all)
    actual_reward = max(potential_reward, 0)

    if verbose and eavesdropper_active:
        user_reward_detail = {
            f"Final reward": np.float16(actual_reward),
            f"Downlink reward": np.float16(downlink_reward_k),
            f"Uplink reward": np.float16(uplink_reward_k),
            f"Eavesdropping reward": np.float16(np.max(eavesdroppers_rewards_all)),
            f"Eavesdropping_Downlink_reward": np.float16(np.max(eavesdroppers_downlink_rewards)),
            f"Eavesdropping_Uplink_reward": np.float16(np.max(eavesdroppers_uplink_rewards)),

        }
        return actual_reward, user_reward_detail
    
    elif verbose:
        user_reward_detail = {
            f"Final reward": np.float16(actual_reward),
            f"Downlink reward": np.float16(downlink_reward_k),
            f"Uplink reward": np.float16(uplink_reward_k),
        }
        return actual_reward, user_reward_detail

    return actual_reward


def compute_backdoor_harcoded_reward(K: int, L: int, SINR_B_k, SINR_S_k,
                        SINR_E_d_k_l, SINR_E_u_k_l,
                        eavesdropper_active: bool, verbose: bool = False):
    user_reward_details = {}

    def compute_reward(k):
        result = independant_R_sec_k(k, L, SINR_B_k, SINR_S_k,
                                     SINR_E_d_k_l, SINR_E_u_k_l,
                                     eavesdropper_active = eavesdropper_active,
                                     verbose = verbose)
        #if verbose:
        actual_reward, user_reward_detail = result
        user_reward_details[k] = user_reward_detail
        """else:
            actual_reward = result"""
        return actual_reward

    users_current_rewards = np.array([compute_reward(k) for k in range(K)])
    total_basic_reward = np.sum(users_current_rewards) 

    if verbose:
        return users_current_rewards, total_basic_reward, user_reward_details
    
    return users_current_rewards, total_basic_reward
