import numpy as np


def R_k_minmax(k: int, L: int, SINR_D_k, SINR_U_k, SINR_E_d_k_l, SINR_E_u_k_l,
             eavesdropper_active: bool,):
    # Calculate the rates
    downlink_reward_k = np.log2(1 + SINR_D_k[k])
    uplink_reward_k = np.log2(1 + SINR_U_k[k])

    if eavesdropper_active:
        R_E_B_k_all_l = np.array([np.log2(1 + SINR_E_d_k_l(k, l)) for l in range(L)]) 
        R_E_S_k_all_l = np.array([ np.log2(1 + SINR_E_u_k_l(k, l)) for l in range(L)])
        downlink_reward_k = max(0, downlink_reward_k - np.max(R_E_B_k_all_l))
        uplink_reward_k = max(0, uplink_reward_k - np.max(R_E_S_k_all_l))

    return downlink_reward_k, uplink_reward_k



def compute_minmax_reward(K: int, L: int, SINR_Downlink_k, SINR_Uplink_k, SINR_E_d_k_l, SINR_E_u_k_l,eavesdropper_active, p: float = 1):
    
    def compute_reward(k):
        R_B_k, R_S_k = R_k_minmax(k, L, SINR_Downlink_k, SINR_Uplink_k, SINR_E_d_k_l, SINR_E_u_k_l, eavesdropper_active)
        return R_B_k, R_S_k

    # Compute rewards for all users
    users_rewards = np.array([compute_reward(k) for k in range(K)])

    # Separate R_B_k and R_S_k
    R_B_k_values = users_rewards[:, 0]
    R_S_k_values = users_rewards[:, 1]

    # Calculate the global reward
    global_reward = K * (np.min(R_B_k_values) +  np.min(R_S_k_values))
    # global_reward = np.sum(R_B_k_values) / ((1+np.std(R_B_k_values))**p)  +np.sum(R_S_k_values) / ( (1+np.std(R_S_k_values)) **p)

    return users_rewards, global_reward