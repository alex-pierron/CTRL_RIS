import numpy as np

def R_k_QoS(k: int, L: int, mu_1: float,mu_2: float,
            target_secrecy_rate : float, target_data_rate : float,
            SINR_B_k, SINR_S_k,
            SINR_E_d_k_l, SINR_E_u_k_l,
            eavesdropper_active: bool):
    # Calculate the rates
    R_B_k = np.log2(1 + SINR_B_k[k])
    R_S_k = np.log2(1 + SINR_S_k[k])

    # Calculate the eavesdropper rates if active
    if eavesdropper_active:
        R_E_k_all_l = np.array([np.log2(1 + SINR_E_d_k_l(k, l)) + np.log2(1 + SINR_E_u_k_l(k, l)) for l in range(L)])
        worst_intercept = np.max(R_E_k_all_l)
    else:
        worst_intercept = 0
    R_k = R_B_k + R_S_k
    R_sec_k = R_k - worst_intercept
    if eavesdropper_active:
        return R_sec_k - mu_1 * (R_sec_k < target_secrecy_rate) - mu_2 * (R_k < target_data_rate)
    else:
        return R_sec_k - mu_1 * (R_k < target_data_rate)

        
def compute_qos_reward(K:int, L:int,
                        mu_1: float, mu_2: float, target_secrecy_rate:float, target_data_rate:float,
                        SINR_B_k, SINR_S_k, SINR_E_d_k_l, SINR_E_u_k_l,
                        eavesdropper_active):
    def compute_reward(k):
        result = R_k_QoS(k, L,mu_1, mu_2, target_secrecy_rate, target_data_rate,
                          SINR_B_k, SINR_S_k, SINR_E_d_k_l, SINR_E_u_k_l,
                          eavesdropper_active)
        return result
    users_current_rewards = np.array([compute_reward(k) for k in range(K)])
    global_reward = np.sum(users_current_rewards)
    return users_current_rewards, global_reward