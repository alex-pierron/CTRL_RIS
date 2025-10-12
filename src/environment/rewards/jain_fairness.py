import numpy as np

def jain_fairness_index(reward_list):
    reward_list = np.array(reward_list)
    return np.sum(reward_list)**2 / (reward_list.shape[0] * np.sum(reward_list**2))



def user_jain_fairness(users_current_rewards):
    
    if not users_current_rewards.size:
        return 0.0
    sum_rewards = np.sum(users_current_rewards)
    sum_squares = np.sum(users_current_rewards**2)
    
    if sum_squares == 0:
        return 0
    
    jain_index = (sum_rewards ** 2) / (len(users_current_rewards) * sum_squares)
    return jain_index