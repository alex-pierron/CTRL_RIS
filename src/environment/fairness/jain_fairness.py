import numpy as np

def jain_fairness(users_current_rewards):
    
    if not users_current_rewards.size:
        return round(1/users_current_rewards.shape[0],ndigits=4)  
    sum_rewards = np.sum(users_current_rewards)
    sum_squares = np.sum(users_current_rewards**2)
    
    if sum_squares == 0:
        return round(1/users_current_rewards.shape[0],ndigits=4)  
    
    jain_index = (sum_rewards ** 2) / (len(users_current_rewards) * sum_squares)
    return jain_index