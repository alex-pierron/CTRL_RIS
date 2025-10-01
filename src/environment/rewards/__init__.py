from .basic_reward import compute_basic_reward
from .jain_fairness import jain_fairness_index, user_jain_fairness
from .minmax_reward import compute_minmax_reward
from.qos_reward import compute_qos_reward
from .minmax_smoothed_reward import compute_minmax_smoothed_reward
authorized_reward_list = ["baseline_reward", "qos_reward", "minmax_reward", "minmax_smoothed_reward"]
authorized_reward_dict = {
    "baseline_reward": compute_basic_reward,
    "qos_reward": compute_qos_reward,
    "minmax_reward": compute_minmax_reward,
    "minmax_smoothed_reward": compute_minmax_smoothed_reward,
    # Add other authorized reward functions here
}


authorized_fairness_list = ["jain_fairness"]
authorized_fairness_dict = {
    "jain_fairness": user_jain_fairness,
    # Add other authorized fairness functions here
}