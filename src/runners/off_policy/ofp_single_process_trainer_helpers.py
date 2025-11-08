"""
Helper functions for off-policy single-process trainer.

This module contains utility functions for logging and message formatting
used in the off-policy training loop.
"""
import numpy as np


def log_position_message(logger, episode, users_position, eavesdroppers_positions, 
                         buffer_filled, buffer_number_of_required_episode, using_eavesdropper):
    """
    Create and log position message for episode start.
    
    Args:
        logger: Logger instance for writing messages
        episode: Current episode number
        users_position: Array of user positions
        eavesdroppers_positions: Array of eavesdropper positions (can be None)
        buffer_filled: Whether the replay buffer has been filled
        buffer_number_of_required_episode: Number of episodes required to fill buffer
        using_eavesdropper: Whether eavesdroppers are present in the environment
    
    Returns:
        str: The formatted position message
    """
    if buffer_filled:
        episode_num = episode - buffer_number_of_required_episode
        prefix = f"\nCommencing training episode {episode_num} with"
    else:
        episode_num = episode
        prefix = f"\n Initializing positions for replay buffer episode {episode_num} with"
    
    if using_eavesdropper:
        position_message = (
            f"{prefix} users and eavesdroppers positions:\n"
            f"   !~ Users Positions: {list(users_position)} \n"
            f"   !~ Eavesdroppers Positions: {list(eavesdroppers_positions)} \n"
        )
    else:
        position_message = (
            f"{prefix} users positions:\n"
            f"   !~ Users Positions: {list(users_position)} \n"
        )
    
    logger.verbose(position_message)
    return position_message


def create_episode_summary_messages(episode, episode_offset, optim_steps, users_position, 
                                    eavesdroppers_positions, avg_reward, max_reward, 
                                    avg_fairness, best_fairness, avg_actor_loss, avg_critic_loss,
                                    basic_reward_episode, instant_user_rewards, 
                                    additional_information_best_case, using_eavesdropper,
                                    avg_eavesdropper_reward=None, best_eaves_reward=None):
    """
    Create formatted console and log messages for episode summary.
    
    Args:
        episode: Current episode number
        episode_offset: Episode number offset (for display purposes)
        optim_steps: Number of optimization steps performed
        users_position: Array of user positions
        eavesdroppers_positions: Array of eavesdropper positions (can be None)
        avg_reward: Average reward for the episode
        max_reward: Maximum reward reached in the episode
        avg_fairness: Average fairness index for the episode
        best_fairness: Fairness index at the best reward step
        avg_actor_loss: Average actor loss for the episode
        avg_critic_loss: Average critic loss for the episode
        basic_reward_episode: Array of baseline rewards for the episode
        instant_user_rewards: Array of instant user rewards
        additional_information_best_case: Additional information for best case
        using_eavesdropper: Whether eavesdroppers are present
        avg_eavesdropper_reward: Average eavesdropper reward (optional)
        best_eaves_reward: Eavesdropper reward at best step (optional)
    
    Returns:
        tuple: (console_message, log_message) formatted strings
    """
    best_baseline = basic_reward_episode[np.argmax(instant_user_rewards)] if len(instant_user_rewards) > 0 else 0.0
    
    if using_eavesdropper:
        console_msg = (
            f"\n\n"
            f"╔══════════════════════════════════════════════════════════════════════════════════════════════════╗\n"
            f"║ 🎯 TRAINING EPISODE #{episode_offset:3d} │ 🧠 Actor Optimizations: {optim_steps:4d} ║\n"
            f"╠══════════════════════════════════════════════════════════════════════════════════════════════════╣\n"
            f"║ 📍 POSITIONING:\n"
            f"║    User Equipment Positions: {users_position}\n"
            f"║ ──────────────────────────────────────────────────────────────────────────────────────────────── ║\n"
            f"║ 🏆 REWARDS:\n"
            f"║    • Average Reward: {avg_reward:8.4f} │ Max Instant: {max_reward:8.4f}\n"
            f"║    • Baseline Reward Avg: {np.mean(basic_reward_episode):8.4f} │ Best Baseline: {best_baseline:8.4f}\n"
            f"║    • Detailed Baseline Reward: {additional_information_best_case}\n"
            f"║ ──────────────────────────────────────────────────────────────────────────────────────────────── ║\n"
            f"║ ⚖️  FAIRNESS:\n"
            f"║    • Average Fairness: {avg_fairness:8.4f} │ Best Reward Fairness: {best_fairness:8.4f}\n"
            f"║ ──────────────────────────────────────────────────────────────────────────────────────────────── ║\n"
            f"║ 🕵️  EAVESDROPPERS:\n"
            f"║    • Average Signal Catched: {avg_eavesdropper_reward:8.4f} | Signal Catched For The Best Case {best_eaves_reward:8.4f}\n"
            f"║ ──────────────────────────────────────────────────────────────────────────────────────────────── ║\n"
            f"║ 📊 PERFORMANCE:\n"
            f"║    • Actor Loss: {avg_actor_loss:8.4f} │ Critic Loss: {avg_critic_loss:8.4f}\n"
            f"╚══════════════════════════════════════════════════════════════════════════════════════════════════╝\n"
        )
        
        log_msg = (
            f"\n+====================================================================================================+\n"
            f"| [TRAIN] EPISODE #{episode_offset:3d} | Actor Optimizations: {optim_steps:4d} |\n"
            f"+====================================================================================================+\n"
            f"| POSITIONING:\n"
            f"|    User Equipment Positions: {users_position}\n"
            f"|    Eavesdroppers Positions: {eavesdroppers_positions}\n"
            f"| ---------------------------------------------------------------------------------------------------- |\n"
            f"| REWARDS:\n"
            f"|    * Average Reward: {avg_reward:8.4f} | Max Instant: {max_reward:8.4f}\n"
            f"|    * Baseline Reward Avg: {np.mean(basic_reward_episode):8.4f} | Best Baseline: {best_baseline:8.4f}\n"
            f"|    * Detailed Baseline Reward: {additional_information_best_case}\n"
            f"| ---------------------------------------------------------------------------------------------------- |\n"
            f"| FAIRNESS:\n"
            f"|    * Average Fairness: {avg_fairness:8.4f} | Best Reward Fairness: {best_fairness:8.4f}\n"
            f"| ---------------------------------------------------------------------------------------------------- |\n"
            f"| EAVESDROPPERS:\n"
            f"|    * Average Signal Catched: {avg_eavesdropper_reward:8.4f} | Signal Catched For The Best Case {best_eaves_reward:8.4f}\n"
            f"| ---------------------------------------------------------------------------------------------------- |\n"
            f"| PERFORMANCE:\n"
            f"|    * Actor Loss: {avg_actor_loss:8.4f} | Critic Loss: {avg_critic_loss:8.4f}\n"
            f"+====================================================================================================+\n"
        )
    else:
        console_msg = (
            f"\n\n"
            f"╔══════════════════════════════════════════════════════════════════════════════════════════════════╗\n"
            f"║ 🎯 TRAINING EPISODE #{episode_offset:3d} │ 🧠 Actor Optimizations: {optim_steps:4d} ║\n"
            f"╠══════════════════════════════════════════════════════════════════════════════════════════════════╣\n"
            f"║ 📍 POSITIONING:\n"
            f"║    User Equipment Positions: {users_position}\n"
            f"║ ──────────────────────────────────────────────────────────────────────────────────────────────── ║\n"
            f"║ 🏆 REWARDS:\n"
            f"║    • Average Reward: {avg_reward:8.4f} │ Max Instant: {max_reward:8.4f}\n"
            f"║    • Baseline Reward Avg: {np.mean(basic_reward_episode):8.4f} │ Best Baseline: {best_baseline:8.4f}\n"
            f"║    • Detailed Baseline Reward: {additional_information_best_case}\n"
            f"║ ──────────────────────────────────────────────────────────────────────────────────────────────── ║\n"
            f"║ ⚖️  FAIRNESS:\n"
            f"║    • Average Fairness: {avg_fairness:8.4f} │ Best Reward Fairness: {best_fairness:8.4f}\n"
            f"║ ──────────────────────────────────────────────────────────────────────────────────────────────── ║\n"
            f"║ 📊 PERFORMANCE:\n"
            f"║    • Actor Loss: {avg_actor_loss:8.4f} │ Critic Loss: {avg_critic_loss:8.4f}\n"
            f"╚══════════════════════════════════════════════════════════════════════════════════════════════════╝\n"
        )
        
        log_msg = (
            f"\n+====================================================================================================+\n"
            f"| [TRAIN] EPISODE #{episode_offset:3d} | Actor Optimizations: {optim_steps:4d} |\n"
            f"+====================================================================================================+\n"
            f"| POSITIONING:\n"
            f"|    User Equipment Positions: {users_position}\n"
            f"| ---------------------------------------------------------------------------------------------------- |\n"
            f"| REWARDS:\n"
            f"|    * Average Reward: {avg_reward:8.4f} | Max Instant: {max_reward:8.4f}\n"
            f"|    * Baseline Reward Avg: {np.mean(basic_reward_episode):8.4f} | Best Baseline: {best_baseline:8.4f}\n"
            f"|    * Detailed Baseline Reward: {additional_information_best_case}\n"
            f"| ---------------------------------------------------------------------------------------------------- |\n"
            f"| FAIRNESS:\n"
            f"|    * Average Fairness: {avg_fairness:8.4f} | Best Reward Fairness: {best_fairness:8.4f}\n"
            f"| ---------------------------------------------------------------------------------------------------- |\n"
            f"| PERFORMANCE:\n"
            f"|    * Actor Loss: {avg_actor_loss:8.4f} | Critic Loss: {avg_critic_loss:8.4f}\n"
            f"+====================================================================================================+\n"
        )
    
    return console_msg, log_msg

