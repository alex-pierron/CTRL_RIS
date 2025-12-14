"""
Multiprocess training loop utilities for RIS Duplex RL experiments.

This module provides comprehensive training infrastructure for reinforcement learning
agents in RIS (Reconfigurable Intelligent Surface) environments. It supports:

- setup_logger: Configures file and console logging with custom VERBOSE level
- ofp_multiprocess_trainer: Main training loop coordinating vectorized environments

Features:
- Multi-environment parallel training with curriculum learning
- Comprehensive reward tracking (decisive and informative rewards)
- Real-time evaluation with separate evaluation environments
- Advanced logging with TensorBoard integration
- Fairness metrics tracking using Jain's Fairness Index

Comment Legend:
- TODO: Future improvements or refactoring opportunities
- NOTE: Important behavior or design decisions
- !: Critical runtime observations
- ?: Design questions or assumptions requiring review
"""
import logging
import os
import sys
import time
import torch
from tqdm import tqdm
import numpy as np
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter
from src.environment.tools import parse_args, parse_config, write_line_to_file, SituationRenderer,TaskManager
from src.environment.multiprocessing import pickable_to_dict


# Define custom VERBOSE log level (between DEBUG=10 and INFO=20)
VERBOSE_LEVEL = 15
logging.addLevelName(VERBOSE_LEVEL, "VERBOSE")

# Custom verbose logging method for detailed training information
def verbose(self, message, *args, **kwargs):
    """Log a message at VERBOSE level if enabled."""
    if self.isEnabledFor(VERBOSE_LEVEL):
        self._log(VERBOSE_LEVEL, message, args, **kwargs)

# Attach the verbose method to the Logger class for easy access
logging.Logger.verbose = verbose


def setup_logger(log_dir, log_filename='training.log'):
    """
    Configure a logger with file and console handlers for training sessions.
    
    Creates a dedicated logger that writes all messages to a file while only
    displaying warnings and errors in the console, keeping the terminal output clean.
    
    Args:
        log_dir (str): Directory path where log files will be stored
        log_filename (str, optional): Name of the log file. Defaults to 'training.log'
        
    Returns:
        logging.Logger: Configured logger instance with file and console handlers
        
    Note:
        The logger uses a custom VERBOSE level (15) for detailed training information
        that is written to file but not displayed in console.
    """
    # Create named logger to prevent duplicate logs in larger applications
    logger = logging.getLogger('TrainingLogger')
    logger.setLevel(logging.DEBUG)  # Capture all log levels

    # Prevent propagation to root logger to avoid duplicate messages
    logger.propagate = False

    # Clear existing handlers to prevent duplication on multiple calls
    if logger.hasHandlers():
        logger.handlers.clear()

    # File handler: captures all log levels for comprehensive logging
    file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
    file_handler.setLevel(logging.DEBUG)

    # Console handler: only shows warnings and errors to keep terminal clean
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)  # Suppress INFO level messages

    # Configure consistent formatting for both handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Attach handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def ofp_multiprocess_trainer(training_envs, network, training_config, log_dir, writer,
                         action_noise_activated=False,
                         batch_instead_of_buff=False,
                         eval_env=None,
                         use_rendering=False):
    """
    Execute multiprocess training loop for RIS Duplex RL agents.
    
    This function orchestrates the complete training pipeline including:
    - Vectorized environment interactions across multiple processes
    - Curriculum learning with adaptive difficulty scheduling
    - Comprehensive reward and fairness metric tracking
    - Periodic evaluation with separate evaluation environments
    - Advanced logging and TensorBoard integration
    
    Args:
        training_envs: Vectorized training environments (DummyVecEnv instance)
        network: Neural network agent implementing the RL algorithm
        training_config (dict): Training hyperparameters and configuration
        log_dir (str): Directory path for saving logs and model checkpoints
        writer: TensorBoard SummaryWriter for experiment tracking
        action_noise_activated (bool, optional): Enable action noise for exploration.
            Defaults to False.
        batch_instead_of_buff (bool, optional): Use batch size instead of buffer size
            for training decisions. Defaults to False.
        eval_env: Evaluation environment for periodic performance assessment.
            Defaults to None (no evaluation).
        use_rendering (bool, optional): Enable environment rendering during training.
            Defaults to False.
            
    Note:
        The function supports both decisive and informative reward functions,
        tracks Jain's Fairness Index for user fairness assessment, and implements
        curriculum learning for progressive difficulty adjustment.
    """
    # Initialize dedicated logger for this training session
    # VERBOSE level messages are written to file but not displayed in console
    logger = setup_logger(log_dir)

    # Extract environment configuration
    env_config = training_envs.env_config

    # Parse training configuration parameters with sensible defaults
    debugging = training_config.get("debugging", False)
    num_episode = training_config.get("number_episodes", 15)
    max_num_step_per_episode = training_config.get("max_steps_per_episode", 20000)
    batch_size = training_config.get("batch_size", 128)
    frequency_information = training_config.get("frequency_information", 500)
    
    # Evaluation configuration
    conduct_eval = eval_env is not None
    episode_per_eval = training_config.get("episode_per_eval_env", 1) if conduct_eval else None
    eval_period = training_config.get("eval_period", 1) if conduct_eval else None

    # Curriculum learning enables progressive difficulty adjustment
    curriculum_learning = training_config.get("Curriculum_Learning", True)

    # Extract additional configuration parameters
    num_users = env_config
    saving_frequency = training_config.get("network_save_checkpoint_frequency", 100)
    n_eval_rollout_threads = training_config.get("n_eval_rollout_threads", 2)
    
    # Determine buffer size for training decisions
    buffer_size = deepcopy(batch_size) if batch_instead_of_buff else network.replay_buffer.buffer_size
    
    # Extract reward function configurations
    decisive_reward_functions = env_config.get("decisive_reward_functions", ["basic_reward"])
    informative_reward_functions = env_config.get("informative_reward_functions", [])

    # Initialize tracking arrays for reward functions across episodes
    decisive_rewards_average_episode = {
        function_name: np.zeros(num_episode) 
        for function_name in decisive_reward_functions
    }
    informative_rewards_average_episode = {
        function_name: np.zeros(num_episode) 
        for function_name in informative_reward_functions
    }

    # Extract environment metadata
    n_rollout_envs = training_envs.nenvs
    num_users = training_envs.num_users
    using_eavesdropper = (training_envs.num_eavesdroppers > 0)

    # Initialize curriculum learning system if enabled
    if curriculum_learning:
        # TaskManager handles progressive difficulty scheduling across environments
        grid_limit = env_config.get("user_spawn_limits")
        downlink_activated = env_config.get("BS_max_power") > 0
        uplink_activated = env_config.get("user_transmit_power") > 0
        
        Task_Manager = TaskManager(
            num_users,
            num_steps_per_episode=max_num_step_per_episode,
            user_limits=grid_limit,
            RIS_position=env_config.get("RIS_position"),
            downlink_uplink_eavesdropper_bools=[downlink_activated, uplink_activated, using_eavesdropper],
            thresholds=training_config.get("Curriculum_Learning_Thresholds", [0.25, 0.25, 0.25]),
            random_seed=training_config.get("Task_Manager_random_seed", 126),
            num_environments=n_rollout_envs
        )
    
    # Initialize training state variables
    index_episode_buffer_filled = 0
    num_eval_periode = 0
    eval_current_step = 0
    buffer_filled = False
    best_average_reward = 0
    optim_steps = 0
    optim_steps_actor = 0
    optim_steps_critic = 0
    
    # Eavesdropper positions are only relevant when eavesdroppers are present
    if not using_eavesdropper:
        eavesdroppers_positions = None


    # Initialize tracking arrays for performance metrics
    average_reward_per_env = np.zeros((num_episode, n_rollout_envs))
    average_best_reward_per_env = np.zeros((num_episode, n_rollout_envs))
    avg_user_fairness_all_episode_general = np.zeros(num_episode)

    # Configure replay buffer filling progress tracking
    # TODO: Optimize buffer progress bar management for better performance
    buffer_bar_finished = False
    buffer_number_of_required_episode = buffer_size // (n_rollout_envs * max_num_step_per_episode)
    buffer_smaller_than_one_episode = (buffer_size // (n_rollout_envs * max_num_step_per_episode) < 1)
    length_episode_rb_matching = ((buffer_size % (n_rollout_envs * max_num_step_per_episode)) == 0)
    
    # Ensure minimum buffer filling episodes
    if buffer_number_of_required_episode == 0:
        buffer_number_of_required_episode = 1
    
    # Initialize progress bar for replay buffer warmup phase
    filling_buffer_progress_bar = (
        tqdm(total=buffer_number_of_required_episode, 
             desc="[BUFFER] FILLING REPLAY BUFFER", 
             position=0, 
             bar_format="{l_bar}{bar:50}{r_bar}{bar:-10b}",
             ncols=140,
             colour='blue',
             ascii="▏▎▍▌▋▊▉█",
             leave=True) 
        if not batch_instead_of_buff else None
    )
    

    # ========================================================================
    # MAIN TRAINING LOOP
    # ========================================================================
    for episode in tqdm(range(num_episode), 
                       desc="[TRAIN] TRAINING", 
                       position=2, 
                       bar_format="{l_bar}{bar:50}{r_bar}{bar:-10b}",
                       ncols=140,
                       colour='magenta',
                       ascii="▏▎▍▌▋▊▉█"):
        
        # Reset episode-level optimization counters
        optim_steps_actor_ep = 0
        optim_steps_critic_ep = 0

        # Apply curriculum learning difficulty configuration if enabled
        if curriculum_learning:
            difficulty_config = Task_Manager.generate_episode_configs()
            training_envs.reset(difficulty_config)
        else:
            training_envs.reset()

        # Record episode start time for performance tracking
        start_episode_time = time.time()
        
        # Get initial user positions for logging
        user_positions = training_envs.get_users_positions()
        if using_eavesdropper:
            eavesdroppers_positions =  training_envs.get_eavesdroppers_positions()
        # Initialize episode tracking arrays
        instant_user_rewards = np.zeros((n_rollout_envs, max_num_step_per_episode))
        basic_reward_episode = np.zeros((max_num_step_per_episode, n_rollout_envs))
        
        # Initialize eavesdropper reward tracking if applicable
        instant_eavesdropper_rewards = np.zeros((n_rollout_envs, max_num_step_per_episode))
        avg_eavesdropper_reward = 0
        
        # Initialize fairness tracking using Jain's Fairness Index
        instant_user_jain_fairness = np.zeros((max_num_step_per_episode, n_rollout_envs))
        
        # Initialize reward function tracking arrays
        decisive_rewards_current_episode = {
            function_name: np.zeros((n_rollout_envs, max_num_step_per_episode + 1)) 
            for function_name in decisive_reward_functions
        }
        informative_rewards_current_episode = {
            function_name: np.zeros((n_rollout_envs, max_num_step_per_episode + 1)) 
            for function_name in informative_reward_functions
        }

        # Initialize episode performance metrics
        avg_actor_loss = 0
        avg_critic_loss = 0
        max_instant_reward = 0
        total_reward = np.zeros(n_rollout_envs)
        step_time_list = []

        # ====================================================================
        # EPISODE STEP LOOP
        # ====================================================================
        for num_step in range(max_num_step_per_episode):
            current_step = episode * max_num_step_per_episode + num_step

            # Get initial states for first step, otherwise use next states from previous step
            if num_step == 0:
                states = training_envs.get_states()
                states = np.squeeze(states, axis=1)
            else:
                states = next_states

            # Action selection with optional noise for exploration
            if action_noise_activated:
                # Apply per-environment exploration noise for better exploration
                # TODO: Optimize this loop using vectorized operations or vmap
                selected_actions = np.zeros(shape=(n_rollout_envs, training_envs.action_dim))
                selected_noised_actions = torch.zeros(size=(n_rollout_envs, training_envs.action_dim))
                
                for i in range(n_rollout_envs):
                    selected_actions[i], selected_noised_actions[i] = network.select_noised_action(states[i])
                
                states, _, rewards, next_states = training_envs.step(states, selected_noised_actions)
            else:
                # Use deterministic actions across all vectorized environments
                selected_actions = torch.stack([network.select_action(states[i]) for i in range(n_rollout_envs)])
                states, selected_actions, rewards, next_states = training_envs.step(states, selected_actions)

            # Process and store reward information
            instant_user_rewards[:, num_step] = rewards
            total_reward += rewards
            rewards = rewards.reshape((n_rollout_envs, 1))
            max_instant_reward = max(max_instant_reward, rewards.max().item())

            # Extract basic reward for this step
            basic_reward_episode[num_step, :] = training_envs.get_basic_reward().reshape((n_rollout_envs))

            # Extract and process decisive and informative rewards
            decisive_rewards = pickable_to_dict(training_envs.get_decisive_rewards())
            informative_rewards = pickable_to_dict(training_envs.get_informative_rewards())

            # Store decisive reward function values for this step
            for function_name in decisive_reward_functions:
                decisive_rewards_current_episode[function_name][:, num_step] = [
                    decisive_rewards[env_idx][function_name]['total_reward'] 
                    for env_idx in decisive_rewards
                ]

            # Store informative reward function values for this step
            for function_name in informative_reward_functions:
                informative_rewards_current_episode[function_name][:, num_step] = [
                    informative_rewards[env_idx][function_name]['total_reward'] 
                    for env_idx in informative_rewards
                ]

            # Ensure rewards are properly shaped for storage
            rewards = rewards.reshape((n_rollout_envs, 1))

            # Store transitions for all environments in a single batch operation
            network.store_transition(states, selected_actions, rewards, next_states, batch_size=n_rollout_envs)

            # Calculate and store Jain's Fairness Index for this step
            current_fairness = np.round(training_envs.get_jain_fairness(), decimals=4).squeeze()
            instant_user_jain_fairness[num_step] = current_fairness

            # Track eavesdropper rewards if eavesdroppers are present
            if using_eavesdropper:
                eavesdropper_rewards = np.array(training_envs.get_eavesdroppers_rewards()).squeeze()
                instant_eavesdropper_rewards[:, num_step] = eavesdropper_rewards

            # Set actor network to training mode
            network.actor.train()
            
            if not buffer_filled:
                    index_episode_buffer_filled = episode

            # Perform network training if replay buffer is sufficiently filled
            if network.replay_buffer.size >= buffer_size:
                if not buffer_filled:
                    index_episode_buffer_filled = episode
                    buffer_filled = True

                # Execute network training step
                training_time_1 = time.time()
                actor_loss, critic_loss, _, updated_actor, updated_critic = network.training(batch_size=batch_size)
                training_time_2 = time.time()

                # Track optimization steps and timing
                if updated_actor:
                    avg_actor_loss += actor_loss
                    optim_steps_actor_ep += 1
                    step_time_list.append(training_time_2 - training_time_1)

                if updated_critic:
                    avg_critic_loss += critic_loss
                    optim_steps_critic_ep += 1

                # Periodic logging of training metrics and performance statistics
                if (current_step + 1) % frequency_information == 0 and (num_step + 1) % frequency_information == 0:
                    
                    # Log replay buffer statistics
                    writer.add_scalar("Replay Buffer/Average reward stored", 
                                     np.mean(network.replay_buffer.reward_buffer), current_step)

                    # Calculate and log current average losses
                    if optim_steps_actor_ep:
                        current_avg_actor_loss = avg_actor_loss / optim_steps_actor_ep
                    else:
                        current_avg_actor_loss = 0
                    if optim_steps_critic_ep:
                        current_avg_critic_loss = avg_critic_loss / optim_steps_critic_ep
                    else:
                        current_avg_critic_loss = 0

                    writer.add_scalar("Actor Loss/Current average actor loss", current_avg_actor_loss, current_step)
                    writer.add_scalar("Critic Loss/Current average critic loss", current_avg_critic_loss, current_step)

                    # Calculate local metrics for the current information window
                    local_average_reward = np.mean(instant_user_rewards[:, num_step + 1 - frequency_information: num_step])
                    local_average_reward_per_env = np.mean(instant_user_rewards[:, num_step + 1 - frequency_information: num_step], axis=1)

                    # Calculate local fairness metrics
                    user_fairness_mean_local = np.round(
                        np.mean(instant_user_jain_fairness[num_step + 1 - frequency_information: num_step]), 
                        decimals=3
                    )
                    user_fairness_mean_local_per_env = np.round(
                        np.mean(instant_user_jain_fairness[num_step + 1 - frequency_information: num_step], axis=0), 
                        decimals=3
                    )

                    # Log reward metrics
                    writer.add_scalar("Rewards/Average local reward", local_average_reward, current_step)
                    writer.add_scalar("Rewards/Average global reward", np.mean(total_reward) / (num_step + 1), current_step)

                    # Log instant loss values
                    writer.add_scalar("Actor Loss/Instant actor loss", actor_loss, current_step)
                    writer.add_scalar("Critic Loss/Instant critic loss", critic_loss, current_step)

                    # Log fairness metrics
                    writer.add_scalar("Fairness/Local average user Fairness", user_fairness_mean_local, current_step)

                    # Log eavesdropper metrics if applicable
                    if using_eavesdropper:
                        local_average_eavesdropper_reward = np.mean(
                            instant_eavesdropper_rewards[:, num_step + 1 - frequency_information: num_step]
                        )
                        writer.add_scalar("Eavesdropper/Local average reward", local_average_eavesdropper_reward, current_step)
                        writer.add_histogram("Eavesdropper/Instant reward", 
                                            instant_eavesdropper_rewards[:, num_step + 1 - frequency_information: num_step], 
                                            current_step)
                    
                    # Create detailed training progress message for console (with emojis)
                    console_message = (
                        f"\n"
                        f"╔════════════════════════════════════════════════════════════════════════════════╗\n"
                        f"║ 🎯 TRAINING EPISODE {episode - index_episode_buffer_filled:3d} │ STEP {num_step + 1:5d} │ ⏱️  {np.mean(step_time_list):.4f}s/step ║\n"
                        f"╠════════════════════════════════════════════════════════════════════════════════╣\n"
                        f"║ 📍 POSITIONING: UEs at {user_positions}\n"
                        f"║ ──────────────────────────────────────────────────────────────────────────────── ║\n"
                        f"║ 🏆 REWARDS:\n"
                        f"║    • Local Avg: {np.float16(local_average_reward):8.4f} │ Max Instant: {max_instant_reward:8.4f}\n"
                        f"║    • Max per Env: {np.max(instant_user_rewards, axis=1)}\n"
                        f"║    • Avg per Env: {local_average_reward_per_env}\n"
                        f"║ ──────────────────────────────────────────────────────────────────────────────── ║\n"
                        f"║ ⚖️  FAIRNESS:\n"
                        f"║    • Global Fairness: {user_fairness_mean_local:8.4f}\n"
                        f"║    • Per Environment: {user_fairness_mean_local_per_env}\n"
                        f"║ ──────────────────────────────────────────────────────────────────────────────── ║\n"
                        f"║ 📉 LOSSES: Actor: {current_avg_actor_loss:8.4f} │ Critic: {current_avg_critic_loss:8.4f}\n"
                        f"╚════════════════════════════════════════════════════════════════════════════════╝\n"
                    )
                    
                    # Create ASCII version for file logging
                    log_message = (
                        f"\n"
                        f"+================================================================================+\n"
                        f"| [TRAIN] EPISODE {episode - index_episode_buffer_filled:3d} | STEP {num_step + 1:5d} | TIME: {np.mean(step_time_list):.4f}s/step |\n"
                        f"+================================================================================+\n"
                        f"| POSITIONING: UEs at {user_positions}\n"
                        f"| -------------------------------------------------------------------------------- |\n"
                        f"| REWARDS:\n"
                        f"|    * Local Avg: {np.float16(local_average_reward):8.4f} | Max Instant: {max_instant_reward:8.4f}\n"
                        f"|    * Max per Env: {np.max(instant_user_rewards, axis=1)}\n"
                        f"|    * Avg per Env: {local_average_reward_per_env}\n"
                        f"| -------------------------------------------------------------------------------- |\n"
                        f"| FAIRNESS:\n"
                        f"|    * Global Fairness: {user_fairness_mean_local:8.4f}\n"
                        f"|    * Per Environment: {user_fairness_mean_local_per_env}\n"
                        f"| -------------------------------------------------------------------------------- |\n"
                        f"| LOSSES: Actor: {current_avg_actor_loss:8.4f} | Critic: {current_avg_critic_loss:8.4f}\n"
                        f"+================================================================================+\n"
                    )
                    
                    # Display beautiful console message
                    tqdm.write(console_message)
                    
                    # Log ASCII version to file
                    logger.verbose(log_message)

        # Update global optimization step counters
        optim_steps_actor += optim_steps_actor_ep
        optim_steps_critic += optim_steps_critic_ep

        # Update replay buffer filling progress bar
        # TODO: Optimize buffer progress bar management for better performance
        if batch_instead_of_buff:
            pass  # No progress bar needed when using batch size
        elif buffer_filled or buffer_bar_finished:
            if buffer_smaller_than_one_episode:
                filling_buffer_progress_bar.update(1)
                if filling_buffer_progress_bar.n == buffer_number_of_required_episode:
                    buffer_bar_finished = True
            elif length_episode_rb_matching and filling_buffer_progress_bar.n == buffer_number_of_required_episode - 1:
                filling_buffer_progress_bar.update(1)
                buffer_bar_finished = True
        else:
            if not buffer_filled and not batch_instead_of_buff:
                filling_buffer_progress_bar.update(1)
                if filling_buffer_progress_bar.n == buffer_number_of_required_episode:
                    buffer_bar_finished = True

        # Update curriculum learning based on episode outcomes
        if curriculum_learning:
            if using_eavesdropper:
                episodes_outcomes = Task_Manager.compute_episodes_outcome(
                    downlink_sum=training_envs.get_downlink_sum_for_success_conditions(),
                    uplink_sum=training_envs.get_uplink_sum_for_success_conditions(),
                    best_eavesdropper_sum=training_envs.get_best_eavesdropper_sum_for_success_conditions()
                )
            else:
                episodes_outcomes = Task_Manager.compute_episodes_outcome(
                    downlink_sum=training_envs.get_downlink_sum_for_success_conditions(),
                    uplink_sum=training_envs.get_uplink_sum_for_success_conditions()
                )
            Task_Manager.update_episode_outcomes(episodes_outcomes)


        # Calculate episode-level averages
        avg_actor_loss /= max_num_step_per_episode
        avg_critic_loss /= max_num_step_per_episode
        avg_reward = np.mean(total_reward) / max_num_step_per_episode

        # Calculate fairness metrics for this episode
        avg_user_fairness_all_envs = np.mean(instant_user_jain_fairness)
        avg_user_fairness_all_episode_general[episode] = avg_user_fairness_all_envs

        # Update reward function tracking arrays
        for function_name in decisive_reward_functions:
            decisive_rewards_average_episode[function_name][episode] = np.mean(
                decisive_rewards_current_episode[function_name]
            )

        for function_name in informative_reward_functions:
            informative_rewards_average_episode[function_name][episode] = np.mean(
                informative_rewards_current_episode[function_name]
            )

        # Save network models based on performance and schedule
        if best_average_reward < avg_reward and not debugging:
            best_average_reward = avg_reward
            saving_model_directory = f"{log_dir}/model_saved/best_model"
            network.save_models(saving_model_directory)

        if episode == num_episode - 1 and not debugging:
            saving_model_directory = f"{log_dir}/model_saved/final_episode"
            network.save_models(saving_model_directory)
        elif episode % saving_frequency == 0 and not debugging:
            saving_model_directory = f"{log_dir}/model_saved/episode_{episode}"
            network.save_models(saving_model_directory)

        # Calculate per-environment performance metrics
        average_reward_per_env[episode] = np.mean(instant_user_rewards, axis=1)
        episode_max_instant_reward_reached_per_env = np.max(instant_user_rewards, axis=1)
        average_best_reward_per_env[episode] = episode_max_instant_reward_reached_per_env

        # Find environment with best reward and associated fairness
        env_idx_best_reward_reached = np.argmax(episode_max_instant_reward_reached_per_env)
        fairness_for_best_reward = instant_user_jain_fairness[
            np.argmax(instant_user_rewards[env_idx_best_reward_reached]), 
            env_idx_best_reward_reached
        ]

        # Calculate basic reward metrics
        average_basic_reward_all_envs = np.mean(basic_reward_episode)
        basic_reward_for_best_env = basic_reward_episode[
            np.argmax(instant_user_rewards[env_idx_best_reward_reached]), 
            env_idx_best_reward_reached
        ]
        best_basic_reward_per_env = np.max(basic_reward_episode, axis=0)

        # Create comprehensive episode summary message for console (with emojis)
        console_message = (
            f"\n\n"
            f"╔══════════════════════════════════════════════════════════════════════════════════════════════════╗\n"
            f"║ 🎯 TRAINING EPISODE #{episode - index_episode_buffer_filled:3d} │ 🧠 Actor Optimizations: {optim_steps_actor:4d} ║\n"
            f"╠══════════════════════════════════════════════════════════════════════════════════════════════════╣\n"
            f"║ 📍 POSITIONING:\n"
            f"║    User Equipment Positions: {user_positions}\n"
            f"|    Eavesdroppers Positions: {eavesdroppers_positions}\n"
            f"║ ──────────────────────────────────────────────────────────────────────────────────────────────── ║\n"
            f"║ 🏆 REWARDS:\n"
            f"║    • Average Reward: {avg_reward:8.4f} │ Max Instant: {np.max(instant_user_rewards):8.4f}\n"
            f"║    • Best per Environment: {episode_max_instant_reward_reached_per_env}\n"
            f"║    • Baseline Reward Avg: {average_basic_reward_all_envs:8.4f} │ Best Baseline: {basic_reward_for_best_env:8.4f}\n"
            f"║    • Best Baseline per Env: {best_basic_reward_per_env}\n"
            f"║ ──────────────────────────────────────────────────────────────────────────────────────────────── ║\n"
            f"║ ⚖️  FAIRNESS:\n"
            f"║    • Average Fairness: {avg_user_fairness_all_envs:8.4f} │ Best Reward Fairness: {fairness_for_best_reward:8.4f}\n"
            f"║ ──────────────────────────────────────────────────────────────────────────────────────────────── ║\n"
            f"║ 📊 PERFORMANCE:\n"
            f"║    • Actor Loss: {avg_actor_loss:8.4f} │ Critic Loss: {avg_critic_loss:8.4f}\n"
            f"╚══════════════════════════════════════════════════════════════════════════════════════════════════╝\n"
        )

        # Create ASCII version for file logging
        log_message = (
            f"\n\n"
            f"+====================================================================================================+\n"
            f"| [TRAIN] EPISODE #{episode - index_episode_buffer_filled:3d} | Actor Optimizations: {optim_steps_actor:4d} |\n"
            f"+====================================================================================================+\n"
            f"| POSITIONING:\n"
            f"|    User Equipment Positions: {user_positions}\n"
            f"| ---------------------------------------------------------------------------------------------------- |\n"
            f"| REWARDS:\n"
            f"|    * Average Reward: {avg_reward:8.4f} | Max Instant: {np.max(instant_user_rewards):8.4f}\n"
            f"|    * Best per Environment: {episode_max_instant_reward_reached_per_env}\n"
            f"|    * Baseline Reward Avg: {average_basic_reward_all_envs:8.4f} | Best Baseline: {basic_reward_for_best_env:8.4f}\n"
            f"|    * Best Baseline per Env: {best_basic_reward_per_env}\n"
            f"| ---------------------------------------------------------------------------------------------------- |\n"
            f"| FAIRNESS:\n"
            f"|    * Average Fairness: {avg_user_fairness_all_envs:8.4f} | Best Reward Fairness: {fairness_for_best_reward:8.4f}\n"
            f"| ---------------------------------------------------------------------------------------------------- |\n"
            f"| PERFORMANCE:\n"
            f"|    * Actor Loss: {avg_actor_loss:8.4f} | Critic Loss: {avg_critic_loss:8.4f}\n"
            f"+====================================================================================================+\n"
        )

        # Display beautiful console message
        tqdm.write(console_message)
        
        # Log ASCII version to file
        logger.verbose(log_message)

        # Log episode-level metrics to TensorBoard
        writer.add_scalar("Actor Loss/Average actor Loss per episode", avg_actor_loss, episode)
        writer.add_scalar("Critic Loss/Average critic Loss per episode", avg_critic_loss, episode)
        writer.add_scalar("General/Time per episode", time.time() - start_episode_time, episode)

        # Log total system performance metrics
        writer.add_scalar("General/Mean total SSR per episode", 
                         np.mean(instant_user_rewards + instant_eavesdropper_rewards), episode)
        writer.add_scalar("General/Max total SSR per episode", 
                         np.max(instant_user_rewards + instant_eavesdropper_rewards), episode)
        
        # Log decisive reward function metrics
        for function_name in decisive_reward_functions:
            writer.add_scalar(f"Decisive Rewards/{function_name}/reward", 
                            decisive_rewards_average_episode[function_name][episode], episode)
            writer.add_scalar(f"Decisive Rewards/{function_name}/Averaged_reward", 
                            np.mean(decisive_rewards_average_episode[function_name][:episode+1]), episode)

        # Log informative reward function metrics
        for function_name in informative_reward_functions:
            writer.add_scalar(f"Informative Rewards/{function_name}", 
                            informative_rewards_average_episode[function_name][episode], episode)
            writer.add_scalar(f"Informative Rewards/Averaged_{function_name}", 
                            np.mean(informative_rewards_average_episode[function_name][:episode+1]), episode)

        # Log comprehensive reward metrics
        writer.add_scalar("Rewards/Average reward per episode", avg_reward, episode)
        writer.add_scalar("Rewards/Best average reward per episode", best_average_reward, episode)
        writer.add_scalar("Rewards/Average Max reward reached per episode", 
                         np.mean(episode_max_instant_reward_reached_per_env), episode)
        writer.add_scalar("Rewards/Best Max reward reached per episode", 
                         np.max(episode_max_instant_reward_reached_per_env), episode)
        writer.add_scalar("Rewards/Mean average reward", 
                         np.mean(average_reward_per_env[:episode+1]), episode)

        # Log fairness metrics
        writer.add_scalar("Fairness/Max Reward's Fairness", fairness_for_best_reward, episode)
        writer.add_scalar("Fairness/Mean Average User Fairness", 
                         np.mean(avg_user_fairness_all_episode_general[:episode+1]), episode)
        writer.add_scalar("Fairness/Average User Fairness per episode", avg_user_fairness_all_envs, episode)
        
        # Log eavesdropper metrics if applicable
        if using_eavesdropper:
            avg_eavesdropper_reward = np.mean(instant_eavesdropper_rewards)
            writer.add_scalar("Eavesdropper/Average reward per episode", avg_eavesdropper_reward, episode)
        
        # Update progress bar and log final episode metrics
        if buffer_filled:
            writer.add_scalar("Replay Buffer/Average reward stored", 
                             np.mean(network.replay_buffer.reward_buffer), current_step)

    # ========================================================================
    # EVALUATION PHASE
    # ========================================================================

        # Conduct periodic evaluation if specified and buffer is filled
        if conduct_eval and episode % eval_period == 0 and buffer_filled:
            tqdm.write(f"\n🔍 CONDUCTING EVALUATION...\n")
        
            # TODO: Implement rendering functionality for evaluation visualization
            # if use_rendering:
            #     renderer = SituationRenderer(
            #         M=eval_env.M,
            #         lambda_h=eval_env.lambda_h,
            #         N_t=training_envs.BS_transmit_antennas,
            #         max_step_per_episode=training_config.get("max_steps_per_episode", 20000),
            #         BS_position=training_envs.BS_position,
            #         d_h=eval_env.d_h,
            #         RIS_position=eval_env.RIS_position,
            #         users_position=eval_env.get_users_positions(),
            #     )

            # Initialize evaluation tracking variables
            num_eval_periode += 1
            eval_current_step = 0

            # Allocate arrays for per-episode evaluation metrics
            whole_episode_rewards = np.zeros(episode_per_eval)
            whole_episode_fairness = np.zeros(episode_per_eval)
            whole_episode_max_rewards = np.zeros(episode_per_eval)
            whole_episode_best_reward_fairness = np.zeros(episode_per_eval)

            eval_total_reward = np.zeros(n_eval_rollout_threads)

            # Initialize eavesdropper reward tracking if applicable
            if using_eavesdropper:
                whole_episode_eaves_rewards = np.zeros(episode_per_eval)

            # ====================================================================
            # EVALUATION EPISODE LOOP
            # ====================================================================
            for eval_episode in tqdm(range(episode_per_eval), 
                                   desc="[EVAL] EVALUATION", 
                                   position=3, 
                                   bar_format="{l_bar}{bar:50}{r_bar}{bar:-10b}",
                                   ncols=140,
                                   colour='red',
                                   ascii="▏▎▍▌▋▊▉█"):

                # Reset evaluation environment for new episode
                eval_env.reset()

                # Log starting positions for evaluation episode
                console_message = (
                    f"\n📍 EVAL Episode {eval_episode} Starting Positions:\n"
                    f"   👥 Users: {eval_env.get_users_positions()}\n"
                    f"   👁️  Eavesdroppers: {eval_env.get_eavesdroppers_positions()}\n"
                )
                log_message = (
                    f"\n[EVAL] Episode {eval_episode} Starting Positions:\n"
                    f"   Users: {eval_env.get_users_positions()}\n"
                    f"   Eavesdroppers: {eval_env.get_eavesdroppers_positions()}\n"
                )
                tqdm.write(console_message)
                logger.verbose(log_message)
                
                # Initialize evaluation episode tracking arrays
                eval_instant_user_jain_fairness = np.zeros((max_num_step_per_episode + 1, n_eval_rollout_threads))
                episode_user_rewards = np.zeros(shape=(max_num_step_per_episode+1, n_eval_rollout_threads))

                if using_eavesdropper:
                    episode_eaves_rewards = np.zeros(shape=(max_num_step_per_episode+1, n_eval_rollout_threads))


                # ================================================================
                # EVALUATION STEP LOOP
                # ================================================================
                for num_step in range(max_num_step_per_episode):
                    eval_current_step += 1

                    # Get initial states for first step, otherwise use next states
                    if num_step == 0:
                        states = eval_env.get_states()
                        states = np.squeeze(states, axis=1)
                    else:
                        states = next_states
                    
                    # Select actions using trained network (no exploration noise)
                    selected_actions = torch.stack([network.select_action(states[i]) for i in range(n_eval_rollout_threads)])
                    states, selected_actions, rewards, next_states = eval_env.step(states, selected_actions)

                    # Process rewards efficiently using vectorized operations
                    rewards_np = rewards
                    episode_user_rewards[num_step] = rewards_np
                    eval_total_reward += rewards_np
                    max_instant_reward = max(max_instant_reward, rewards.max().item())

                    # Calculate and store fairness metrics for this step
                    current_fairness = np.round(eval_env.get_jain_fairness(), decimals=3).squeeze()
                    eval_instant_user_jain_fairness[num_step] = current_fairness

                    # Track eavesdropper rewards if applicable
                    if using_eavesdropper:
                        eavesdropper_rewards = np.array(eval_env.get_eavesdroppers_rewards()).squeeze()
                        episode_eaves_rewards[num_step, :] = eavesdropper_rewards


                # Store evaluation episode metrics
                whole_episode_rewards[eval_episode] = np.mean(episode_user_rewards)
                whole_episode_fairness[eval_episode] = np.mean(eval_instant_user_jain_fairness)
                whole_episode_max_rewards[eval_episode] = np.max(episode_user_rewards)
                whole_episode_best_reward_fairness[eval_episode] = eval_instant_user_jain_fairness[
                    np.unravel_index(np.argmax(episode_user_rewards), eval_instant_user_jain_fairness.shape)
                ]

            # Calculate comprehensive evaluation metrics across all episodes
            mean_reward = np.mean(whole_episode_rewards)
            mean_fairness = np.mean(whole_episode_fairness)
            mean_max_reward = np.mean(whole_episode_max_rewards)
            mean_best_fairness = np.mean(whole_episode_best_reward_fairness)

            # Calculate eavesdropper-specific metrics if applicable
            if using_eavesdropper:
                mean_eaves_reward = np.mean(whole_episode_eaves_rewards)
                std_eaves_reward = np.std(whole_episode_eaves_rewards)
                reward_eaves_ratio = whole_episode_rewards / (whole_episode_eaves_rewards + 1e-6)
                mean_ratio = np.mean(reward_eaves_ratio)
                composite_score = whole_episode_rewards - 0.5 * whole_episode_eaves_rewards
                mean_composite_score = np.mean(composite_score)

                # Log eavesdropper-specific evaluation metrics
                writer.add_scalar("Evaluation/Mean total SSR per episode", mean_reward + mean_eaves_reward, episode)
                writer.add_scalar("Evaluation/Max total SSR per episode", 
                                 np.max(whole_episode_rewards + whole_episode_eaves_rewards), episode)
                writer.add_scalar("Evaluation/Total Eavesdropper Reward per episode", mean_eaves_reward, episode)
                writer.add_scalar("Evaluation/Reward-to-Eavesdropper Ratio", mean_ratio, episode)
                writer.add_scalar("Evaluation/Composite Score", mean_composite_score, episode)

            else:
                # Log standard evaluation metrics without eavesdroppers
                writer.add_scalar("Evaluation/Mean total SSR per episode", mean_reward, episode)
                writer.add_scalar("Evaluation/Max total SSR per episode", np.max(whole_episode_rewards), episode)

            # Log standard evaluation metrics
            writer.add_scalar("Evaluation/Average Fairness", mean_fairness, episode)
            writer.add_scalar("Evaluation/Fairness for the best reward", mean_best_fairness, episode)
            writer.add_scalar("Evaluation/Max reward reached per episode", mean_max_reward, episode)
            writer.add_scalar("Evaluation/Total Reward per episode", mean_reward, episode)

            # Create evaluation summary message for console (with emojis)
            if using_eavesdropper:
                console_message = (
                    f"\n╔══════════════════════════════════════════════════════════════════════════════════════════════╗\n"
                    f"║ 📊 EVALUATION PERIOD #{num_eval_periode:2d} ║\n"
                    f"╠══════════════════════════════════════════════════════════════════════════════════════════════╣\n"
                    f"║ 🏆 REWARDS: Avg: {mean_reward:8.4f} │ Max: {mean_max_reward:8.4f}\n"
                    f"║ ⚖️  FAIRNESS: Best: {mean_best_fairness:8.4f} │ Avg: {mean_fairness:8.4f}\n"
                    f"║ 👁️  EAVESDROPPER: Reward: {mean_eaves_reward:8.4f} │ Ratio: {mean_ratio:8.4f}\n"
                    f"║ 🎯 COMPOSITE SCORE: {mean_composite_score:8.4f}\n"
                    f"╚══════════════════════════════════════════════════════════════════════════════════════════════╝\n"
                )
                log_message = (
                    f"\n+====================================================================================================+\n"
                    f"| [EVAL] PERIOD #{num_eval_periode:2d} |\n"
                    f"+====================================================================================================+\n"
                    f"| REWARDS: Avg: {mean_reward:8.4f} | Max: {mean_max_reward:8.4f}\n"
                    f"| FAIRNESS: Best: {mean_best_fairness:8.4f} | Avg: {mean_fairness:8.4f}\n"
                    f"| EAVESDROPPER: Reward: {mean_eaves_reward:8.4f} | Ratio: {mean_ratio:8.4f}\n"
                    f"| COMPOSITE SCORE: {mean_composite_score:8.4f}\n"
                    f"+====================================================================================================+\n"
                )
            else:
                console_message = (
                    f"\n╔══════════════════════════════════════════════════════════════════════════════════════════════╗\n"
                    f"║ 📊 EVALUATION PERIOD #{num_eval_periode:2d} ║\n"
                    f"╠══════════════════════════════════════════════════════════════════════════════════════════════╣\n"
                    f"║ 🏆 REWARDS: Avg: {mean_reward:8.4f} │ Max: {mean_max_reward:8.4f}\n"
                    f"║ ⚖️  FAIRNESS: Best: {mean_best_fairness:8.4f} │ Avg: {mean_fairness:8.4f}\n"
                    f"╚══════════════════════════════════════════════════════════════════════════════════════════════╝\n"
                )
                log_message = (
                    f"\n+====================================================================================================+\n"
                    f"| [EVAL] PERIOD #{num_eval_periode:2d} |\n"
                    f"+====================================================================================================+\n"
                    f"| REWARDS: Avg: {mean_reward:8.4f} | Max: {mean_max_reward:8.4f}\n"
                    f"| FAIRNESS: Best: {mean_best_fairness:8.4f} | Avg: {mean_fairness:8.4f}\n"
                    f"+====================================================================================================+\n"
                )
                
            # Display beautiful console message and log ASCII version to file
            tqdm.write(console_message)
            logger.verbose(log_message)