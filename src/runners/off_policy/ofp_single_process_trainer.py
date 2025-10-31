"""
Single-process training loop utilities for RIS Duplex RL experiments.

This module provides comprehensive training infrastructure for reinforcement learning
agents in RIS (Reconfigurable Intelligent Surface) environments using single-process execution.
It supports:

- setup_logger: Configures file and console logging with custom VERBOSE level
- ofp_single_process_trainer: Main training loop for single environment execution

Features:
- Single-environment training with curriculum learning (optionnal)
- Comprehensive reward tracking (decisive and informative rewards)
- Real-time evaluation with separate evaluation environments
- Advanced logging with TensorBoard integration
- Fairness metrics tracking using Jain's Fairness Index
- Rendering and visualization capabilities

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
from tqdm import tqdm
import numpy as np
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter
from src.environment.tools import parse_args, parse_config, write_line_to_file, SituationRenderer, TaskManager
from src.environment.physics.signal_processing import watts_to_dbm


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


def ofp_single_process_trainer(training_envs, network, training_config, log_dir, writer,
                           eval_env=None, action_noise_activated=False,
                           batch_instead_of_buff=False, use_rendering=False):
    """
    Execute single-process training loop for RIS Duplex RL agents.
    
    This function orchestrates the complete training pipeline for single-environment
    execution including:
    - Single environment interactions with curriculum learning
    - Comprehensive reward and fairness metric tracking
    - Periodic evaluation with separate evaluation environments
    - Advanced logging and TensorBoard integration
    - Rendering and visualization capabilities
    
    Args:
        training_envs: Single training environment instance
        network: Neural network agent implementing the RL algorithm
        training_config (dict): Training hyperparameters and configuration
        log_dir (str): Directory path for saving logs and model checkpoints
        writer: TensorBoard SummaryWriter for experiment tracking
        eval_env: Evaluation environment for periodic performance assessment.
            Defaults to None (no evaluation).
        action_noise_activated (bool, optional): Enable action noise for exploration.
            Defaults to False.
        batch_instead_of_buff (bool, optional): Use batch size instead of buffer size
            for training decisions. Defaults to False.
        use_rendering (bool, optional): Enable environment rendering and visualization.
            Defaults to False.
            
    Note:
        The function supports both decisive and informative reward functions,
        tracks Jain's Fairness Index for user fairness assessment, implements
        curriculum learning for progressive difficulty adjustment, and provides
        comprehensive visualization capabilities through rendering.
    """
    # Initialize dedicated logger for this training session
    # VERBOSE level messages are written to file but not displayed in console
    logger = setup_logger(log_dir)

    # Extract environment configuration
    env_config = training_envs.env_config

    # Parse training configuration parameters with sensible defaults
    debugging = training_config.get("debugging", False)
    env_debugging = env_config.get("debugging", False)
    num_episode = training_config.get("number_episodes", 15)
    solo_episode = num_episode == 1
    max_num_step_per_episode = training_config.get("max_steps_per_episode", 20000)
    reward_smoothing_factor = training_config.get("reward_smoothing_factor", 0.5)
    batch_size = training_config.get("batch_size", 128)
    frequency_information = training_config.get("frequency_information", 500)
    
    # Evaluation configuration
    conduct_eval = eval_env is not None
    episode_per_eval = training_config.get("episode_per_eval_env", 1) if conduct_eval else None
    eval_period = training_config.get("eval_period", 1) if conduct_eval else None
    
    # Model and plot saving configuration
    saving_frequency = training_config.get("network_save_checkpoint_frequency", 100)
    plot_saving_frequency = training_config.get("plot_save_checkpoint_frequency", 100)

    # Curriculum learning enables progressive difficulty adjustment
    curriculum_learning = training_config.get("Curriculum_Learning", False)
    
    # Action noise configuration for exploration
    noise_scale = training_config.get("noise_scale", 0.1)
    
    # Extract environment metadata
    num_users = training_envs.num_users
    using_eavesdropper = (training_envs.num_eavesdroppers > 0)

    # Initialize curriculum learning system if enabled
    if curriculum_learning:
        # TaskManager handles progressive difficulty scheduling
        grid_limit = env_config.get("user_spawn_limits")
        downlink_activated = env_config.get("BS_max_power") > 0
        uplink_activated = env_config.get("user_transmit_power") > 0
        
        Task_Manager = TaskManager(
            num_users,
            num_steps_per_episode=max_num_step_per_episode,
            user_limits=grid_limit,
            RIS_position=env_config.get("RIS_position"),
            downlink_uplink_eavesdropper_bools=[downlink_activated, uplink_activated, using_eavesdropper],
            thresholds=training_config.get("Curriculum_Learning_Thresholds", [0.5, 0.5]),
            random_seed=training_config.get("Task_Manager_random_seed", 126)
        )

    # Determine buffer size based on training mode
    buffer_size = deepcopy(batch_size) if batch_instead_of_buff else network.replay_buffer.buffer_size

    # Initialize training state variables
    index_episode_buffer_filled = None
    users_trajectory = np.zeros((num_episode, training_envs.num_users, 2))
    eavesdroppers_trajectory = np.zeros((num_episode, training_envs.num_eavesdroppers, 2))

    # Evaluation tracking variables
    num_eval_periode = 0
    eval_current_step = 0

    # Training progress tracking
    buffer_filled = False
    best_average_reward = 0

    # Eavesdropper positions are only relevant when eavesdroppers are present
    if not using_eavesdropper:
        eavesdroppers_positions = None
        
    # Initialize rendering data structures if visualization is enabled
    if use_rendering:
        best_power_patterns = {
            "theta_power_pattern": [],
            "theta_phi_power_pattern": [],
            "W_power_pattern": [],
            "rewards": [],
            "steps": [],
            "user_positions": [],
            "eavesdroppers_positions": []
        }

    # Initialize optimization step counters
    optim_steps_actor = 0
    optim_steps_critic = 0
    average_reward_per_env = np.zeros(num_episode)
    smoothed_average_fairness = np.zeros(num_episode)

    # Configure replay buffer filling progress tracking
    # TODO: Optimize buffer progress bar management for better performance
    buffer_bar_finished = False
    buffer_number_of_required_episode = buffer_size // (max_num_step_per_episode)
    buffer_smaller_than_one_episode = (buffer_size // (max_num_step_per_episode) < 1)
    length_episode_rb_matching = ((buffer_size % (max_num_step_per_episode)) == 0)
    
    # Ensure minimum buffer filling episodes
    if buffer_number_of_required_episode == 0:
        buffer_number_of_required_episode = 1

    # Initialize progress bar for replay buffer warmup phase
    filling_buffer_progress_bar = (
        tqdm(
            total=buffer_number_of_required_episode,
            desc="[BUFFER] FILLING REPLAY BUFFER",
            position=0,
            bar_format="{l_bar}{bar:50}{r_bar}{bar:-10b}",
            ncols=140,
            colour='blue',
            ascii="▏▎▍▌▋▊▉█",
            leave=True
        ) if not batch_instead_of_buff else None
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
        # Record episode start time for performance tracking
        start_episode_time = time.time()
        
        # Reset episode-level optimization counters
        optim_steps_actor_ep = 0
        optim_steps_critic_ep = 0

        # Apply curriculum learning difficulty configuration if enabled
        if curriculum_learning:
            difficulty_config = Task_Manager.generate_episode_configs()
            training_envs.reset(difficulty_config)
        else:
            training_envs.reset()
        
        # Get initial user positions for logging
        users_position = training_envs.get_users_positions()

        # Initialize eavesdropper tracking if applicable
        if using_eavesdropper:
            max_episode_eavesdropper_reward = 0
            eavesdroppers_positions = training_envs.get_eavesdroppers_positions()
            eavesdroppers_trajectory[episode] = eavesdroppers_positions
    
            # Initialize power pattern tracking for eavesdroppers
            episode_best_4_eavesdroppers_power_patterns = {
                "downlink_power_patterns": [],
                "uplink_power_patterns": [],
                "W_power_patterns": [],
                "rewards": [],
                "steps": [],
            }

        # Create position logging message based on training phase
        if buffer_filled:
            if using_eavesdropper:
                position_message = (
                    f"\nCommencing training episode {episode-buffer_number_of_required_episode} with users and eavesdroppers positions:\n"
                    f"   !~ Users Positions: {list(users_position)} \n"
                    f"   !~ Eavesdroppers Positions: {list(eavesdroppers_positions)} \n"
                )
            else:
                position_message = (
                    f"\nCommencing training episode {episode-buffer_number_of_required_episode} with users positions:\n"
                    f"   !~ Users Positions: {list(users_position)} \n"
                )
        else:
            if using_eavesdropper:
                position_message = (
                    f"\n Initializing positions for replay buffer episode {episode} with users and eavesdroppers positions:\n"
                    f"   !~ Users Positions: {list(users_position)} \n"
                    f"   !~ Eavesdroppers Positions: {list(eavesdroppers_positions)} \n"
                )
            else:
                position_message = (
                    f"\n Initializing positions for replay buffer episode {episode} with users positions:\n"
                    f"   !~ Users Positions: {list(users_position)} \n"
                )

        # Store user trajectory for analysis
        users_trajectory[episode] = users_position

        # Log episode starting positions
        logger.verbose(position_message)

        # Initialize episode tracking arrays
        instant_user_rewards = np.zeros(max_num_step_per_episode) - np.inf
        instant_eavesdropper_rewards = np.zeros(max_num_step_per_episode)

        # Initialize reward and fairness tracking arrays
        average_rewards = np.zeros(max_num_step_per_episode + 1)
        paper_average_rewards = np.zeros(max_num_step_per_episode + 1)
        instant_user_jain_fairness = np.zeros(max_num_step_per_episode + 1)
        
        # Initialize communication rates and signal strength tracking for solo episode
        if solo_episode:
            # Smoothed communication rates per user (downlink, uplink, total)
            smoothed_user_downlink_rates = np.zeros((num_users, max_num_step_per_episode))
            smoothed_user_uplink_rates = np.zeros((num_users, max_num_step_per_episode))
            smoothed_user_total_rates = np.zeros((num_users, max_num_step_per_episode))
            
            # Smoothed communication rates for eavesdroppers (max per user across eavesdroppers)
            if using_eavesdropper:
                smoothed_eavesdropper_downlink_rates = np.zeros((num_users, max_num_step_per_episode))
                smoothed_eavesdropper_uplink_rates = np.zeros((num_users, max_num_step_per_episode))
                smoothed_eavesdropper_total_rates = np.zeros((num_users, max_num_step_per_episode))
            
            # Signal strength tracking (min/max per user)
            user_min_downlink_signals = np.zeros((num_users, max_num_step_per_episode))
            user_max_downlink_signals = np.zeros((num_users, max_num_step_per_episode))
            user_min_uplink_signals = np.zeros((num_users, max_num_step_per_episode))
            user_max_uplink_signals = np.zeros((num_users, max_num_step_per_episode))
            
            if using_eavesdropper:
                eavesdropper_min_signals = np.zeros((num_users, max_num_step_per_episode))
                eavesdropper_max_signals = np.zeros((num_users, max_num_step_per_episode))
        else:
            # Team average tracking for non-solo episodes
            team_user_rewards = []
            team_eavesdropper_rewards = []
            team_avg_signal_strengths = {'min': [], 'max': []}
        
        # Initialize loss and performance tracking variables
        avg_actor_loss = 0
        avg_critic_loss = 0
        total_reward = 0
        additional_information_best_case = 0
        basic_reward_episode = np.zeros(max_num_step_per_episode)
        step_time_list = []
        max_episode_reward = -np.inf

        # Initialize renderer for visualization if enabled
        if use_rendering:
            # SituationRenderer collects visualization artifacts for later export
            renderer = SituationRenderer(
                M=training_envs.M,
                L=training_envs.num_eavesdroppers,
                N_t=training_envs.BS_transmit_antennas,
                lambda_h=training_envs.lambda_h,
                max_step_per_episode=training_config.get("max_steps_per_episode", 20000),
                BS_position=training_envs.BS_position,
                d_h=training_envs.d_h,
                RIS_position=training_envs.RIS_position,
                users_position=users_position,
                eavesdroppers_positions=eavesdroppers_positions,
                eavesdropper_moving=training_envs.moving_eavesdroppers,
                num_frames=200
            )

            # Initialize power pattern tracking for best episode performance
            episode_best_power_patterns = {
                "downlink_power_patterns": [],
                "uplink_power_patterns": [],
                "W_power_patterns": [],
                "rewards": [],
                "steps": [],
            }


        # ====================================================================
        # EPISODE STEP LOOP
        # ====================================================================
        for num_step in range(max_num_step_per_episode):
            current_step = episode * max_num_step_per_episode + num_step

            # Get current state from environment
            state = training_envs.get_states()[0] if num_step == 0 else next_state[0]

            # Action selection with optional noise for exploration
            if action_noise_activated:
                selected_action, selected_noised_action = network.select_noised_action(state, noise_scale=noise_scale)
                state, _, reward, next_state = training_envs.step(state, selected_noised_action)
            else:
                selected_action = network.select_action(state)
                state, selected_action, reward, next_state = training_envs.step(state, selected_action)

            # Store transition in replay buffer (batch size = 1 for single-process)
            network.store_transition(state, selected_action, reward, next_state, batch_size=1)

            # Process reward information
            reward_value = reward.item()

            # Track rewards and calculate running averages
            instant_user_rewards[num_step] = reward_value
            total_reward = np.sum(instant_user_rewards)
            avg_reward = total_reward / (num_step + 1)
            average_rewards[num_step] = avg_reward
            
            # Apply reward smoothing if not the first step
            if num_step > 1:
                paper_average_rewards[num_step] = (
                    reward_smoothing_factor * paper_average_rewards[num_step - 1] + 
                    (1 - reward_smoothing_factor) * reward_value
                )

            # Calculate and store Jain's Fairness Index
            instant_fairness = round(training_envs.get_user_jain_fairness(), ndigits=4)
            instant_user_jain_fairness[num_step] = instant_fairness
            # Track eavesdropper rewards if applicable
            if using_eavesdropper:
                eavesdropper_reward = training_envs.get_eavesdropper_rewards()
                instant_eavesdropper_rewards[num_step] = eavesdropper_reward
                total_eavesdropper_reward = np.sum(instant_eavesdropper_rewards)

            # Track baseline reward for comparison
            instant_baseline_reward = training_envs.get_basic_reward()
            basic_reward_episode[num_step] = instant_baseline_reward

            # Track communication rates and signal strengths
            if solo_episode:
                # Get communication rates from environment
                user_rates = training_envs.get_user_communication_rates()
                
                # Apply smoothing and store per-user rates
                if num_step == 0:
                    smoothed_user_downlink_rates[:, num_step] = user_rates['downlink']
                    smoothed_user_uplink_rates[:, num_step] = user_rates['uplink']
                    smoothed_user_total_rates[:, num_step] = user_rates['total']
                else:
                    smoothed_user_downlink_rates[:, num_step] = (
                        reward_smoothing_factor * smoothed_user_downlink_rates[:, num_step - 1] +
                        (1 - reward_smoothing_factor) * user_rates['downlink']
                    )
                    smoothed_user_uplink_rates[:, num_step] = (
                        reward_smoothing_factor * smoothed_user_uplink_rates[:, num_step - 1] +
                        (1 - reward_smoothing_factor) * user_rates['uplink']
                    )
                    smoothed_user_total_rates[:, num_step] = (
                        reward_smoothing_factor * smoothed_user_total_rates[:, num_step - 1] +
                        (1 - reward_smoothing_factor) * user_rates['total']
                    )
                
                # Track eavesdropper rates if applicable
                if using_eavesdropper:
                    eavesdropper_rates = training_envs.get_eavesdropper_communication_rates()
                    if num_step == 0:
                        # Use max across eavesdroppers for each user
                        smoothed_eavesdropper_downlink_rates[:, num_step] = np.max(eavesdropper_rates['downlink'], axis=1)
                        smoothed_eavesdropper_uplink_rates[:, num_step] = np.max(eavesdropper_rates['uplink'], axis=1)
                        smoothed_eavesdropper_total_rates[:, num_step] = eavesdropper_rates['max_per_user']
                    else:
                        max_dl = np.max(eavesdropper_rates['downlink'], axis=1)
                        max_ul = np.max(eavesdropper_rates['uplink'], axis=1)
                        smoothed_eavesdropper_downlink_rates[:, num_step] = (
                            reward_smoothing_factor * smoothed_eavesdropper_downlink_rates[:, num_step - 1] +
                            (1 - reward_smoothing_factor) * max_dl
                        )
                        smoothed_eavesdropper_uplink_rates[:, num_step] = (
                            reward_smoothing_factor * smoothed_eavesdropper_uplink_rates[:, num_step - 1] +
                            (1 - reward_smoothing_factor) * max_ul
                        )
                        smoothed_eavesdropper_total_rates[:, num_step] = (
                            reward_smoothing_factor * smoothed_eavesdropper_total_rates[:, num_step - 1] +
                            (1 - reward_smoothing_factor) * eavesdropper_rates['max_per_user']
                        )
                
                # Track signal strengths
                user_signals = training_envs.get_user_signal_strengths()
                user_min_downlink_signals[:, num_step] = user_signals['min_downlink']
                user_max_downlink_signals[:, num_step] = user_signals['max_downlink']
                user_min_uplink_signals[:, num_step] = user_signals['min_uplink']
                user_max_uplink_signals[:, num_step] = user_signals['max_uplink']
                
                if using_eavesdropper:
                    eavesdropper_signals = training_envs.get_eavesdropper_signal_strengths()
                    eavesdropper_min_signals[:, num_step] = eavesdropper_signals['min_across_eaves']
                    eavesdropper_max_signals[:, num_step] = eavesdropper_signals['max_across_eaves']
            else:
                # Track team averages for non-solo episodes (throughout the episode)
                team_user_rewards.append(reward_value)
                if using_eavesdropper:
                    team_eavesdropper_rewards.append(eavesdropper_reward)
                
                # Get signal strengths and compute team averages
                user_signals = training_envs.get_user_signal_strengths()
                # Average across all users for downlink and uplink separately, then average those
                avg_min_signal = np.mean([
                    np.mean(user_signals['min_downlink']), 
                    np.mean(user_signals['min_uplink'])
                ])
                avg_max_signal = np.mean([
                    np.mean(user_signals['max_downlink']), 
                    np.mean(user_signals['max_uplink'])
                ])
                team_avg_signal_strengths['min'].append(avg_min_signal)
                team_avg_signal_strengths['max'].append(avg_max_signal)

            # Store power patterns for visualization at specified intervals
            if use_rendering and num_step % (max_num_step_per_episode // renderer.num_frames) == 0:
                W = training_envs.get_W()
                total_power_deployed = round(np.sum((W @ W.conj().T).diagonal().real), 4)
                renderer.store_power_patterns(
                    reward=reward_value,
                    W_power_patterns=training_envs.get_W_power_patterns(),
                    Downlink_power_patterns=training_envs.get_downlink_power_patterns(),
                    Uplink_power_patterns=training_envs.get_uplink_power_patterns()
                )

            # Set actor network to training mode
            network.actor.train()
            
            if not buffer_filled:
                    index_episode_buffer_filled = episode
                    
            # Perform network training if replay buffer is sufficiently filled
            if network.replay_buffer.size >= buffer_size:
                if not buffer_filled:
                    index_episode_buffer_filled = episode
                    buffer_filled = True

                # Track and store best reward patterns for visualization
                if reward_value > max_episode_reward and use_rendering:
                    max_episode_reward = round(reward_value, 4)
                    
                    # Get additional information for best case analysis
                    additional_information_best_case = training_envs.get_additionnal_informations()
                    W = training_envs.get_W()
                    total_power_deployed = round(np.trace(np.diag(np.diag(W @ W.conj().T)).real), 4)
                    
                    # Extract power patterns for visualization
                    max_W_power_patterns = training_envs.get_W_power_patterns()
                    max_downlink_power_patterns = training_envs.get_downlink_power_patterns()
                    max_uplink_power_patterns = training_envs.get_uplink_power_patterns()

                    # Store best power patterns for animation
                    episode_best_power_patterns["rewards"].append(deepcopy(reward_value))
                    episode_best_power_patterns["steps"].append(deepcopy(num_step))
                    episode_best_power_patterns["W_power_patterns"].append(deepcopy(max_W_power_patterns))
                    episode_best_power_patterns["downlink_power_patterns"].append(deepcopy(max_downlink_power_patterns))
                    episode_best_power_patterns["uplink_power_patterns"].append(deepcopy(max_uplink_power_patterns))

                # Track and store best eavesdropper reward patterns
                if using_eavesdropper and eavesdropper_reward > max_episode_eavesdropper_reward and use_rendering:
                    max_episode_eavesdropper_reward = round(eavesdropper_reward, 4)
                    W = training_envs.get_W()
                    total_power_deployed = round(np.trace(np.diag(np.diag(W @ W.conj().T)).real), 4)
                    
                    # Extract eavesdropper power patterns for visualization
                    eave_max_W_power_patterns = training_envs.get_W_power_patterns()
                    eave_max_downlink_power_patterns = training_envs.get_downlink_power_patterns()
                    eave_max_uplink_power_patterns = training_envs.get_uplink_power_patterns()

                    # Store best eavesdropper power patterns for animation
                    episode_best_4_eavesdroppers_power_patterns["rewards"].append(deepcopy(max_episode_eavesdropper_reward))
                    episode_best_4_eavesdroppers_power_patterns["steps"].append(deepcopy(num_step))
                    episode_best_4_eavesdroppers_power_patterns["W_power_patterns"].append(deepcopy(eave_max_W_power_patterns))
                    episode_best_4_eavesdroppers_power_patterns["downlink_power_patterns"].append(deepcopy(eave_max_downlink_power_patterns))
                    episode_best_4_eavesdroppers_power_patterns["uplink_power_patterns"].append(deepcopy(eave_max_uplink_power_patterns))

                elif reward_value > max_episode_reward:
                    max_episode_reward = round(reward_value, 4)
                    additional_information_best_case = training_envs.get_additionnal_informations()

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


                # Log detailed metrics for single episode runs
                if solo_episode:
                    writer.add_scalar("Rewards/Instant User reward", reward_value, current_step)
                    writer.add_scalar("Rewards/Baseline Instant Reward", instant_baseline_reward, current_step)
                    writer.add_scalar("Fairness/Instant JFI", instant_fairness, current_step)
                    writer.add_scalar("Fairness/JFI for best reward obtained", 
                                     instant_user_jain_fairness[np.argmax(instant_user_rewards)], current_step)
                    if using_eavesdropper:
                        writer.add_scalar("Eavesdropper/Instant Eavesdroppers reward", eavesdropper_reward, current_step)
                    
                    # Log smoothed communication rates per user
                    for k in range(num_users):
                        writer.add_scalar(f"Communication Rates/User_{k}/Smoothed Downlink Rate", 
                                         smoothed_user_downlink_rates[k, num_step], current_step)
                        writer.add_scalar(f"Communication Rates/User_{k}/Smoothed Uplink Rate", 
                                         smoothed_user_uplink_rates[k, num_step], current_step)
                        writer.add_scalar(f"Communication Rates/User_{k}/Smoothed Total Rate", 
                                         smoothed_user_total_rates[k, num_step], current_step)
                    
                    # Log eavesdropper communication rates
                    if using_eavesdropper:
                        for k in range(num_users):
                            writer.add_scalar(f"Communication Rates/Eavesdropper_User_{k}/Smoothed Downlink Rate", 
                                             smoothed_eavesdropper_downlink_rates[k, num_step], current_step)
                            writer.add_scalar(f"Communication Rates/Eavesdropper_User_{k}/Smoothed Uplink Rate", 
                                             smoothed_eavesdropper_uplink_rates[k, num_step], current_step)
                            writer.add_scalar(f"Communication Rates/Eavesdropper_User_{k}/Smoothed Total Rate", 
                                             smoothed_eavesdropper_total_rates[k, num_step], current_step)
                    
                    # Log signal strength evolution (min/max)
                    for k in range(num_users):
                        writer.add_scalar(f"Signal Strength/User_{k}/Min Downlink (dBm)", 
                                         user_min_downlink_signals[k, num_step], current_step)
                        writer.add_scalar(f"Signal Strength/User_{k}/Max Downlink (dBm)", 
                                         user_max_downlink_signals[k, num_step], current_step)
                        writer.add_scalar(f"Signal Strength/User_{k}/Min Uplink (dBm)", 
                                         user_min_uplink_signals[k, num_step], current_step)
                        writer.add_scalar(f"Signal Strength/User_{k}/Max Uplink (dBm)", 
                                         user_max_uplink_signals[k, num_step], current_step)
                    
                    if using_eavesdropper:
                        for k in range(num_users):
                            writer.add_scalar(f"Signal Strength/Eavesdropper_User_{k}/Min Signal (dBm)", 
                                             eavesdropper_min_signals[k, num_step], current_step)
                            writer.add_scalar(f"Signal Strength/Eavesdropper_User_{k}/Max Signal (dBm)", 
                                             eavesdropper_max_signals[k, num_step], current_step)


                # Periodic logging of training metrics and performance statistics
                if (current_step + 1) % frequency_information == 0:
                    W = training_envs.get_W()
                    total_power_deployed = round(np.trace(np.diag(np.diag(W @ W.conj().T)).real), 4)

                    # Log replay buffer statistics
                    writer.add_scalar("Replay Buffer/Average reward stored", 
                                     np.mean(network.replay_buffer.reward_buffer), current_step)
                    writer.add_histogram("Replay Buffer/Rewards", 
                                        network.replay_buffer.reward_buffer.squeeze(), current_step)

                    # Calculate combined reward metrics
                    reward_combined = instant_user_rewards + instant_eavesdropper_rewards

                    # Calculate and log local metrics for the current information window
                    if (num_step + 1) % frequency_information == 0:
                        local_average_reward = np.mean(instant_user_rewards[num_step + 1 - frequency_information: num_step])
                        local_average_basic_reward = np.mean(basic_reward_episode[num_step + 1 - frequency_information: num_step])
                        local_user_fairness = round(
                            np.mean(instant_user_jain_fairness[num_step + 1 - frequency_information: num_step]), 
                            ndigits=4
                        )

                        # Log reward metrics
                        writer.add_scalar("Rewards/Local Average Reward", local_average_reward, current_step)
                        writer.add_histogram("Rewards/Paper Average Reward", 
                                            paper_average_rewards[num_step + 1 - frequency_information: num_step], current_step)
                        writer.add_scalar("Rewards/Global Average Reward", avg_reward, current_step)
                        writer.add_scalar("Rewards/Max Instant Reward", np.max(instant_user_rewards), current_step)
                        writer.add_scalar("Rewards/Local Average Baseline Reward", local_average_basic_reward, current_step)
                        writer.add_histogram("Rewards/Instant reward", 
                                            instant_user_rewards[num_step + 1 - frequency_information: num_step], current_step)

                        # Log loss metrics
                        if optim_steps_actor_ep:
                            current_avg_actor_loss = avg_actor_loss / optim_steps_actor_ep
                        else:
                            current_avg_actor_loss = 0
                        if optim_steps_critic_ep:
                            current_avg_critic_loss = avg_critic_loss / optim_steps_critic_ep
                        else:
                            current_avg_critic_loss = 0
                        writer.add_scalar("Actor Loss/Instant actor loss", actor_loss, current_step)
                        writer.add_scalar("Actor Loss/Current average actor loss", current_avg_actor_loss, current_step)
                        writer.add_scalar("Critic Loss/Instant critic loss", critic_loss, current_step)
                        writer.add_scalar("Critic Loss/Current average critic loss", current_avg_critic_loss, current_step)
                        writer.add_scalar("Fairness/User local average Fairness", local_user_fairness, current_step)
                        
                        # Log system performance metrics
                        writer.add_scalar("General/Power deployed (Watts)", total_power_deployed, current_step)

                        # Log eavesdropper metrics if applicable
                        if using_eavesdropper:
                            reward_combined = instant_user_rewards + instant_eavesdropper_rewards
                            local_average_reward_combined = np.mean(reward_combined[num_step + 1 - frequency_information: num_step])
                            writer.add_scalar("General/Local average total SSR", local_average_reward_combined, current_step)

                            avg_eavesdropper_reward = deepcopy(total_eavesdropper_reward) / num_step
                            local_average_eavesdropper_reward = np.mean(
                                instant_eavesdropper_rewards[num_step + 1 - frequency_information: num_step]
                            )
                            writer.add_scalar("Eavesdropper/Local average reward", local_average_eavesdropper_reward, current_step)
                            writer.add_histogram("Eavesdropper/Instant reward", 
                                                instant_eavesdropper_rewards[num_step + 1 - frequency_information: num_step], current_step)



                        # Create detailed training progress message
                        message = (
                            f"\n|--> TRAINING EPISODE {episode - index_episode_buffer_filled}, STEP {num_step + 1}\n"
                            f"     |--> Training the Actor NN takes {np.mean(step_time_list):.4f} sec on average across {len(step_time_list)} steps \n"
                            f"  |--> LOCAL AVERAGE REWARD {np.float16(local_average_reward)}, MAX INSTANT REWARD REACHED {np.max(instant_user_rewards)}\n"
                            f"  |--> LOCAL AVERAGE BASIC REWARD {np.float16(local_average_basic_reward)}, MAXIMUM INSTANT BASIC REWARD: {basic_reward_episode[np.argmax(instant_user_rewards)]:.4f}\n"
                            f"  |--> Detailed basic reward for best case: {additional_information_best_case}\n"
                            f" |--> EPISODE AVERAGE ACTOR LOSS {current_avg_actor_loss}, EPISODE AVERAGE CRITIC LOSS {current_avg_critic_loss}\n"
                            f" |--> USER FAIRNESS FOR THE BEST INSTANT REWARD {instant_user_jain_fairness[np.argmax(instant_user_rewards)]}, LOCAL USER FAIRNESS {local_user_fairness}\n"
                            f" |--> POWER DEPLOYED: {total_power_deployed:,.4f} Watts"
                            f"---------------------------\n"
                        )
                        
                        # Log detailed training information to file
                        logger.verbose(message)

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

        # ========================================================================
        # EPISODE SUMMARY METRICS
        # ========================================================================
        # Calculate episode-level averages
        if buffer_filled:
            if optim_steps_actor_ep:
                avg_actor_loss /= optim_steps_actor_ep
            else:
                avg_actor_loss = 0
            if optim_steps_critic_ep:
                avg_critic_loss /= optim_steps_critic_ep
            else:
                avg_critic_loss = 0
        avg_reward = np.mean(instant_user_rewards)
        avg_fairness = np.mean(instant_user_jain_fairness)
        
        # Calculate eavesdropper metrics if applicable
        if using_eavesdropper:
            avg_eavesdropper_reward = np.mean(instant_eavesdropper_rewards)

        # Save network models based on performance and schedule
        if best_average_reward < avg_reward and not debugging:
            best_average_reward = avg_reward
            saving_model_directory = f"{log_dir}/model_saved/best_model"
            
            # Create visualization artifacts for best model
            if use_rendering and buffer_filled:
                tqdm.write("Creating animations")
                os.makedirs(f'{log_dir}/data/episode_{episode}', exist_ok=True)

                if using_eavesdropper:

                    np.savez(f'{log_dir}/data/episode_{episode}/best_user_profile.npz',
                         BS_position=training_envs.BS_position,
                         eavesdroppers_positions=eavesdroppers_positions,
                         RIS_position=training_envs.RIS_position,
                         users_position=users_position, 
                         W_matrix = training_envs.get_W(),
                         max_W_power_patterns=max_W_power_patterns,
                         max_downlink_power_patterns=max_downlink_power_patterns,
                         max_uplink_power_patterns=max_uplink_power_patterns,
                         reward=max_episode_reward,
                         step_number=episode_best_power_patterns["steps"][-1],
                         episode=episode)
                    
                    renderer.plotly_single_image(
                        eave_max_W_power_patterns,
                        eave_max_downlink_power_patterns,
                        eave_max_uplink_power_patterns,
                        reward=max_episode_reward,
                        step_number=episode_best_4_eavesdroppers_power_patterns["steps"][-1],
                        episode=episode,
                        save_dir=f"{log_dir}/plots/episode_{episode}",
                        name_file=f"Ep_{episode}_Best_Eavesdropper_Profile.html"
                    )


                else:
                    np.savez(f'{log_dir}/data/episode_{episode}/best_user_profile.npz',
                                BS_position=training_envs.BS_position,
                                RIS_position=training_envs.RIS_position,
                                users_position=users_position, 
                                max_W_power_patterns=max_W_power_patterns,
                                W_matrix = training_envs.get_W(),
                                max_downlink_power_patterns=max_downlink_power_patterns,
                                max_uplink_power_patterns=max_uplink_power_patterns,
                                reward=max_episode_reward,
                                step_number=episode_best_power_patterns["steps"][-1],
                                episode=episode)
                
                renderer.plotly_single_image(
                    max_W_power_patterns,
                    max_downlink_power_patterns,
                    max_uplink_power_patterns,
                    reward=max_episode_reward,
                    step_number=episode_best_power_patterns["steps"][-1],
                    episode=episode,
                    save_dir=f"{log_dir}/plots/episode_{episode}",
                    name_file=f"Ep_{episode}_Best_User_Profile.html"
                )

                
                renderer.render_situation(save_dir=f"{log_dir}/animations/top_episodes", name_file=f"episode_{episode}_dynamic_profile.mp4")
                renderer.animate_power_patterns(
                    episode_best_power_patterns,
                    save_dir=f"{log_dir}/animations/top_episodes",
                    name_file=f"episode_{episode}_best_profiles.mp4"
                )

            if buffer_filled:
                network.save_models(saving_model_directory)

        # Save plots at designated frequency to visualize system evolution
        elif episode % plot_saving_frequency == 0 and not debugging:
            rendering = use_rendering and buffer_filled
            
            if rendering:
                # Create single image visualization
                renderer.plotly_single_image(
                    max_W_power_patterns,
                    max_downlink_power_patterns,
                    max_uplink_power_patterns,
                    reward=max_episode_reward,
                    step_number=episode_best_power_patterns["steps"][-1],
                    episode=episode,
                    save_dir=f"{log_dir}/plots/episode_{episode}",
                    name_file=f"Ep_{episode}_Best_User_Profile.html"
                )

                # Create animation visualizations
                tqdm.write("Creating animations")
                renderer.render_situation(save_dir=f"{log_dir}/animations/checkpoints", 
                                        name_file=f"episode_{episode}_dynamic_profile.mp4")
                renderer.animate_power_patterns(
                    episode_best_power_patterns,
                    save_dir=f"{log_dir}/animations/checkpoints",
                    name_file=f"episode_{episode}_best_profiles.mp4"
                )
            
            if rendering and using_eavesdropper:
                    renderer.plotly_single_image(
                        eave_max_W_power_patterns,
                        eave_max_downlink_power_patterns,
                        eave_max_uplink_power_patterns,
                        reward=max_episode_reward,
                        step_number=episode_best_4_eavesdroppers_power_patterns["steps"][-1],
                        episode=episode,
                        save_dir=f"{log_dir}/plots/episode_{episode}",
                        name_file=f"Ep_{episode}_Best_Eavesdropper_Profile.html"
                    )


        # Save final model at the end of training
        if episode == num_episode - 1 and not debugging:
            saving_model_directory = f"{log_dir}/model_saved/final_episode"
            network.save_models(saving_model_directory)
        elif episode % saving_frequency == 0 and not debugging:
            saving_model_directory = f"{log_dir}/model_saved/episode_{episode}"
            network.save_models(saving_model_directory)

        # ========================================================================
        # EPISODE SUMMARY LOGGING
        # ========================================================================
        episode_max_instant_reward_reached = max(instant_user_rewards)
        if index_episode_buffer_filled is None:
            index_episode_buffer_filled = 0
        
        if buffer_filled:
            # Log replay buffer statistics
            writer.add_scalar("Replay Buffer/Average reward stored", 
                             np.mean(network.replay_buffer.reward_buffer), current_step)

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

            # Create console message for episode summary (with emojis)
            console_message = (
                f"\n\n"
                f"╔══════════════════════════════════════════════════════════════════════════════════════════════════╗\n"
                f"║ 🎯 TRAINING EPISODE #{episode - index_episode_buffer_filled:3d} │ 🧠 Actor Optimizations: {optim_steps_actor:4d} ║\n"
                f"╠══════════════════════════════════════════════════════════════════════════════════════════════════╣\n"
                f"║ 📍 POSITIONING:\n"
                f"║    User Equipment Positions: {users_position}\n"
                f"║ ──────────────────────────────────────────────────────────────────────────────────────────────── ║\n"
                f"║ 🏆 REWARDS:\n"
                f"║    • Average Reward: {avg_reward:8.4f} │ Max Instant: {episode_max_instant_reward_reached:8.4f}\n"
                f"║    • Baseline Reward Avg: {np.mean(basic_reward_episode):8.4f} │ Best Baseline: {basic_reward_episode[np.argmax(instant_user_rewards)]:8.4f}\n"
                f"║    • Detailed Baseline Reward: {additional_information_best_case}\n"
                f"║ ──────────────────────────────────────────────────────────────────────────────────────────────── ║\n"
                f"║ ⚖️  FAIRNESS:\n"
                f"║    • Average Fairness: {avg_fairness:8.4f} │ Best Reward Fairness: {instant_user_jain_fairness[np.argmax(instant_user_rewards)]:8.4f}\n"
                f"║ ──────────────────────────────────────────────────────────────────────────────────────────────── ║\n"
                f"║ 📊 PERFORMANCE:\n"
                f"║    • Actor Loss: {avg_actor_loss:8.4f} │ Critic Loss: {avg_critic_loss:8.4f}\n"
                f"╚══════════════════════════════════════════════════════════════════════════════════════════════════╝\n"
            )

            # Create ASCII version for file logging
            log_message = (
                f"\n+====================================================================================================+\n"
                f"| [TRAIN] EPISODE #{episode - index_episode_buffer_filled:3d} | Actor Optimizations: {optim_steps_actor:4d} |\n"
                f"+====================================================================================================+\n"
                f"| POSITIONING:\n"
                f"|    User Equipment Positions: {users_position}\n"
                f"|    Eavesdroppers Positions: {eavesdroppers_positions}\n"
                f"| ---------------------------------------------------------------------------------------------------- |\n"
                f"| REWARDS:\n"
                f"|    * Average Reward: {avg_reward:8.4f} | Max Instant: {episode_max_instant_reward_reached:8.4f}\n"
                f"|    * Baseline Reward Avg: {np.mean(basic_reward_episode):8.4f} | Best Baseline: {basic_reward_episode[np.argmax(instant_user_rewards)]:8.4f}\n"
                f"|    * Detailed Baseline Reward: {additional_information_best_case}\n"
                f"| ---------------------------------------------------------------------------------------------------- |\n"
                f"| FAIRNESS:\n"
                f"|    * Average Fairness: {avg_fairness:8.4f} | Best Reward Fairness: {instant_user_jain_fairness[np.argmax(instant_user_rewards)]:8.4f}\n"
                f"| ---------------------------------------------------------------------------------------------------- |\n"
                f"| PERFORMANCE:\n"
                f"|    * Actor Loss: {avg_actor_loss:8.4f} | Critic Loss: {avg_critic_loss:8.4f}\n"
                f"+====================================================================================================+\n"
            )

            # Display console message and log to file
            tqdm.write(console_message)
            logger.verbose(log_message)
        

        """if not training_envs.verbose:
            # Message for printing to the console
            console_message = (
                f"\nTRAINING EPISODE N° {episode - index_episode_buffer_filled} | Optimization Steps Performed: {optim_steps}\n"
                f"--------------------------------------------------------------------------------\n"
                f" REWARDS:\n"
                f"    Average Reward: {avg_reward:.4f} | Max Instant Reward: {episode_max_instant_reward_reached:.4f}\n"
                f"    Average Baseline Reward: {np.mean(basic_reward_episode):.4f} | Baseline Reward for Maximum Instant Reward: {basic_reward_episode[np.argmax(instant_user_rewards)]:.4f}\n"
                f"--------------------------------------------------------------------------------\n"
                f" FAIRNESS:\n"
                f"    Average User Fairness: {avg_fairness:.4f} | User Fairness for Best Instant Reward: {instant_user_jain_fairness[np.argmax(instant_user_rewards)]:.4f}\n"
                f"--------------------------------------------------------------------------------\n"
                f" OTHERS:\n"
                f"    Average Actor Loss: {avg_actor_loss:.4f} | Average Critic Loss: {avg_critic_loss:.4f}\n"
            )

            # Message for logging to a file
            log_message = (
                f"\n+{'=' * 100}+\n"
                f"| TRAINING EPISODE N° {episode - index_episode_buffer_filled} | Optimization Steps Performed: {optim_steps} |\n"
                f"+{'=' * 100}+\n"
                f"|  REWARDS: |\n"
                f"|     Average Reward: {avg_reward:.4f} | Max Instant Reward: {episode_max_instant_reward_reached:.4f} |\n"
                f"|     Average Baseline Reward: {np.mean(basic_reward_episode):.4f} | Baseline Reward for Maximum Instant Reward: {basic_reward_episode[np.argmax(instant_user_rewards)]:.4f} |\n"
                f"+{'=' * 100}+\n"
                f"|  FAIRNESS: |\n"
                f"|     Average User Fairness: {avg_fairness:.4f} | User Fairness for Best Instant Reward: {instant_user_jain_fairness[np.argmax(instant_user_rewards)]:.4f} |\n"
                f"+{'=' * 100}+\n"
                f"|  OTHERS: |\n"
                f"|     Average Actor Loss: {avg_actor_loss:.4f} | Average Critic Loss: {avg_critic_loss:.4f} |\n"
                f"+{'=' * 100}+\n"
            )

            # Log to file
            logger.verbose(log_message)"""


        # Store episode reward and fairness for tracking
        average_reward_per_env[episode] = avg_reward
        smoothed_average_fairness[episode] = avg_fairness
        # Log comprehensive episode metrics to TensorBoard
        writer.add_scalar("General/Time per episode", time.time() - start_episode_time, episode)
        
        # Log reward metrics
        if buffer_filled:
            writer.add_scalar("Rewards/Mean average reward", np.mean(average_reward_per_env[buffer_number_of_required_episode-1:episode+1]), episode)
            writer.add_scalar("Fairness/Smoothed average fairness", np.mean(smoothed_average_fairness[buffer_number_of_required_episode-1:episode+1]), episode)
        writer.add_scalar("Rewards/Max reward reached per episode", episode_max_instant_reward_reached, episode)
        writer.add_scalar("Rewards/Average Reward per episode", avg_reward, episode)
        writer.add_scalar("Rewards/Best average reward per episode", best_average_reward, episode)
        writer.add_scalar("Fairness/Average User Fairness per episode", avg_fairness, episode)

        # Log loss metrics
        writer.add_scalar("Actor Loss/Average actor Loss per episode", avg_actor_loss, episode)
        writer.add_scalar("Critic Loss/Average critic Loss per episode", avg_critic_loss, episode)

        # Log eavesdropper metrics if applicable
        if using_eavesdropper:
            writer.add_scalar("General/Mean total SSR per episode", 
                             np.mean(instant_user_rewards + instant_eavesdropper_rewards), episode)
            writer.add_scalar("General/Max total SSR per episode", 
                             np.max(instant_user_rewards + instant_eavesdropper_rewards), episode)
            writer.add_scalar("Eavesdropper/Total Reward per episode", total_eavesdropper_reward, episode)
            writer.add_scalar("Eavesdropper/Average reward per episode", avg_eavesdropper_reward, episode)

        # Log team average metrics for non-solo episodes
        if not solo_episode:
            if len(team_user_rewards) > 0:
                writer.add_scalar("Team Metrics/Average User Team Reward", np.mean(team_user_rewards), episode)
                writer.add_scalar("Team Metrics/Std User Team Reward", np.std(team_user_rewards), episode)
                if using_eavesdropper and len(team_eavesdropper_rewards) > 0:
                    writer.add_scalar("Team Metrics/Average Eavesdropper Team Reward", np.mean(team_eavesdropper_rewards), episode)
                    writer.add_scalar("Team Metrics/Std Eavesdropper Team Reward", np.std(team_eavesdropper_rewards), episode)
            
            if len(team_avg_signal_strengths['min']) > 0:
                writer.add_scalar("Team Metrics/Average Min Signal Strength (dBm)", 
                                 np.mean(team_avg_signal_strengths['min']), episode)
                writer.add_scalar("Team Metrics/Average Max Signal Strength (dBm)", 
                                 np.mean(team_avg_signal_strengths['max']), episode)


        """np.savez(f'{log_dir}/data/test_W_matrix.npz',
                 test_W_matrix = best_W,
                 test_theta = best_Theta )
        
        W_message = (
            f" \n The final W matrix is: \n {best_W} \n"
            f" \n The final Theta matrix is: \n {best_Theta} \n"
        )

        logger.verbose(W_message)"""

        # ========================================================================
        # EVALUATION PHASE
        # ========================================================================
        # Conduct periodic evaluation if specified and buffer is filled
        if conduct_eval and episode % eval_period == 0 and buffer_filled:
            tqdm.write(f" \n Conducting Evaluation \n")
        
            # Initialize renderer for evaluation visualization if enabled
            if use_rendering:
                renderer = SituationRenderer(
                    M=eval_env.M,
                    L=eval_env.num_eavesdroppers,
                    lambda_h=eval_env.lambda_h,
                    N_t=training_envs.BS_transmit_antennas,
                    max_step_per_episode=training_config.get("max_steps_per_episode", 20000),
                    BS_position=training_envs.BS_position,
                    d_h=eval_env.d_h,
                    RIS_position=eval_env.RIS_position,
                    users_position=eval_env.get_users_positions(),
                    eavesdroppers_positions=eavesdroppers_positions,
                    eavesdropper_moving=eval_env.moving_eavesdroppers
                )


            # Initialize evaluation tracking variables
            num_eval_periode += 1
            eval_current_step = 0

            # Allocate arrays for per-episode evaluation metrics
            all_episode_rewards = np.zeros(episode_per_eval)
            all_episode_fairness = np.zeros(episode_per_eval)
            all_episode_max_rewards = np.zeros(episode_per_eval)
            all_episode_best_fairness = np.zeros(episode_per_eval)

            # Initialize eavesdropper reward tracking if applicable
            if using_eavesdropper:
                all_episode_eaves_rewards = np.zeros(episode_per_eval)


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

                # Create position logging message for evaluation episode
                if using_eavesdropper:
                    message = (
                        f"\nStarting positions for EVAL episode {eval_episode}:\n"
                        f"Users: {eval_env.get_users_positions()}\n"
                        f"Eavesdroppers: {eval_env.get_eavesdroppers_positions()}\n"
                    )
                else:
                    message = (
                        f"\nStarting positions for EVAL episode {eval_episode}:\n"
                        f"Users: {eval_env.get_users_positions()}\n"
                    )

                # Log evaluation episode starting positions
                logger.verbose(message)
                
                # Initialize evaluation episode tracking arrays
                episode_user_rewards = np.zeros(max_num_step_per_episode)
                episode_user_fairness = np.zeros(max_num_step_per_episode)

                if using_eavesdropper:
                    episode_eaves_rewards = np.zeros(max_num_step_per_episode)

                # ================================================================
                # EVALUATION STEP LOOP
                # ================================================================
                for num_step in range(max_num_step_per_episode):
                    eval_current_step += 1

                    # Get current state and select action (no exploration noise)
                    state = eval_env.get_states()[0] if num_step == 0 else next_state[0]
                    selected_action = network.select_action(state)
                    state, selected_action, reward, next_state = eval_env.step(state, selected_action)

                    # Process reward and fairness information
                    reward_value = reward.item()
                    fairness_value = eval_env.get_user_jain_fairness()

                    # Track eavesdropper rewards if applicable
                    if using_eavesdropper:
                        eavesdropper_reward = eval_env.get_eavesdropper_rewards()
                        episode_eaves_rewards[num_step] = eavesdropper_reward

                    # Store evaluation metrics
                    episode_user_rewards[num_step] = reward_value
                    episode_user_fairness[num_step] = fairness_value

                # Store evaluation episode metrics
                all_episode_rewards[eval_episode] = np.mean(episode_user_rewards)
                all_episode_fairness[eval_episode] = np.mean(episode_user_fairness)
                all_episode_max_rewards[eval_episode] = np.max(episode_user_rewards)
                all_episode_best_fairness[eval_episode] = episode_user_fairness[np.argmax(episode_user_rewards)]

                if using_eavesdropper:
                    all_episode_eaves_rewards[eval_episode] = np.mean(episode_eaves_rewards)


            # Calculate comprehensive evaluation metrics across all episodes
            mean_reward = np.mean(all_episode_rewards)
            std_reward = np.std(all_episode_rewards)
            mean_fairness = np.mean(all_episode_fairness)
            std_fairness = np.std(all_episode_fairness)
            mean_max_reward = np.mean(all_episode_max_rewards)
            mean_best_fairness = np.mean(all_episode_best_fairness)


            # Calculate eavesdropper-specific metrics if applicable
            if using_eavesdropper:
                mean_eaves_reward = np.mean(all_episode_eaves_rewards)
                std_eaves_reward = np.std(all_episode_eaves_rewards)
                reward_eaves_ratio = all_episode_rewards / (all_episode_eaves_rewards + 1e-6)
                mean_ratio = np.mean(reward_eaves_ratio)
                composite_score = all_episode_rewards - 0.5 * all_episode_eaves_rewards
                mean_composite_score = np.mean(composite_score)

                # Log eavesdropper-specific evaluation metrics
                writer.add_scalar("Evaluation/Mean total SSR per episode", mean_reward + mean_eaves_reward, episode)
                writer.add_scalar("Evaluation/Max total SSR per episode", 
                                 np.max(all_episode_rewards + all_episode_eaves_rewards), episode)
                writer.add_scalar("Evaluation/Total Eavesdropper Reward per episode", mean_eaves_reward, episode)
                writer.add_scalar("Evaluation/Reward-to-Eavesdropper Ratio", mean_ratio, episode)
                writer.add_scalar("Evaluation/Composite Score", mean_composite_score, episode)

            else:
                # Log standard evaluation metrics without eavesdroppers
                writer.add_scalar("Evaluation/Mean total SSR per episode", mean_reward, episode)
                writer.add_scalar("Evaluation/Max total SSR per episode", np.max(all_episode_rewards), episode)

            # Log standard evaluation metrics
            writer.add_scalar("Evaluation/Average Fairness", mean_fairness, episode)
            writer.add_scalar("Evaluation/Fairness for the best reward", mean_best_fairness, episode)
            writer.add_scalar("Evaluation/Max reward reached per episode", mean_max_reward, episode)
            writer.add_scalar("Evaluation/Total Reward per episode", mean_reward, episode)

            # Create and log evaluation summary message
            if using_eavesdropper:
                message = (
                    f"\n[EVAL PERIOD {num_eval_periode}] "
                    f"Avg Reward: {mean_reward:.4f}, Max Reward: {mean_max_reward:.4f},\n"
                    f"Fairness for Max Reward: {mean_best_fairness:.4f}, Avg Fairness: {mean_fairness:.4f},\n"
                    f"Eavesdropper Reward: {mean_eaves_reward:.4f}, Reward/Eavesdropper Ratio: {mean_ratio:.4f},\n"
                    f"Composite Score: {mean_composite_score:.4f}\n"
                )
            else:
                message = (
                    f"\n[EVAL PERIOD {num_eval_periode}] "
                    f"Avg Reward: {mean_reward:.4f}, Max Reward: {mean_max_reward:.4f},\n"
                    f"Fairness for Max Reward: {mean_best_fairness:.4f}, Avg Fairness: {mean_fairness:.4f},\n"
                )
                
            # Log evaluation summary to file and display in console
            logger.verbose(message)
            tqdm.write(message)
