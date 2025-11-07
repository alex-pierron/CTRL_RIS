"""
Single-process training loop adapted for ON-POLICY PPO with comprehensive metrics and logging.

This file provides:
 - setup_logger: configure file and console logging with a custom VERBOSE level
 - onp_single_process_trainer: main training loop adapted to work with PPO (on-policy)

Design notes:
 - Uses the network's rollout buffer instead of an off-policy replay buffer.
 - Stores (state, action, raw_action, reward, done, value, logprob) at every step via network.store_transition
 - Calls network.training() when the rollout is full or when an episode ends (to handle episode boundaries)
 - Resets the rollout after each training() call so the next collection starts fresh.
 - Comprehensive metrics logging matching the off-policy trainer
 - Advanced rendering and evaluation capabilities

Better Comments legend used throughout:
 - TODO: future improvements or refactors (non-functional here)
 - NOTE: important behavior or design intent
 - !: important runtime remark
 - ?: questioning a choice or highlighting an assumption
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
from src.runners.on_policy.onp_single_process_trainer_helpers import (
    log_position_message,
    create_episode_summary_messages
)


# Define a new log level (between DEBUG=10 and INFO=20)
VERBOSE_LEVEL = 15
logging.addLevelName(VERBOSE_LEVEL, "VERBOSE")

# Custom log method
def verbose(self, message, *args, **kwargs):
    if self.isEnabledFor(VERBOSE_LEVEL):
        self._log(VERBOSE_LEVEL, message, args, **kwargs)

# Attach the method to the Logger class
logging.Logger.verbose = verbose


def setup_logger(log_dir, log_filename='training.log'):
    """
    Set up the logger to write logs to a file while keeping the terminal clean.

    Parameters:
    - log_dir: Directory for logging.
    - log_filename: Name of the log file.

    Returns:
    - logger: Configured logger object.
    """
    # NOTE: We isolate logging under a named logger and disable propagation to
    # avoid duplicate messages when imported within larger applications.
    logger = logging.getLogger('TrainingLogger')
    logger.setLevel(logging.DEBUG)  # Set the base logging level

    # Prevent propagation to the root logger
    logger.propagate = False
    
    # Remove any existing handlers to avoid duplication
    if logger.hasHandlers():
        logger.handlers.clear()

    # File handler (logs everything)
    file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
    file_handler.setLevel(logging.DEBUG)

    # Console handler (logs only WARNING and above, keeps INFO logs out)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)  # Ignore INFO messages

    # Formatter for both handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def onp_single_process_trainer(training_envs, network, training_config, log_dir, writer,
                           eval_env=None, action_noise_activated=False, # Note: action_noise_activated is unused for PPO
                           batch_instead_of_buff=False, use_rendering=False):
    """
    Single-process on-policy trainer for PPO-like agents with comprehensive metrics and logging.

    Parameters:
        training_envs: The environment object.
        network: The neural network object.
        training_config: Configuration dictionary for training parameters.
        log_dir: Directory for logging.
        writer: TensorBoard SummaryWriter object for logging.
        eval_env: Optional evaluation environment.
        action_noise_activated: Boolean flag (unused for PPO, kept for compatibility).
        batch_instead_of_buff: Boolean flag (unused for on-policy, kept for compatibility).
        use_rendering: Boolean to know if a rendering tool is needed.
    """
    # ========================================================================
    # INITIALIZATION
    # ========================================================================
    logger = setup_logger(log_dir)
    env_config = training_envs.env_config

    # Training configuration
    debugging = training_config.get("debugging", False)
    num_episode = training_config.get("number_episodes", 15)
    max_num_step_per_episode = training_config.get("max_steps_per_episode", 20000)
    reward_smoothing_factor = training_config.get("reward_smoothing_factor", 0.5)
    frequency_information = training_config.get("frequency_information", 500)
    saving_frequency = training_config.get("network_save_checkpoint_frequency", 100)
    plot_saving_frequency = training_config.get("plot_save_checkpoint_frequency", 100)
    curriculum_learning = training_config.get("Curriculum_Learning", True)
    
    # Evaluation configuration
    conduct_eval = eval_env is not None
    episode_per_eval = training_config.get("episode_per_eval_env", 1) if conduct_eval else None
    eval_period = training_config.get("eval_period", 1) if conduct_eval else None
    
    # Environment metadata
    num_users = training_envs.num_users
    using_eavesdropper = (training_envs.num_eavesdroppers > 0)

    # Curriculum learning setup
    if curriculum_learning:
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

    # Rollout buffer configuration
    rollout_size = getattr(network.rollout, 'buffer_size', training_config.get('rollout_size', 2048))

    # Training state tracking
    best_average_reward = -np.inf
    total_steps = 0
    optim_count = 0
    
    # Trajectory tracking
    users_trajectory = np.zeros((num_episode, training_envs.num_users, 2))
    eavesdroppers_trajectory = np.zeros((num_episode, training_envs.num_eavesdroppers, 2))
    average_reward_per_env = np.zeros(num_episode)

    # ========================================================================
    # MAIN TRAINING LOOP
    # ========================================================================
    for episode in tqdm(range(num_episode), 
                       desc="[TRAIN] TRAINING", 
                       position=1, 
                       bar_format="{l_bar}{bar:50}{r_bar}{bar:-10b}",
                       ncols=140,
                       colour='magenta',
                       ascii="▏▎▍▌▋▊▉█"):
        start_episode_time = time.time()

        if curriculum_learning:
            difficulty_config = Task_Manager.generate_episode_configs()
            training_envs.reset(difficulty_config)
        else:
            training_envs.reset()

        users_position = training_envs.get_users_positions()
        users_trajectory[episode] = users_position

        if using_eavesdropper:
            eavesdroppers_positions = training_envs.get_eavesdroppers_positions()
            eavesdroppers_trajectory[episode] = eavesdroppers_positions
            
            episode_best_4_eavesdroppers_power_patterns = {
                "downlink_power_patterns": [],
                "uplink_power_patterns": [],
                "W_power_patterns": [],
                "rewards": [],
                "steps": [],
            }
        else:
            eavesdroppers_positions = None

        # Log starting positions
        log_position_message(logger, episode, users_position, eavesdroppers_positions, using_eavesdropper)

        # Initialize arrays to track rewards and losses for the episode
        instant_user_rewards = np.zeros(max_num_step_per_episode) - np.inf
        instant_eavesdropper_rewards = np.zeros(max_num_step_per_episode)
        instant_user_jain_fairness = np.zeros(max_num_step_per_episode)
        basic_reward_episode = np.zeros(max_num_step_per_episode)
        
        # Additional tracking variables for comprehensive logging
        average_rewards = np.zeros(max_num_step_per_episode + 1)
        paper_average_rewards = np.zeros(max_num_step_per_episode + 1)
        total_reward = 0
        additional_information_best_case = 0
        step_time_list = []
        
        avg_actor_loss_episode = 0
        avg_critic_loss_episode = 0

        ppo_epochs = network.ppo_epochs

        optim_steps_epoch = 0

        max_episode_reward = -np.inf
        max_episode_eavesdropper_reward = 0
        
        # Initialize renderer for the episode if rendering is activated
        if use_rendering:
            # NOTE: Renderer collects visualization artifacts for later export
            renderer = SituationRenderer(
                M=training_envs.M,
                L=training_envs.num_eavesdroppers,
                N_t=training_envs.BS_transmit_antennas,
                lambda_h=training_envs.lambda_h,
                max_step_per_episode=max_num_step_per_episode,
                BS_position=training_envs.BS_position,
                d_h=training_envs.d_h,
                RIS_position=training_envs.RIS_position,
                users_position=users_position,
                eavesdroppers_positions=eavesdroppers_positions,
                eavesdropper_moving=training_envs.moving_eavesdroppers,
                num_frames=200
            )

            episode_best_power_patterns = {
                "downlink_power_patterns": [],
                "uplink_power_patterns": [],
                "W_power_patterns": [],
                "rewards": [],
                "steps": [],
            }

        for num_step in range(max_num_step_per_episode):
            current_step = episode * max_num_step_per_episode + num_step

            # Get the current state from the environment
            state = training_envs.get_states()[0] if num_step == 0 else next_state[0]

            # NOTE: PPO explores by sampling from the policy (no noise needed)
            processed_action, logprob, raw_action = network.select_action(state, eval_mode=False)
            _, _, reward, next_state = training_envs.step(state, processed_action)
            done = bool(training_envs.is_done()) if hasattr(training_envs, 'is_done') else False

            # NOTE: Store the transition in the rollout buffer
            network.store_transition(state, processed_action, raw_action, float(reward), 
                                     next_state, done=done, logprob=logprob)

            reward_value = float(reward)
            instant_user_rewards[num_step] = reward_value
            instant_user_jain_fairness[num_step] = training_envs.get_user_jain_fairness()
            basic_reward_episode[num_step] = training_envs.get_basic_reward()
            
            # Track cumulative rewards and averages
            total_reward = np.sum(instant_user_rewards[instant_user_rewards > -np.inf])
            avg_reward = total_reward / (num_step + 1)
            average_rewards[num_step] = avg_reward
            if num_step > 1:
                paper_average_rewards[num_step] = reward_smoothing_factor * paper_average_rewards[num_step - 1] + (1 - reward_smoothing_factor) * reward_value
            
            if using_eavesdropper:
                instant_eavesdropper_rewards[num_step] = training_envs.get_eavesdropper_rewards()

            # NOTE: Track best rewards for rendering
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

            elif reward_value > max_episode_reward:
                max_episode_reward = round(reward_value, 4)
                additional_information_best_case = training_envs.get_additionnal_informations()

            # Track and render best eavesdropper rewards
            if using_eavesdropper and instant_eavesdropper_rewards[num_step] > max_episode_eavesdropper_reward and use_rendering:
                max_episode_eavesdropper_reward = round(instant_eavesdropper_rewards[num_step], 4)
                eave_max_W_power_patterns = training_envs.get_W_power_patterns()
                eave_max_downlink_power_patterns = training_envs.get_downlink_power_patterns()
                eave_max_uplink_power_patterns = training_envs.get_uplink_power_patterns()

                episode_best_4_eavesdroppers_power_patterns["rewards"].append(deepcopy(max_episode_eavesdropper_reward))
                episode_best_4_eavesdroppers_power_patterns["steps"].append(deepcopy(num_step))
                episode_best_4_eavesdroppers_power_patterns["W_power_patterns"].append(deepcopy(eave_max_W_power_patterns))
                episode_best_4_eavesdroppers_power_patterns["downlink_power_patterns"].append(deepcopy(eave_max_downlink_power_patterns))
                episode_best_4_eavesdroppers_power_patterns["uplink_power_patterns"].append(deepcopy(eave_max_uplink_power_patterns))

            # NOTE: Train the network when rollout buffer is full or episode ends
            if network.rollout.ptr >= rollout_size or (done and network.rollout.ptr > 0):
                training_time_1 = time.time()
                actor_loss, critic_loss, mean_reward = network.training()
                step_time_list.append(time.time() - training_time_1)
                avg_actor_loss_episode += actor_loss
                avg_critic_loss_episode += critic_loss
                optim_steps_epoch += ppo_epochs

                optim_count += ppo_epochs

                writer.add_scalar("Training/Actor loss (per update)", actor_loss, total_steps)
                writer.add_scalar("Training/Critic loss (per update)", critic_loss, total_steps)
                writer.add_scalar("Training/Mean reward from rollout", mean_reward, total_steps)

                # Reset rollout buffer
                try:
                    network.rollout.reset()
                except Exception:
                    if hasattr(network.rollout, 'ptr'):
                        network.rollout.ptr = 0

            # NOTE: Log training information at specified frequency
            if (total_steps + 1) % frequency_information == 0 and num_step > 0:
                W = training_envs.get_W()
                total_power_deployed = round(np.trace(np.diag(np.diag(W @ W.conj().T)).real), 4)

                # Calculate local metrics over the last `frequency_information` steps
                local_rewards = instant_user_rewards[max(0, num_step + 1 - frequency_information): num_step + 1]
                valid_local_rewards = local_rewards[local_rewards > -np.inf]
                local_average_reward = np.mean(valid_local_rewards) if len(valid_local_rewards) > 0 else 0.0
                
                local_basic_rewards = basic_reward_episode[max(0, num_step + 1 - frequency_information): num_step + 1]
                local_average_basic_reward = np.mean(local_basic_rewards)
                
                local_fairness = instant_user_jain_fairness[max(0, num_step + 1 - frequency_information): num_step + 1]
                local_user_fairness = round(np.mean(local_fairness), ndigits=4)

                # Log comprehensive TensorBoard metrics
                writer.add_scalar("Rewards/Local Average Reward", local_average_reward, total_steps)
                writer.add_histogram("Rewards/Paper Average Reward", paper_average_rewards[max(0, num_step + 1 - frequency_information): num_step + 1], total_steps)
                writer.add_scalar("Rewards/Global Average Reward", avg_reward, total_steps)
                writer.add_scalar("Rewards/Max Instant Reward", np.max(instant_user_rewards), total_steps)
                writer.add_scalar("Rewards/Local Average Baseline Reward", local_average_basic_reward, total_steps)
                writer.add_histogram("Rewards/Instant reward", instant_user_rewards[max(0, num_step + 1 - frequency_information): num_step + 1], total_steps)

                # Calculate current average losses
                current_avg_actor_loss = avg_actor_loss_episode / max(1, optim_steps_epoch) if optim_steps_epoch > 0 else 0.0
                current_avg_critic_loss = avg_critic_loss_episode / max(1, optim_steps_epoch) if optim_steps_epoch > 0 else 0.0
                
                # Log loss metrics (only if we have training steps)
                if optim_steps_epoch > 0:
                    writer.add_scalar("Actor Loss/Current average actor loss", current_avg_actor_loss, total_steps)
                    writer.add_scalar("Critic Loss/Current average critic loss", current_avg_critic_loss, total_steps)
                
                writer.add_scalar("Fairness/User local average Fairness", local_user_fairness, total_steps)
                writer.add_scalar("General/Power deployed (Watts)", total_power_deployed, total_steps)

                if using_eavesdropper:
                    reward_combined = instant_user_rewards + instant_eavesdropper_rewards
                    local_average_reward_combined = np.mean(reward_combined[max(0, num_step + 1 - frequency_information): num_step + 1])
                    writer.add_scalar("General/Local average total SSR", local_average_reward_combined, total_steps)

                    local_eaves_rewards = instant_eavesdropper_rewards[max(0, num_step + 1 - frequency_information): num_step + 1]
                    local_average_eavesdropper_reward = np.mean(local_eaves_rewards)
                    writer.add_scalar("Eavesdropper/Local average reward", local_average_eavesdropper_reward, total_steps)
                    writer.add_histogram("Eavesdropper/Instant reward", instant_eavesdropper_rewards[max(0, num_step + 1 - frequency_information): num_step + 1], total_steps)

                message = (
                    f"\n|--> ON-POLICY TRAINING EPISODE {episode}, STEP {num_step + 1}\n"
                    f"     |--> Training the NN takes {np.mean(step_time_list):.4f} sec on average across {len(step_time_list)} steps \n"
                    f"  |--> LOCAL AVERAGE REWARD {np.float16(local_average_reward)}, MAX INSTANT REWARD REACHED {np.max(instant_user_rewards)}\n"
                    f"  |--> LOCAL AVERAGE BASIC REWARD {np.float16(local_average_basic_reward)}, MAXIMUM INSTANT BASIC REWARD: {basic_reward_episode[np.argmax(instant_user_rewards)]:.4f}\n"
                    f"  |--> Detailed basic reward for best case: {additional_information_best_case}\n"       
                    f" |--> ACTOR LOSS {current_avg_actor_loss:.4f}, CRITIC LOSS {current_avg_critic_loss:.4f}\n"
                    f" |--> USER FAIRNESS FOR THE BEST INSTANT REWARD {instant_user_jain_fairness[np.argmax(instant_user_rewards)]}, LOCAL USER FAIRNESS {local_user_fairness}\n"
                    f" |--> POWER DEPLOYED: {total_power_deployed} Watts\n"
                    f"---------------------------\n"
                )

                if using_eavesdropper:
                    message += f"  |--> LOCAL EAVESDROPPER REWARD: {local_average_eavesdropper_reward:.4f}\n---------------------------\n"
                
                logger.verbose(message)
            
            if done:
                break

            total_steps += 1

        # === Episode summary metrics ===
        # NOTE: Calculate average losses and rewards for the episode
        valid_rewards = instant_user_rewards[instant_user_rewards > -np.inf]
        avg_reward_episode = np.mean(valid_rewards) if len(valid_rewards) > 0 else 0.0
        avg_fairness_episode = np.mean(instant_user_jain_fairness[:len(valid_rewards)])
        avg_actor_loss = avg_actor_loss_episode / max(1, optim_steps_epoch)
        avg_critic_loss = avg_critic_loss_episode / max(1, optim_steps_epoch)
        
        writer.add_scalar("Rewards/Average Reward per episode", avg_reward_episode, episode)
        writer.add_scalar("Rewards/Max reward reached per episode", max_episode_reward, episode)
        writer.add_scalar("Fairness/Average User Fairness per episode", avg_fairness_episode, episode)
        writer.add_scalar("Loss/Average Actor Loss per episode", avg_actor_loss, episode)
        writer.add_scalar("Loss/Average Critic Loss per episode", avg_critic_loss, episode)
        
        if using_eavesdropper:
            avg_eaves_reward = np.mean(instant_eavesdropper_rewards[:len(valid_rewards)])
            writer.add_scalar("Eavesdropper/Average reward per episode", avg_eaves_reward, episode)

        # NOTE: Save the best model based on average reward
        if avg_reward_episode > best_average_reward and not debugging:
            best_average_reward = avg_reward_episode
            saving_model_directory = f"{log_dir}/model_saved/best_model"
            network.save_models(saving_model_directory)
            logger.info(f"Saved new best model at episode {episode}, avg_reward={avg_reward_episode:.4f}")
            
            if use_rendering and len(episode_best_power_patterns["steps"]) > 0:
                renderer.plotly_single_image(
                    max_W_power_patterns, max_downlink_power_patterns, max_uplink_power_patterns,
                    reward=max_episode_reward, step_number=episode_best_power_patterns["steps"][-1],
                    episode=episode, save_dir=f"{log_dir}/plots/best_model_plots",
                    name_file=f"Ep_{episode}_Best_User_Profile.html"
                )
                renderer.animate_power_patterns(
                    episode_best_power_patterns, save_dir=f"{log_dir}/animations/best_model_anims",
                    name_file=f"episode_{episode}_best_profiles.mp4"
                )

        # NOTE: Save plots at designated frequency to visualize system evolution
        if episode % plot_saving_frequency == 0 and not debugging:
            if use_rendering and len(episode_best_power_patterns["steps"]) > 0:
                renderer.plotly_single_image(
                    max_W_power_patterns, max_downlink_power_patterns, max_uplink_power_patterns,
                    reward=max_episode_reward, step_number=episode_best_power_patterns["steps"][-1],
                    episode=episode, save_dir=f"{log_dir}/plots/episode_{episode}",
                    name_file=f"Ep_{episode}_Best_User_Profile.html"
                )

                renderer.render_situation(save_dir=f"{log_dir}/animations/checkpoints", name_file=f"episode_{episode}_dynamic_profile.mp4")
                renderer.animate_power_patterns(
                    episode_best_power_patterns, save_dir=f"{log_dir}/animations/checkpoints",
                    name_file=f"episode_{episode}_best_profiles.mp4"
                )

        # NOTE: Save checkpoint at specified frequency
        if episode % saving_frequency == 0 and not debugging:
            ckpt_dir = f"{log_dir}/model_saved/ckpt_episode_{episode}"
            network.save_models(ckpt_dir)
            logger.info(f"Saved checkpoint for episode {episode}")

        # Create and display episode summary messages
        episode_max_instant_reward_reached = max(instant_user_rewards) if len(valid_rewards) > 0 else 0.0
        best_fairness = instant_user_jain_fairness[np.argmax(instant_user_rewards)] if len(valid_rewards) > 0 else 0.0
        best_eaves_reward = instant_eavesdropper_rewards[np.argmax(instant_user_rewards)] if using_eavesdropper and len(valid_rewards) > 0 else None
        
        console_message, log_message = create_episode_summary_messages(
            episode=episode,
            optim_steps=optim_count,
            users_position=users_position,
            eavesdroppers_positions=eavesdroppers_positions,
            avg_reward=avg_reward_episode,
            max_reward=episode_max_instant_reward_reached,
            avg_fairness=avg_fairness_episode,
            best_fairness=best_fairness,
            avg_actor_loss=avg_actor_loss,
            avg_critic_loss=avg_critic_loss,
            basic_reward_episode=basic_reward_episode,
            instant_user_rewards=instant_user_rewards,
            additional_information_best_case=additional_information_best_case,
            using_eavesdropper=using_eavesdropper,
            avg_eaves_reward=avg_eaves_reward if using_eavesdropper else None,
            best_eaves_reward=best_eaves_reward
        )
        
        tqdm.write(console_message)
        logger.verbose(log_message)

        average_reward_per_env[episode] = avg_reward_episode

        writer.add_scalar("General/Time per episode", time.time() - start_episode_time, episode)
        writer.add_scalar("Rewards/Mean average reward", np.mean(average_reward_per_env[:episode+1]), episode)
        writer.add_scalar("Rewards/Best average reward per episode", best_average_reward, episode)

        if using_eavesdropper:
            writer.add_scalar("General/Mean total SSR per episode", np.mean(instant_user_rewards + instant_eavesdropper_rewards), episode)
            writer.add_scalar("General/Max total SSR per episode", np.max(instant_user_rewards + instant_eavesdropper_rewards), episode)

        # * CONDUCT EVALUATION IF SPECIFIED 
        if conduct_eval and episode % eval_period == 0:
            tqdm.write("--- Conducting Evaluation ---")
            all_episode_rewards = np.zeros(episode_per_eval)
            all_episode_fairness = np.zeros(episode_per_eval)
            all_episode_max_rewards = np.zeros(episode_per_eval)
            all_episode_best_fairness = np.zeros(episode_per_eval)

            if using_eavesdropper:
                all_episode_eaves_rewards = np.zeros(episode_per_eval)

            # NOTE: Loop over evaluation episodes
            for eval_episode in tqdm(range(episode_per_eval), 
                                   desc="[EVAL] EVALUATION", 
                                   position=2, 
                                   bar_format="{l_bar}{bar:50}{r_bar}{bar:-10b}",
                                   ncols=140,
                                   colour='red',
                                   ascii="▏▎▍▌▋▊▉█"):
                eval_env.reset()
                episode_user_rewards = np.zeros(max_num_step_per_episode)
                episode_user_fairness = np.zeros(max_num_step_per_episode)

                if using_eavesdropper:
                    episode_eaves_rewards = np.zeros(max_num_step_per_episode)

                # NOTE: Loop over steps within the evaluation episode
                for num_step in range(max_num_step_per_episode):
                    state = eval_env.get_states()[0] if num_step == 0 else next_state[0]
                    action, _, _ = network.select_action(state, eval_mode=True)
                    _, _, reward, next_state = eval_env.step(state, action)
                    
                    reward_value = float(reward)
                    fairness_value = eval_env.get_user_jain_fairness()

                    if using_eavesdropper:
                        eavesdropper_reward = eval_env.get_eavesdropper_rewards()
                        episode_eaves_rewards[num_step] = eavesdropper_reward

                    episode_user_rewards[num_step] = reward_value
                    episode_user_fairness[num_step] = fairness_value

                    if eval_env.is_done():
                        break

                # Store per-episode metrics
                all_episode_rewards[eval_episode] = np.mean(episode_user_rewards)
                all_episode_fairness[eval_episode] = np.mean(episode_user_fairness)
                all_episode_max_rewards[eval_episode] = np.max(episode_user_rewards)
                all_episode_best_fairness[eval_episode] = episode_user_fairness[np.argmax(episode_user_rewards)]

                if using_eavesdropper:
                    all_episode_eaves_rewards[eval_episode] = np.mean(episode_eaves_rewards)

            # NOTE: Compute evaluation metrics across evaluation episodes
            mean_reward = np.mean(all_episode_rewards)
            mean_fairness = np.mean(all_episode_fairness)
            mean_max_reward = np.mean(all_episode_max_rewards)
            mean_best_fairness = np.mean(all_episode_best_fairness)

            if using_eavesdropper:
                mean_eaves_reward = np.mean(all_episode_eaves_rewards)
                reward_eaves_ratio = all_episode_rewards / (all_episode_eaves_rewards + 1e-6)
                mean_ratio = np.mean(reward_eaves_ratio)
                composite_score = all_episode_rewards - 0.5 * all_episode_eaves_rewards
                mean_composite_score = np.mean(composite_score)

                writer.add_scalar("Evaluation/Mean total SSR per episode", mean_reward + mean_eaves_reward, episode)
                writer.add_scalar("Evaluation/Max total SSR per episode", np.max(all_episode_rewards + all_episode_eaves_rewards), episode)
                writer.add_scalar("Evaluation/Total Eavesdropper Reward per episode", mean_eaves_reward, episode)
                writer.add_scalar("Evaluation/Reward-to-Eavesdropper Ratio", mean_ratio, episode)
                writer.add_scalar("Evaluation/Composite Score", mean_composite_score, episode)
            else:
                writer.add_scalar("Evaluation/Mean total SSR per episode", mean_reward, episode)
                writer.add_scalar("Evaluation/Max total SSR per episode", np.max(all_episode_rewards), episode)

            writer.add_scalar("Evaluation/Average Fairness", mean_fairness, episode)
            writer.add_scalar("Evaluation/Fairness for the best reward", mean_best_fairness, episode)
            writer.add_scalar("Evaluation/Max reward reached per episode", mean_max_reward, episode)
            writer.add_scalar("Evaluation/Total Reward per episode", mean_reward, episode)

            # NOTE: Log evaluation summary
            if using_eavesdropper:
                message = (
                    f"\n[EVAL PERIOD] "
                    f"Avg Reward: {mean_reward:.4f}, Max Reward: {mean_max_reward:.4f},\n"
                    f"Fairness for Max Reward: {mean_best_fairness:.4f}, Avg Fairness: {mean_fairness:.4f},\n"
                    f"Eavesdropper Reward: {mean_eaves_reward:.4f}, Reward/Eavesdropper Ratio: {mean_ratio:.4f},\n"
                    f"Composite Score: {mean_composite_score:.4f}\n"
                )
            else:
                message = (
                    f"\n[EVAL PERIOD] "
                    f"Avg Reward: {mean_reward:.4f}, Max Reward: {mean_max_reward:.4f},\n"
                    f"Fairness for Max Reward: {mean_best_fairness:.4f}, Avg Fairness: {mean_fairness:.4f},\n"
                )
                
            logger.verbose(message)
            tqdm.write(message)

    logger.info("Training loop finished.")