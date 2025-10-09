"""
Multiprocess training loop utilities for RIS Duplex RL experiments with PPO.

This module provides:
 - setup_logger: configure file and console logging with a custom VERBOSE level
 - onp_multiprocess_trainer: the main loop coordinating vectorized environments for on-policy PPO

Design notes:
 - Uses the network's rollout buffer instead of an off-policy replay buffer.
 - Stores (state, action, raw_action, reward, done, value, logprob) at every step via network.store_transition
 - Calls network.training() when the rollout is full or when an episode ends
 - Resets the rollout after each training() call so the next collection starts fresh.
 - Comprehensive metrics logging matching the off-policy trainer
 - Same data registration and logging as ofp_multiprocess_trainer.py

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
import torch
from tqdm import tqdm
import numpy as np
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter
from src.environment.tools import parse_args, parse_config, write_line_to_file, SituationRenderer,TaskManager
from src.environment.multiprocessing import pickable_to_dict


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
    # NOTE: Named logger with no propagation prevents duplicate logs in larger apps
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


def onp_multiprocess_trainer(training_envs, network, training_config, log_dir, writer,
                         action_noise_activated=False,
                         batch_instead_of_buff = False,
                         eval_env = None,
                         use_rendering = False):
    """
    Executes the on-policy PPO training loop for the network with improved logging in a multiprocess setup.

    Parameters:
    - training_envs: The environment object, here a DummyVecEnv.
    - network: The neural network object (PPO with rollout buffer).
    - training_config: Configuration dictionary for training parameters.
    - log_dir: Directory for logging.
    - writer: TensorBoard SummaryWriter object for logging.
    - action_noise_activated: Boolean flag (unused for PPO, kept for compatibility).
    - batch_instead_of_buff: Boolean flag (unused for on-policy, kept for compatibility).
    - eval_env: Optional evaluation environment.
    - use_rendering: Boolean to know if a rendering tool is needed.
    """
    # NOTE: Dedicated logger for this run; VERBOSE level goes to file only
    logger = setup_logger(log_dir)

    env_config = training_envs.env_config

    n_rollout_envs = training_envs.nenvs
    
    # Extract training configuration parameters
    debugging = training_config.get("debugging", False)
    num_episode = training_config.get("number_episodes", 15)
    max_num_step_per_episode = training_config.get("max_steps_per_episode", 20000)
    frequency_information = training_config.get("frequency_information", 500)
    conduct_eval = eval_env is not None
    episode_per_eval = training_config.get("episode_per_eval_env", 1) if conduct_eval else None
    eval_period = training_config.get("eval_period", 1) if conduct_eval else None

    curriculum_learning = training_config.get("Curriculum_Learning", True)  # ! WIP: difficulty scheduling

    num_users = env_config
    saving_frequency = training_config.get("network_save_checkpoint_frequency", 100)

    ppo_epochs = network.ppo_epochs
    n_eval_rollout_threads = training_config.get("n_eval_rollout_threads", 2)
    
    # NOTE: For PPO, we use rollout buffer instead of replay buffer
    rollout_size = getattr(network.rollout, 'buffer_size', training_config.get('rollout_size', 2048))
    buffer_size = rollout_size * n_rollout_envs # Use rollout size for compatibility with existing code
    
    decisive_reward_functions = env_config.get("decisive_reward_functions", ["basic_reward"])
    informative_reward_functions = env_config.get("informative_reward_functions", [])

    # Initialize dictionaries to store zero arrays for each function
    
    decisive_rewards_average_episode = {function_name: np.zeros(num_episode) for function_name in decisive_reward_functions}
    informative_rewards_average_episode = {function_name: np.zeros(num_episode) for function_name in informative_reward_functions}

    num_users = training_envs.num_users
    using_eavesdropper = (training_envs.num_eavesdroppers > 0)

    if curriculum_learning:
        # NOTE: TaskManager produces per-env difficulty configs and aggregates outcomes
        grid_limit = env_config.get("user_spawn_limits") 
        downlink_activated = env_config.get("BS_max_power") > 0
        uplink_activated = env_config.get("user_transmit_power") > 0
        Task_Manager = TaskManager(num_users,
                                   num_steps_per_episode = max_num_step_per_episode,
                                   user_limits=  grid_limit ,
                                   RIS_position= env_config.get("RIS_position"),downlink_uplink_eavesdropper_bools= [downlink_activated, uplink_activated, using_eavesdropper],
                                   thresholds = training_config.get("Curriculum_Learning_Thresholds", [0.25,0.25, 0.25]),
                                   random_seed = training_config.get("Task_Manager_random_seed", 126),
                                    num_environments = n_rollout_envs )
    
    index_episode_buffer_filled = 0
    num_eval_periode = 0
    eval_current_step = 0
    # Initialize variables
    buffer_filled = False
    best_average_reward = 0
    optim_steps = 0
    
    optim_steps_per_ep = int(ppo_epochs * ( max_num_step_per_episode // (rollout_size//n_rollout_envs) ) )
    
    if not using_eavesdropper:
        eavesdroppers_positions = None
    

    average_reward_per_env = np.zeros((num_episode, n_rollout_envs))
    average_best_reward_per_env = np.zeros((num_episode, n_rollout_envs))
    avg_user_fairness_all_episode_general = np.zeros(num_episode)

    #TODO: optimize this code snippet:  managing the tqdm buffer bar, can be optimized
    buffer_number_of_required_episode = buffer_size//(n_rollout_envs *max_num_step_per_episode)
    
    
    if buffer_number_of_required_episode == 0:
        buffer_number_of_required_episode = 1

    # === Main training loop over episodes ===
    for episode in tqdm(range(num_episode), desc="TRAINING", position=1, ascii="->#"):
        
        current_optim_steps_ep = 0

        if curriculum_learning:
            difficulty_config = Task_Manager.generate_episode_configs()
            # print(difficulty_config)
            training_envs.reset(difficulty_config)
        else:
            training_envs.reset()

        start_episode_time = time.time()
        
        user_positions = training_envs.get_users_positions()
        
        """message = (f"\n here are this episode starting positions for the users and the eavesdroppers: \n"
                   f"for the users: {user_positions} \n"
                   f" eavesdroppers: {training_envs.get_eavesdroppers_positions()} \n")
        logger.verbose(message)"""

        instant_user_rewards = np.zeros((n_rollout_envs, max_num_step_per_episode))
        basic_reward_episode = np.zeros((max_num_step_per_episode, n_rollout_envs))
        
        instant_eavesdropper_rewards = np.zeros((n_rollout_envs, max_num_step_per_episode))
        avg_eavesdropper_reward =  0
        instant_user_jain_fairness = np.zeros((max_num_step_per_episode , n_rollout_envs))
        
        decisive_rewards_current_episode = {function_name: np.zeros((n_rollout_envs, max_num_step_per_episode + 1)) for function_name in decisive_reward_functions}
        informative_rewards_current_episode = {function_name: np.zeros((n_rollout_envs, max_num_step_per_episode + 1)) for function_name in informative_reward_functions}

        avg_actor_loss = 0
        avg_critic_loss = 0
        max_instant_reward = 0
        total_reward = np.zeros(n_rollout_envs)
        step_time_list = []

        for num_step in range(max_num_step_per_episode):
            current_step = episode * max_num_step_per_episode + num_step

            if num_step == 0:
                states = training_envs.get_states()
                states = np.squeeze(states, axis=1)
            else:
                states = next_states

            # NOTE: PPO explores by sampling from the policy (no noise needed)
            selected_actions = []
            raw_actions = []
            logprobs = []
            
            for i in range(n_rollout_envs):
                action, logprob, raw_action = network.select_action(states[i], eval_mode=False)
                selected_actions.append(action)
                raw_actions.append(raw_action)
                logprobs.append(logprob)
            
            selected_actions = torch.stack(selected_actions)
            raw_actions = torch.stack(raw_actions)
            logprobs = torch.stack(logprobs)
            
            states, selected_actions, rewards, next_states = training_envs.step(states, selected_actions)

            # Reshaping the rewards
            instant_user_rewards[:, num_step] = rewards
            total_reward += rewards
            rewards = rewards.reshape((n_rollout_envs, 1))
            max_instant_reward = max(max_instant_reward, rewards.max().item())

            basic_reward_episode[num_step,:] = training_envs.get_basic_reward().reshape((n_rollout_envs))

            decisive_rewards = pickable_to_dict(training_envs.get_decisive_rewards())
            informative_rewards = pickable_to_dict(training_envs.get_informative_rewards())

            for function_name in decisive_reward_functions:
                decisive_rewards_current_episode[function_name][:, num_step] = [decisive_rewards[env_idx][function_name]['total_reward'] for env_idx in decisive_rewards]

            for function_name in informative_reward_functions:
                informative_rewards_current_episode[function_name][:, num_step] = [informative_rewards[env_idx][function_name]['total_reward'] for env_idx in informative_rewards]

            
            rewards = rewards.reshape((n_rollout_envs, 1))

            # NOTE: Store transitions for all envs in PPO rollout buffer
            for i in range(n_rollout_envs):
                done = bool(training_envs.is_done()) if hasattr(training_envs, 'is_done') else False
                success = network.store_transition(states[i], selected_actions[i], raw_actions[i], 
                                       float(rewards[i]), next_states[i], done=done, 
                                       logprob=logprobs[i], env_id = i)
                if not success:
                    # Buffer is full, break out of the loop
                    break

            # Update max reward in a single operation
            current_fairness = np.round(training_envs.get_jain_fairness(), decimals=4).squeeze()
            instant_user_jain_fairness[num_step] = current_fairness

            if using_eavesdropper:
                eavesdropper_rewards = np.array(training_envs.get_eavesdroppers_rewards()).squeeze()
                instant_eavesdropper_rewards[:, num_step] = eavesdropper_rewards  # Remove dimension of size 1

            network.actor.train()
            # NOTE: Train when rollout buffer is full or episode ends
            if network.rollout.ptr >= rollout_size or (num_step == max_num_step_per_episode - 1):
                if not buffer_filled:
                    index_episode_buffer_filled = episode
                    buffer_filled = True

                training_time_1 = time.time()
                actor_loss, critic_loss, mean_reward = network.training()
                step_time_list.append(time.time() - training_time_1)
                current_optim_steps_ep += ppo_epochs
                avg_actor_loss += actor_loss
                avg_critic_loss += critic_loss
                
                # Reset rollout buffer after training
                network.rollout.reset()
                # NOTE: Periodic logging of buffer stats, losses, and fairness
                if (current_step + 1) % frequency_information == 0 and (num_step +1) % frequency_information == 0:

                    current_avg_actor_loss = avg_actor_loss / current_optim_steps_ep
                    current_avg_critic_loss = avg_critic_loss / current_optim_steps_ep
                    writer.add_scalar("Actor Loss/Current average actor loss", current_avg_actor_loss, current_step)
                    writer.add_scalar("Critic Loss/Current average critic loss", current_avg_critic_loss, current_step)

                    # Local items (average and so on) represent the items for the current slice between last frequency information and the new one.
                    local_average_reward = np.mean(instant_user_rewards[:, num_step + 1 - frequency_information: num_step])

                    local_average_reward_per_env = np.mean(instant_user_rewards[:, num_step + 1 - frequency_information: num_step],axis=1)

                    user_fairness_mean_local = np.round(np.mean(instant_user_jain_fairness[num_step + 1 - frequency_information: num_step]), decimals=3)

                    user_fairness_mean_local_per_env = np.round(np.mean(instant_user_jain_fairness[num_step + 1 - frequency_information: num_step], axis=0), decimals=3)

                    writer.add_scalar("Rewards/Average local reward", local_average_reward, current_step)
                    writer.add_scalar("Rewards/Average global reward", np.mean(total_reward) / (num_step + 1), current_step)

                    writer.add_scalar("Actor Loss/Instant actor loss", actor_loss, current_step)
                    writer.add_scalar("Critic Loss/Instant critic loss", critic_loss, current_step)

                    writer.add_scalar("Fairness/Local average user Fairness", user_fairness_mean_local, current_step)

                    
                    if using_eavesdropper:
                        local_average_eavesdropper_reward = np.mean(instant_eavesdropper_rewards[:, num_step + 1 - frequency_information: num_step])

                        writer.add_scalar("Eavesdropper/Local average reward", local_average_eavesdropper_reward, current_step)
                        writer.add_histogram("Eavesdropper/Instant reward", instant_eavesdropper_rewards[:, num_step + 1 - frequency_information: num_step], current_step)
                    
                    message = (
                        f"\n"
                        f"================================================================================\n"
                        f" TRAINING EPISODE {episode - index_episode_buffer_filled} | STEP {num_step + 1} \n"
                        f" Training the NN takes {np.mean(step_time_list):.4f} sec on average across {len(step_time_list)} steps \n"
                        f"================================================================================\n"
                        f"| ~ POSITIONING: Positions of UEs: {user_positions}\n"
                        f"|--------------------------------------------------------------------------------|\n"
                        f"| ~ REWARDS: \n"
                        f"|     - Local Avg Reward: {np.float16(local_average_reward):.4f} | Max Instant Reward: {max_instant_reward:.4f} |\n"
                        f"|     - Max Instant Reward per Subprocess: {np.max(instant_user_rewards, axis=1)} |\n"
                        f"|     - Local Avg Reward per Subprocess: {local_average_reward_per_env} |\n"
                        f"|--------------------------------------------------------------------------------|\n"
                        f"| ~ FAIRNESS: \n"
                        f"|     - Local User Fairness (All Subprocesses): {user_fairness_mean_local:.4f} |\n"
                        f"|     - Local User Fairness per Subprocess: {user_fairness_mean_local_per_env} |\n"
                        f"|--------------------------------------------------------------------------------|\n"
                        f"| LOSSES: Avg Actor Loss: {current_avg_actor_loss:.4f} | Avg Critic Loss: {current_avg_critic_loss:.4f} |\n"
                        f"================================================================================\n")
                    
                    logger.verbose(message)


        if curriculum_learning:
                if using_eavesdropper:
                    episodes_outcomes = Task_Manager.compute_episodes_outcome( downlink_sum = training_envs.get_downlink_sum_for_success_conditions() ,
                                                                              uplink_sum = training_envs.get_uplink_sum_for_success_conditions(),
                                                                              best_eavesdropper_sum = training_envs.get_best_eavesdropper_sum_for_success_conditions() )
                else:
                    episodes_outcomes = Task_Manager.compute_episodes_outcome( downlink_sum = training_envs.get_downlink_sum_for_success_conditions() , uplink_sum = training_envs.get_uplink_sum_for_success_conditions())
                Task_Manager.update_episode_outcomes(episodes_outcomes)

        optim_steps += optim_steps_per_ep
            
        avg_actor_loss /= optim_steps_per_ep
        avg_critic_loss /= optim_steps_per_ep
        avg_reward = np.mean(total_reward) / max_num_step_per_episode

        avg_user_fairness_all_envs = np.mean(instant_user_jain_fairness)
        avg_user_fairness_all_episode_general[episode] = avg_user_fairness_all_envs


        for function_name in decisive_reward_functions:
            decisive_rewards_average_episode[function_name][episode] = np.mean(decisive_rewards_current_episode[function_name])

        for function_name in informative_reward_functions:
            informative_rewards_average_episode[function_name][episode] = np.mean(informative_rewards_current_episode[function_name])

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

        average_reward_per_env[episode] = np.mean(instant_user_rewards, axis=1)
        episode_max_instant_reward_reached_per_env = np.max(instant_user_rewards, axis=1)
        average_best_reward_per_env[episode] = episode_max_instant_reward_reached_per_env

        env_idx_best_reward_reached = np.argmax(episode_max_instant_reward_reached_per_env)
        fairness_for_best_reward = instant_user_jain_fairness[np.argmax(instant_user_rewards[env_idx_best_reward_reached]), env_idx_best_reward_reached]

        average_basic_reward_all_envs = np.mean(basic_reward_episode)
        basic_reward_for_best_env = basic_reward_episode[np.argmax(instant_user_rewards[env_idx_best_reward_reached]), env_idx_best_reward_reached] 
        
        best_basic_reward_per_env = np.max(basic_reward_episode,axis=0)

        message = (
            f"\n\n"
            f"====================================================================================================\n"
            f"| TRAINING EPISODE N° {episode - index_episode_buffer_filled} | Optimization Steps Performed: {optim_steps} |\n"
            f"====================================================================================================\n"
            f"| POSITIONING: \n"
            f"|----------------------------------------------------------------------------------------------------\n"
            f"| Positions of UEs for Each Subprocess: {user_positions}\n"
            f"| \n"
            f"| REWARDS: \n"
            f"|----------------------------------------------------------------------------------------------------\n"
            f"| --> Average Reward: {avg_reward:.4f} | Max Instant Reward: {np.max(instant_user_rewards):.4f} |\n"
            f"| Best Reward Obtained per Subprocess: {episode_max_instant_reward_reached_per_env}\n"
            f"| --> Average Basic Reward: {average_basic_reward_all_envs:.4f} | Basic Reward for Maximum Instant Reward: {basic_reward_for_best_env:.4f} |\n"
            f"| Best Basic Reward Obtained per Subprocess: {best_basic_reward_per_env}\n"
            f"| \n"
            f"| FAIRNESS: \n"
            f"|----------------------------------------------------------------------------------------------------\n"
            f"| --> Average User Fairness: {avg_user_fairness_all_envs:.4f} | User Fairness for Maximum Instant Reward: {fairness_for_best_reward:.4f} |\n"
            f"| \n"
            f"| OTHERS: \n"
            f"|----------------------------------------------------------------------------------------------------\n"
            f"| Average Actor Loss: {avg_actor_loss:.4f} | Average Critic Loss: {avg_critic_loss:.4f} |\n"
            f"====================================================================================================\n"
        )


        logger.verbose(message)

        writer.add_scalar("Actor Loss/Average actor Loss per episode", avg_actor_loss, episode)
        writer.add_scalar("Critic Loss/Average critic Loss per episode", avg_critic_loss, episode)
        writer.add_scalar("General/Time per episode", time.time() - start_episode_time, episode)

        writer.add_scalar("General/Mean total SSR per episode", np.mean(instant_user_rewards + instant_eavesdropper_rewards), episode)
        writer.add_scalar("General/Max total SSR per episode", np.max(instant_user_rewards + instant_eavesdropper_rewards), episode)
        
        for function_name in decisive_reward_functions:
            writer.add_scalar(f"Decisive Rewards/{function_name}/reward", decisive_rewards_average_episode[function_name][episode], episode)
            writer.add_scalar(f"Decisive Rewards/{function_name}/Averaged_reward", np.mean(decisive_rewards_average_episode[function_name][:episode+1]), episode)

        for function_name in informative_reward_functions:
            writer.add_scalar(f"Informative Rewards/{function_name}", informative_rewards_average_episode[function_name][episode], episode)
            writer.add_scalar(f"Informative Rewards/Averaged_{function_name}", np.mean(informative_rewards_average_episode[function_name][:episode+1]), episode)

        writer.add_scalar("Rewards/Average reward per episode", avg_reward, episode)
        writer.add_scalar("Rewards/Best average reward per episode", best_average_reward, episode)
        writer.add_scalar("Rewards/Average Max reward reached per episode", np.mean(episode_max_instant_reward_reached_per_env), episode)
        writer.add_scalar("Rewards/Best Max reward reached per episode", np.max(episode_max_instant_reward_reached_per_env), episode)
        writer.add_scalar("Rewards/Mean average reward", np.mean(average_reward_per_env[:episode+1]), episode)

        writer.add_scalar("Fairness/Max Reward's Fairness", fairness_for_best_reward, episode)
        writer.add_scalar("Fairness/Mean Average User Fairness", np.mean(avg_user_fairness_all_episode_general[:episode+1]), episode)
        writer.add_scalar("Fairness/Average User Fairness per episode", avg_user_fairness_all_envs, episode)
        
        """scalars = {f"_mean_average_reward_env_{j}": np.mean(average_reward_per_env[:episode,j]) for j in range(0,n_rollout_envs) }
        writer.add_scalars('Per_env', scalars, episode)"""

        if using_eavesdropper:
            avg_eavesdropper_reward = np.mean(instant_eavesdropper_rewards)
            writer.add_scalar("Eavesdropper/Average reward per episode", avg_eavesdropper_reward, episode)
        
        # Update the progress bar description with the last ending message
        if buffer_filled:
            tqdm.write(message)

    #*  CONDUCT EVALUATION IF SPECIFIED

        if conduct_eval and episode % eval_period == 0 and buffer_filled:
            tqdm.write(f" \n Conducting Evaluation \n")
        
            """if use_rendering:
                renderer = SituationRenderer(
                    M=eval_env.M,
                    lambda_h=eval_env.lambda_h,
                    N_t=training_envs.BS_transmit_antennas,
                    max_step_per_episode=training_config.get("max_steps_per_episode", 20000),
                    BS_position=training_envs.BS_position,
                    d_h=eval_env.d_h,
                    RIS_position=eval_env.RIS_position,
                    users_position=eval_env.get_users_positions(),
                )"""

            num_eval_periode += 1
            eval_current_step = 0

            # Allocate arrays for per-episode evaluation metrics
            whole_episode_rewards = np.zeros(episode_per_eval)
            whole_episode_fairness = np.zeros(episode_per_eval)
            whole_episode_max_rewards = np.zeros(episode_per_eval)
            whole_episode_best_reward_fairness = np.zeros(episode_per_eval)

            eval_total_reward = np.zeros(n_eval_rollout_threads)

            
            if using_eavesdropper:
                whole_episode_eaves_rewards = np.zeros(episode_per_eval)

            # Loop over evaluation episodes
            # NOTE: Loop over evaluation episodes
            for eval_episode in tqdm(range(episode_per_eval), desc="EVAL", position=2, ascii="->#"):

                eval_env.reset()

                # Log starting positions of users for the evaluation episode
                message = (
                    f"\nStarting positions for EVAL episode {eval_episode}:\n"
                    f"Users: {eval_env.get_users_positions()}\n"
                    f"Eavesdroppers: {eval_env.get_eavesdroppers_positions()}\n"
                )
                logger.verbose(message)
                eval_instant_user_jain_fairness = np.zeros((max_num_step_per_episode + 1, n_eval_rollout_threads))
                episode_user_rewards = np.zeros(shape = (max_num_step_per_episode+1, n_eval_rollout_threads))

                if using_eavesdropper:
                    episode_eaves_rewards = np.zeros(shape = (max_num_step_per_episode+1, n_eval_rollout_threads))


                # Loop over steps within the evaluation episode
                # NOTE: Rollout evaluation steps
                for num_step in range(max_num_step_per_episode):
                    eval_current_step += 1

                    if num_step == 0:
                        states = eval_env.get_states()
                        states = np.squeeze(states, axis=1)
                    else:
                        states = next_states
                    
                    # NOTE: For PPO evaluation, use deterministic actions
                    selected_actions = []
                    for i in range(n_eval_rollout_threads):
                        action, _, _ = network.select_action(states[i], eval_mode=True)
                        selected_actions.append(action)
                    selected_actions = torch.stack(selected_actions)
                    states, selected_actions, rewards, next_states = eval_env.step(states, selected_actions)

                    """for i,r in enumerate(rewards):
                        episode_user_rewards[num_step,i] = r.item()
                        eval_total_reward[i] += r.item()
                        
                        if r > max_instant_reward:
                            max_instant_reward = r"""
                    
                    # Convert the rewards tensor to numpy array and assign to the entire row at once
                    rewards_np = rewards
                    episode_user_rewards[num_step] = rewards_np  # Assign entire row at once
                    eval_total_reward += rewards_np  # Add to running totals in vector form
                    max_instant_reward = max(max_instant_reward, rewards.max().item())  # Find maximum in one operation

                    current_fairness = np.round(eval_env.get_jain_fairness(), decimals=3).squeeze()
                    
                    eval_instant_user_jain_fairness[num_step] = current_fairness

                    if using_eavesdropper:
                        eavesdropper_rewards = np.array(eval_env.get_eavesdroppers_rewards()).squeeze()
                        episode_eaves_rewards[num_step,:] = eavesdropper_rewards  # Remove dimension of size 1


                # Store per-episode metrics
                whole_episode_rewards[eval_episode] = np.mean(episode_user_rewards)
                whole_episode_fairness[eval_episode] = np.mean(eval_instant_user_jain_fairness)
                whole_episode_max_rewards[eval_episode] = np.max(episode_user_rewards)
                whole_episode_best_reward_fairness[eval_episode] = eval_instant_user_jain_fairness[np.unravel_index(np.argmax(episode_user_rewards), eval_instant_user_jain_fairness.shape)]

            # NOTE: Compute evaluation metrics across episodes
            mean_reward = np.mean(whole_episode_rewards)
            mean_fairness = np.mean(whole_episode_fairness)
            mean_max_reward = np.mean(whole_episode_max_rewards)
            mean_best_fairness = np.mean(whole_episode_best_reward_fairness)


            if using_eavesdropper:
                mean_eaves_reward = np.mean(whole_episode_eaves_rewards)
                std_eaves_reward = np.std(whole_episode_eaves_rewards)
                reward_eaves_ratio = whole_episode_rewards / (whole_episode_eaves_rewards + 1e-6)
                mean_ratio = np.mean(reward_eaves_ratio)
                composite_score = whole_episode_rewards - 0.5 * whole_episode_eaves_rewards
                mean_composite_score = np.mean(composite_score)

                writer.add_scalar("Evaluation/Mean total SSR per episode", mean_reward + mean_eaves_reward, episode)
                writer.add_scalar("Evaluation/Max total SSR per episode", np.max(whole_episode_rewards + whole_episode_eaves_rewards), episode)
                writer.add_scalar("Evaluation/Total Eavesdropper Reward per episode", mean_eaves_reward, episode)
                writer.add_scalar("Evaluation/Reward-to-Eavesdropper Ratio", mean_ratio, episode)
                writer.add_scalar("Evaluation/Composite Score", mean_composite_score, episode)

            else:
                writer.add_scalar("Evaluation/Mean total SSR per episode", mean_reward, episode)
                writer.add_scalar("Evaluation/Max total SSR per episode", np.max(whole_episode_rewards), episode)

            writer.add_scalar("Evaluation/Average Fairness", mean_fairness, episode)
            writer.add_scalar("Evaluation/Fairness for the best reward", mean_best_fairness, episode)
            writer.add_scalar("Evaluation/Max reward reached per episode", mean_max_reward, episode)
            writer.add_scalar("Evaluation/Total Reward per episode", mean_reward, episode)

            # Log evaluation summary
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
                
            logger.verbose(message)
            tqdm.write(message)