"""
Single-process training loop utilities for RIS Duplex RL experiments.

This module provides:
 - setup_logger: configure file and console logging with a custom VERBOSE level
 - single_process_trainer: the main training loop for a single environment

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


def ofp_single_process_trainer(training_envs, network, training_config, log_dir, writer,
                           eval_env=None, action_noise_activated=False,
                           batch_instead_of_buff=False, use_rendering=False):
    """
    Executes the training loop for the network with improved logging.

    Parameters:
        training_envs: The environment object.
        network: The neural network object.
        training_config: Configuration dictionary for training parameters.
        log_dir: Directory for logging.
        writer: TensorBoard SummaryWriter object for logging.
        eval_env: Optional evaluation environment.
        action_noise_activated: Boolean flag to activate noise in actions.
        batch_instead_of_buff: Boolean flag to use batch instead of buffer.
        use_rendering: Boolean to know if a rendering tool is needed.
    """
    # NOTE: Dedicated logger for this run; messages at VERBOSE level go to file only
    logger = setup_logger(log_dir)

    env_config = training_envs.env_config

    # Extract training configuration parameters from the provided config dictionary
    debugging = training_config.get("debugging", False)

    env_debugging = env_config.get("debugging", False)
    num_episode = training_config.get("number_episodes", 15)

    solo_episode = num_episode == 1
    max_num_step_per_episode = training_config.get("max_steps_per_episode", 20000)
    reward_smoothing_factor = training_config.get("reward_smoothing_factor", 0.5)
    batch_size = training_config.get("batch_size", 128)
    frequency_information = training_config.get("frequency_information", 500)
    conduct_eval = eval_env is not None
    episode_per_eval = training_config.get("episode_per_eval_env", 1) if conduct_eval else None
    eval_period = training_config.get("eval_period", 1) if conduct_eval else None
    saving_frequency = training_config.get("network_save_checkpoint_frequency", 100)
    plot_saving_frequency = training_config.get("plot_save_checkpoint_frequency", 100)

    curriculum_learning = training_config.get("Curriculum_Learning", True)  # ! WIP: heuristics to vary difficulty
    
    noise_scale = training_config.get("noise_scale", 0.1)
    
    num_users = training_envs.num_users
    
    using_eavesdropper = (training_envs.num_eavesdroppers > 0)

    if curriculum_learning:
        # NOTE: TaskManager encapsulates curriculum generation and episode outcomes
        grid_limit = env_config.get("user_spawn_limits") 
        downlink_activated = env_config.get("BS_max_power") > 0
        uplink_activated = env_config.get("user_transmit_power") > 0
        Task_Manager = TaskManager(num_users,
                                   num_steps_per_episode = max_num_step_per_episode,
                                   user_limits=  grid_limit ,
                                   RIS_position= env_config.get("RIS_position"),
                                   downlink_uplink_eavesdropper_bools= [downlink_activated, uplink_activated, using_eavesdropper],
                                   thresholds = training_config.get("Curriculum_Learning_Thresholds", [0.5,0.5]),
                                   random_seed = training_config.get("Task_Manager_random_seed", 126) )

    # NOTE: Determine the buffer size based on whether batch mode is activated
    buffer_size = deepcopy(batch_size) if batch_instead_of_buff else network.replay_buffer.buffer_size

    index_episode_buffer_filled = None
    users_trajectory = np.zeros((num_episode, training_envs.num_users, 2))
    eavesdroppers_trajectory = np.zeros((num_episode, training_envs.num_eavesdroppers, 2))

    num_eval_periode = 0
    eval_current_step = 0

    # Initialize variables to track training progress
    buffer_filled = False
    best_average_reward = 0

    if not using_eavesdropper:
        eavesdroppers_positions = None
        
    # Initialize renderer data if rendering is activated
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

    optim_steps = 0

    average_reward_per_env = np.zeros(num_episode)
    
    #TODO: optimize this code snippet:  managing the tqdm buffer bar, can be optimized
    buffer_bar_finished = False
    buffer_number_of_required_episode = buffer_size//(max_num_step_per_episode)
    buffer_smaller_than_one_episode = (buffer_size//(max_num_step_per_episode) < 1)
    length_episode_rb_matching = ( (buffer_size % (max_num_step_per_episode)) == 0)
    if buffer_number_of_required_episode == 0:
        buffer_number_of_required_episode = 1

    # NOTE: Initialize progress bar for filling the replay buffer if not using batch mode
    filling_buffer_progress_bar = tqdm(
        total=buffer_number_of_required_episode,
        desc="FILLING REPLAY BUFFER",
        position=0,
        ascii="->#",
        leave=True
    ) if not batch_instead_of_buff else None

    # === Main training loop over episodes ===
    for episode in tqdm(range(num_episode), desc="TRAINING", position=1, ascii="->#"):
        start_episode_time = time.time()
        
        if curriculum_learning:
            difficulty_config = Task_Manager.generate_episode_configs()
            # print(difficulty_config)
            training_envs.reset(difficulty_config)
        else:
            training_envs.reset()
        users_position=training_envs.get_users_positions()

        if using_eavesdropper:
            
            max_episode_eavesdropper_reward = 0
            eavesdroppers_positions = training_envs.get_eavesdroppers_positions()
            eavesdroppers_trajectory[episode] = eavesdroppers_positions
    
            episode_best_4_eavesdroppers_power_patterns = {
                "downlink_power_patterns": [],
                "uplink_power_patterns": [],
                "W_power_patterns": [],
                "rewards": [],
                "steps": [],
            }
            
        #print(f"\n User positions are: \n {users_position} \n Eavesdroppers positions are: \n{eavesdroppers_positions} \n")

        if buffer_filled:
            if using_eavesdropper:
                # Log starting positions of users for the episode
                position_message = (
                    f"\nCommencing training episode {episode-buffer_number_of_required_episode} with users and eavesdroppers positions:\n"
                    f"   !~ Users Positions: {list(users_position)} \n"
                    f"   !~ Eavesdroppers Positions: {list(eavesdroppers_positions)} \n"
                )
            else:
                # Log starting positions of users for the episode
                position_message = (
                    f"\nCommencing training episode {episode-buffer_number_of_required_episode} with users positions:\n"
                    f"   !~ Users Positions: {list(users_position)} \n"
                )
        else:
            if using_eavesdropper:
                # Log starting positions of users for the episode
                position_message = (
                    f"\n Initializing positions for replay buffer episode {episode} with users and eavesdroppers positions:\n"
                    f"   !~ Users Positions: {list(users_position)} \n"
                    f"   !~ Eavesdroppers Positions: {list(eavesdroppers_positions)} \n"
                )
            else:
                # Log starting positions of users for the episode
                position_message = (
                    f"\n Initializing positions for replay buffer episode {episode} with users positions:\n"
                    f"   !~ Users Positions: {list(users_position)} \n"
                )

        users_trajectory[episode] = users_position

        logger.verbose(position_message)

        # Initialize arrays to track rewards and losses for the episode
        instant_user_rewards = np.zeros(max_num_step_per_episode) - np.inf
        instant_eavesdropper_rewards = np.zeros(max_num_step_per_episode)

        average_rewards = np.zeros(max_num_step_per_episode + 1)
        paper_average_rewards = np.zeros(max_num_step_per_episode + 1)
        instant_user_jain_fairness = np.zeros(max_num_step_per_episode + 1)
        avg_actor_loss = 0
        avg_critic_loss = 0
        total_reward = 0
        additional_information_best_case = 0
        basic_reward_episode = np.zeros(max_num_step_per_episode)
        step_time_list = []
        max_episode_reward = -np.inf

        # Initialize renderer for the episode if rendering is activated
        if use_rendering:
            # NOTE: Renderer collects visualization artifacts for later export
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

            """renderer = SituationRenderer(
                M=training_envs.M,
                N_t=training_envs.BS_transmit_antennas,
                lambda_h=training_envs.lambda_h,
                max_step_per_episode=training_config.get("max_steps_per_episode", 20000),
                BS_position=training_envs.BS_position,
                d_h=training_envs.d_h,
                RIS_position=training_envs.RIS_position,
                users_position=users_position,
                num_frames=200
            )"""

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

            # NOTE: Select action with or without noise based on the noise_activated flag
            if action_noise_activated:
                selected_action, selected_noised_action = network.select_noised_action(state,noise_scale = noise_scale) 
                state, _, reward, next_state = training_envs.step(state, selected_noised_action)
            else:
                selected_action = network.select_action(state)
                state, selected_action, reward, next_state = training_envs.step(state, selected_action)

            # NOTE: Store the transition in the replay buffer (batch size = 1 in single-process)
            network.store_transition(state, selected_action, reward, next_state, batch_size = 1)

            reward_value = reward.item()

            # NOTE: Track rewards and fairness for the step
            instant_user_rewards[num_step] = reward_value
            total_reward = np.sum(instant_user_rewards)
            avg_reward = total_reward / (num_step + 1)
            average_rewards[num_step] = avg_reward
            if num_step > 1:
                paper_average_rewards[num_step] = reward_smoothing_factor * paper_average_rewards[num_step - 1] + (1 - reward_smoothing_factor) * reward_value

            instant_fairness = round(training_envs.get_user_jain_fairness(), ndigits=4)
            instant_user_jain_fairness[num_step] = instant_fairness
            
            if using_eavesdropper:
                # Track eavesdropper rewards if eavesdroppers are present
                eavesdropper_reward = training_envs.get_eavesdropper_rewards()
                instant_eavesdropper_rewards[num_step] = eavesdropper_reward
                total_eavesdropper_reward = np.sum(instant_eavesdropper_rewards)

            instant_baseline_reward = training_envs.get_basic_reward()
            basic_reward_episode[num_step] = instant_baseline_reward


            # NOTE: Render power patterns if rendering is activated and at the specified interval
            if use_rendering and num_step % (max_num_step_per_episode // renderer.num_frames) == 0:
                W = training_envs.get_W()
                total_power_deployed = round(np.sum((W @ W.conj().T).diagonal().real), 4)
                renderer.store_power_patterns(
                    reward=reward_value,
                    W_power_patterns=training_envs.get_W_power_patterns(),
                    Downlink_power_patterns=training_envs.get_downlink_power_patterns(),
                    Uplink_power_patterns=training_envs.get_uplink_power_patterns()
                )

            # NOTE: Train the network if the replay buffer has enough samples
            network.actor.train()
            if network.replay_buffer.size >= buffer_size:
                if not buffer_filled:
                    index_episode_buffer_filled = episode
                    buffer_filled = True

                # Track and render best rewards
                if reward_value > max_episode_reward and use_rendering:
                    max_episode_reward = round(reward_value, 4)
                    
                    additional_information_best_case = training_envs.get_additionnal_informations()
                    W = training_envs.get_W()
                    """
                    theta = training_envs.get_Theta()
                    theta_phi = training_envs.get_theta_phis()
                    """
                    best_W = training_envs.get_W()
                    best_Theta = training_envs.get_Theta()
                    total_power_deployed = round(np.trace(np.diag(np.diag(W @ W.conj().T)).real), 4)
                    max_W_power_patterns = training_envs.get_W_power_patterns()
                    max_downlink_power_patterns = training_envs.get_downlink_power_patterns()
                    max_uplink_power_patterns = training_envs.get_uplink_power_patterns()

                    episode_best_power_patterns["rewards"].append(deepcopy(reward_value))
                    episode_best_power_patterns["steps"].append(deepcopy(num_step))
                    episode_best_power_patterns["W_power_patterns"].append(deepcopy(max_W_power_patterns))
                    episode_best_power_patterns["downlink_power_patterns"].append(deepcopy(max_downlink_power_patterns))
                    episode_best_power_patterns["uplink_power_patterns"].append(deepcopy(max_uplink_power_patterns))

                # Track and render best eavesdropper rewards
                if using_eavesdropper and eavesdropper_reward > max_episode_eavesdropper_reward and use_rendering:
                    max_episode_eavesdropper_reward = round(eavesdropper_reward, 4)
                    W = training_envs.get_W()
                    """theta = training_envs.get_Theta()
                    theta_phi = training_envs.get_theta_phis()"""
                    total_power_deployed = round(np.trace(np.diag(np.diag(W @ W.conj().T)).real), 4)
                    eave_max_W_power_patterns = training_envs.get_W_power_patterns()
                    eave_max_downlink_power_patterns = training_envs.get_downlink_power_patterns()
                    eave_max_uplink_power_patterns = training_envs.get_uplink_power_patterns()

                    episode_best_4_eavesdroppers_power_patterns["rewards"].append(deepcopy(max_episode_eavesdropper_reward))
                    episode_best_4_eavesdroppers_power_patterns["steps"].append(deepcopy(num_step))
                    episode_best_4_eavesdroppers_power_patterns["W_power_patterns"].append(deepcopy(eave_max_W_power_patterns))
                    episode_best_4_eavesdroppers_power_patterns["downlink_power_patterns"].append(deepcopy(eave_max_downlink_power_patterns))
                    episode_best_4_eavesdroppers_power_patterns["uplink_power_patterns"].append(deepcopy(eave_max_uplink_power_patterns))

                elif reward_value > max_episode_reward:
                    max_episode_reward = round(reward_value, 4)
                    additional_information_best_case = training_envs.get_additionnal_informations()

                # NOTE: Perform a training step and log the time taken
                training_time_1 = time.time()
                actor_loss, critic_loss, rewards = network.training(batch_size=batch_size)
                step_time_list.append(time.time() - training_time_1)
                optim_steps += 1

                avg_actor_loss += actor_loss
                avg_critic_loss += critic_loss

                if solo_episode :
                    writer.add_scalar("Rewards/Instant User reward", reward_value, current_step)
                    writer.add_scalar("Rewards/Baseline Instant Reward", instant_baseline_reward, current_step)
                    writer.add_scalar("Fairness/Instant JFI", instant_fairness, current_step)
                    writer.add_scalar("Fairness/JFI for best reward obtained",instant_user_jain_fairness[np.argmax(instant_user_rewards)], current_step)
                    if using_eavesdropper:
                        writer.add_scalar("Eavesdropper/Instant Eavesdroppers reward", eavesdropper_reward, current_step)


                # NOTE: Log training information at specified frequency
                if (current_step + 1) % frequency_information == 0:
                    W = training_envs.get_W()
                    total_power_deployed = round(np.trace(np.diag(np.diag(W @ W.conj().T)).real), 4)

                    writer.add_scalar("Replay Buffer/Average reward stored", np.mean(network.replay_buffer.reward_buffer), current_step)
                    writer.add_histogram("Replay Buffer/Rewards", network.replay_buffer.reward_buffer.squeeze(), current_step)

                    reward_combined = instant_user_rewards + instant_eavesdropper_rewards

                    if (num_step + 1) % frequency_information == 0:
                        local_average_reward = np.mean(instant_user_rewards[num_step + 1 - frequency_information: num_step])
                        local_average_basic_reward = np.mean(basic_reward_episode[num_step + 1 - frequency_information: num_step])

                        local_user_fairness = round(np.mean(instant_user_jain_fairness[num_step + 1 - frequency_information: num_step]), ndigits=4)

                        writer.add_scalar("Rewards/Local Average Reward", local_average_reward, current_step)
                        writer.add_histogram("Rewards/Paper Average Reward", paper_average_rewards[num_step + 1 - frequency_information: num_step], current_step)
                        writer.add_scalar("Rewards/Global Average Reward", avg_reward, current_step)
                        writer.add_scalar("Rewards/Max Instant Reward", np.max(instant_user_rewards), current_step)
                        writer.add_scalar("Rewards/Local Average Baseline Reward", local_average_basic_reward, current_step)
                        writer.add_histogram("Rewards/Instant reward", instant_user_rewards[num_step + 1 - frequency_information: num_step], current_step)

                        current_avg_actor_loss = avg_actor_loss / num_step
                        current_avg_critic_loss = avg_critic_loss / num_step
                        writer.add_scalar("Actor Loss/Instant actor loss", actor_loss, current_step)
                        writer.add_scalar("Actor Loss/Current average actor loss", current_avg_actor_loss, current_step)
                        writer.add_scalar("Critic Loss/Instant critic loss", critic_loss, current_step)
                        writer.add_scalar("Critic Loss/Current average critic loss", current_avg_critic_loss, current_step)
                        writer.add_scalar("Fairness/User local average Fairness", local_user_fairness, current_step)
                        
                        writer.add_scalar("General/Power deployed (Watts)", total_power_deployed, current_step)

                        if using_eavesdropper:
                            reward_combined = instant_user_rewards + instant_eavesdropper_rewards

                            local_average_reward_combined = np.mean(reward_combined[num_step + 1 - frequency_information: num_step])
                            writer.add_scalar("General/Local average total SSR", local_average_reward_combined, current_step)

                            avg_eavesdropper_reward = deepcopy(total_eavesdropper_reward) / num_step
                            local_average_eavesdropper_reward = np.mean(instant_eavesdropper_rewards[num_step + 1 - frequency_information: num_step])
                            writer.add_scalar("Eavesdropper/Local average reward", local_average_eavesdropper_reward, current_step)
                            writer.add_histogram("Eavesdropper/Instant reward", instant_eavesdropper_rewards[num_step + 1 - frequency_information: num_step], current_step)



                        message = (
                            f"\n|--> TRAINING EPISODE {episode - index_episode_buffer_filled}, STEP {num_step + 1}\n"
                            f"     |--> Training the NN takes {np.mean(step_time_list):.4f} sec on average across {len(step_time_list)} steps \n"
                            f"  |--> LOCAL AVERAGE REWARD {np.float16(local_average_reward)}, MAX INSTANT REWARD REACHED {np.max(instant_user_rewards)}\n"
                            f"  |--> LOCAL AVERAGE BASIC REWARD {np.float16(local_average_basic_reward)}, MAXIMUM INSTANT BASIC REWARD: {basic_reward_episode[np.argmax(instant_user_rewards)]:.4f}\n"
                            f"  |--> Detailed basic reward for best case: {additional_information_best_case}\n"       
                            f" |--> ACTOR LOSS {actor_loss}, CRITIC LOSS {critic_loss}\n"
                            f" |--> USER FAIRNESS FOR THE BEST INSTANT REWARD {instant_user_jain_fairness[np.argmax(instant_user_rewards)]}, LOCAL USER FAIRNESS {local_user_fairness}\n"
                            f" |--> POWER DEPLOYED: {total_power_deployed} Watts\n"
                            f"---------------------------\n"
                        )
                        logger.verbose(message)


        #TODO: optimize this code snippet: Correctly handling the number management of the replay buffer loading bar for the tqdm.
        if batch_instead_of_buff:
            pass
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

        """if curriculum_learning: 
            episodes_outcomes = Task_Manager.compute_episodes_outcome( downlink_sum = training_envs.get_downlink_sum_for_success_conditions() , uplink_sum = training_envs.get_uplink_sum_for_success_conditions())
            Task_Manager.update_episode_outcomes(episodes_outcomes)"""

        # === Episode summary metrics ===
        # NOTE: Calculate average losses and rewards for the episode
        avg_actor_loss /= max_num_step_per_episode
        avg_critic_loss /= max_num_step_per_episode
        avg_reward = np.mean(instant_user_rewards)
        avg_fairness = np.mean(instant_user_jain_fairness)
        
        if using_eavesdropper:
            avg_eavesdropper_reward = np.mean(instant_eavesdropper_rewards)

        
        # NOTE: Save the best model based on average reward
        if best_average_reward < avg_reward and not debugging:
            best_average_reward = avg_reward
            saving_model_directory = f"{log_dir}/model_saved/best_model"
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

        # NOTE: Save plots at designated frequency to visualize system evolution
        elif episode % plot_saving_frequency == 0 and not debugging:

            rendering = use_rendering and buffer_filled
            if rendering:
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

                tqdm.write("Creating animations")
                renderer.render_situation(save_dir=f"{log_dir}/animations/checkpoints", name_file=f"episode_{episode}_dynamic_profile.mp4")
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


        # NOTE: Save the final model at the end of training
        if episode == num_episode - 1 and not debugging:
            saving_model_directory = f"{log_dir}/model_saved/final_episode"
            network.save_models(saving_model_directory)
            
        elif episode % saving_frequency == 0 and not debugging:
            saving_model_directory = f"{log_dir}/model_saved/episode_{episode}"
            network.save_models(saving_model_directory)

        # NOTE: Log episode summary (console + file)
        episode_max_instant_reward_reached = max(instant_user_rewards)
        if index_episode_buffer_filled is None:
            index_episode_buffer_filled = 0
        
        if buffer_filled:
            
            writer.add_scalar("Replay Buffer/Average reward stored", np.mean(network.replay_buffer.reward_buffer), current_step)

            if curriculum_learning:
                if using_eavesdropper:
                    episodes_outcomes = Task_Manager.compute_episodes_outcome( downlink_sum = training_envs.get_downlink_sum_for_success_conditions() ,
                                                                              uplink_sum = training_envs.get_uplink_sum_for_success_conditions(),
                                                                              best_eavesdropper_sum = training_envs.get_best_eavesdropper_sum_for_success_conditions() )
                else:
                    episodes_outcomes = Task_Manager.compute_episodes_outcome( downlink_sum = training_envs.get_downlink_sum_for_success_conditions() , uplink_sum = training_envs.get_uplink_sum_for_success_conditions())
                Task_Manager.update_episode_outcomes(episodes_outcomes)

            # Message for printing to the console
            console_message = (
                f"\n\n !!~ TRAINING EPISODE No {episode - index_episode_buffer_filled} | Optimization Steps Performed: {optim_steps}\n"
                f"--------------------------------------------------------------------------------\n"
                f"   ~ ~ REWARDS:\n"
                f"     |--> Average Reward: {avg_reward:.4f} | Max Instant Reward: {episode_max_instant_reward_reached:.4f}\n"
                f"     |--> Average Basic Reward: {np.mean(basic_reward_episode):.4f} | Basic Reward for Maximum Instant Reward: {basic_reward_episode[np.argmax(instant_user_rewards)]:.4f}\n"
                f"     |--> Detailed Basic Reward for Best Case: {additional_information_best_case}\n"
                f"--------------------------------------------------------------------------------\n"
                f"  ~ ~ FAIRNESS:\n"
                f"     |--> Average User Fairness: {avg_fairness:.4f} | User Fairness for Best Instant Reward: {instant_user_jain_fairness[np.argmax(instant_user_rewards)]:.4f}\n"
                f"--------------------------------------------------------------------------------\n"
                f"  ~ ~ OTHERS:\n"
                f"     |--> Average Actor Loss: {avg_actor_loss:.4f} | Average Critic Loss: {avg_critic_loss:.4f}\n\n"
            )

            # Message for logging to a file
            log_message = (
                f"\n+{'=' * 100}+\n"
                f"| !~ TRAINING EPISODE N° {episode - index_episode_buffer_filled} | Optimization Steps Performed: {optim_steps} |\n"
                f"+{'=' * 100}+\n"
                f"|  ~ ~ REWARDS: |\n"
                f"|     --> Average Reward: {avg_reward:.4f} | Max Instant Reward: {episode_max_instant_reward_reached:.4f} |\n"
                f"|     --> Average Basic Reward: {np.mean(basic_reward_episode):.4f} | Basic Reward for Maximum Instant Reward: {basic_reward_episode[np.argmax(instant_user_rewards)]:.4f} |\n"
                f"|     --> Detailed Basic Reward for Best Case: {additional_information_best_case} |\n"
                f"+{'=' * 100}+\n"
                f"|  ~ ~ FAIRNESS: |\n"
                f"|     --> Average User Fairness: {avg_fairness:.4f} | User Fairness for Best Instant Reward: {instant_user_jain_fairness[np.argmax(instant_user_rewards)]:.4f} |\n"
                f"+{'=' * 100}+\n"
                f"|  ~ ~ OTHERS: |\n"
                f"|     --> Average Actor Loss: {avg_actor_loss:.4f} | Average Critic Loss: {avg_critic_loss:.4f} |\n"
                f"+{'=' * 100}+\n")

            tqdm.write(console_message)

            # Log to file
            logger.verbose(log_message)
        

        """if not training_envs.verbose:
            # Message for printing to the console
            console_message = (
                f"\nTRAINING EPISODE N° {episode - index_episode_buffer_filled} | Optimization Steps Performed: {optim_steps}\n"
                f"--------------------------------------------------------------------------------\n"
                f" REWARDS:\n"
                f"    Average Reward: {avg_reward:.4f} | Max Instant Reward: {episode_max_instant_reward_reached:.4f}\n"
                f"    Average Basic Reward: {np.mean(basic_reward_episode):.4f} | Basic Reward for Maximum Instant Reward: {basic_reward_episode[np.argmax(instant_user_rewards)]:.4f}\n"
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
                f"|     Average Basic Reward: {np.mean(basic_reward_episode):.4f} | Basic Reward for Maximum Instant Reward: {basic_reward_episode[np.argmax(instant_user_rewards)]:.4f} |\n"
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


        average_reward_per_env[episode] = avg_reward

        

        writer.add_scalar("General/Time per episode", time.time() - start_episode_time, episode)
        
        writer.add_scalar("Rewards/Mean average reward", np.mean(average_reward_per_env[:episode+1]), episode)
        writer.add_scalar("Rewards/Max reward reached per episode", episode_max_instant_reward_reached, episode)
        writer.add_scalar("Rewards/Average Reward per episode", avg_reward, episode)
        writer.add_scalar("Rewards/Best average reward per episode", best_average_reward, episode)
        writer.add_scalar("Fairness/Average User Fairness per episode", avg_fairness, episode)

        writer.add_scalar("Actor Loss/Average actor Loss per episode", avg_actor_loss, episode)
        writer.add_scalar("Critic Loss/Average critic Loss per episode", avg_critic_loss, episode)

        if using_eavesdropper:
            writer.add_scalar("General/Mean total SSR per episode", np.mean(instant_user_rewards + instant_eavesdropper_rewards), episode)
            writer.add_scalar("General/Max total SSR per episode", np.max(instant_user_rewards + instant_eavesdropper_rewards), episode)
            writer.add_scalar("Eavesdropper/Total Reward per episode", total_eavesdropper_reward, episode)
            writer.add_scalar("Eavesdropper/Average reward per episode", avg_eavesdropper_reward, episode)


        """np.savez(f'{log_dir}/data/test_W_matrix.npz',
                 test_W_matrix = best_W,
                 test_theta = best_Theta )
        
        W_message = (
            f" \n The final W matrix is: \n {best_W} \n"
            f" \n The final Theta matrix is: \n {best_Theta} \n"
        )

        logger.verbose(W_message)"""

        # * CONDUCT EVALUATION IF SPECIFIED 

        if conduct_eval and episode % eval_period == 0 and buffer_filled:
            tqdm.write(f" \n Conducting Evaluation \n")
        
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


            num_eval_periode += 1
            eval_current_step = 0

            # NOTE: Allocate arrays for per-episode evaluation metrics
            all_episode_rewards = np.zeros(episode_per_eval)
            all_episode_fairness = np.zeros(episode_per_eval)
            all_episode_max_rewards = np.zeros(episode_per_eval)
            all_episode_best_fairness = np.zeros(episode_per_eval)

            if using_eavesdropper:
                all_episode_eaves_rewards = np.zeros(episode_per_eval)


            # NOTE: Loop over evaluation episodes
            for eval_episode in tqdm(range(episode_per_eval), desc="EVAL", position=2, ascii="->#"):

                eval_env.reset()

                if using_eavesdropper:
                    # Log starting positions of users and eavesdroppers for the evaluation episode
                    message = (
                        f"\nStarting positions for EVAL episode {eval_episode}:\n"
                        f"Users: {eval_env.get_users_positions()}\n"
                        f"Eavesdroppers: {eval_env.get_eavesdroppers_positions()}\n"
                    )
                else:
                    # Log starting positions of users for the evaluation episode
                    message = (
                        f"\nStarting positions for EVAL episode {eval_episode}:\n"
                        f"Users: {eval_env.get_users_positions()}\n"
                    )

                logger.verbose(message)
                episode_user_rewards = np.zeros(max_num_step_per_episode)
                episode_user_fairness = np.zeros(max_num_step_per_episode)

                if using_eavesdropper:
                    episode_eaves_rewards = np.zeros(max_num_step_per_episode)

                # NOTE: Loop over steps within the evaluation episode
                for num_step in range(max_num_step_per_episode):
                    eval_current_step += 1

                    state = eval_env.get_states()[0] if num_step == 0 else next_state[0]
                    selected_action = network.select_action(state)
                    state, selected_action, reward, next_state = eval_env.step(state, selected_action)

                    reward_value = reward.item()
                    fairness_value = eval_env.get_user_jain_fairness()

                    if using_eavesdropper:
                        eavesdropper_reward = eval_env.get_eavesdropper_rewards()
                        episode_eaves_rewards[num_step] = eavesdropper_reward

                    episode_user_rewards[num_step] = reward_value
                    episode_user_fairness[num_step] = fairness_value

                # Store per-episode metrics
                all_episode_rewards[eval_episode] = np.mean(episode_user_rewards)
                all_episode_fairness[eval_episode] = np.mean(episode_user_fairness)
                all_episode_max_rewards[eval_episode] = np.max(episode_user_rewards)
                all_episode_best_fairness[eval_episode] = episode_user_fairness[np.argmax(episode_user_rewards)]

                if using_eavesdropper:
                    all_episode_eaves_rewards[eval_episode] = np.mean(episode_eaves_rewards)


            # NOTE: Compute evaluation metrics across evaluation episodes
            mean_reward = np.mean(all_episode_rewards)
            std_reward = np.std(all_episode_rewards)
            mean_fairness = np.mean(all_episode_fairness)
            std_fairness = np.std(all_episode_fairness)
            mean_max_reward = np.mean(all_episode_max_rewards)
            mean_best_fairness = np.mean(all_episode_best_fairness)


            if using_eavesdropper:
                mean_eaves_reward = np.mean(all_episode_eaves_rewards)
                std_eaves_reward = np.std(all_episode_eaves_rewards)
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
