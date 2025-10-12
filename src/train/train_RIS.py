"""
Entrypoint for training RIS Duplex Reinforcement Learning agents.

This module serves as the main training pipeline for RIS (Reconfigurable Intelligent Surface)
Duplex communication systems using various deep reinforcement learning algorithms.

Workflow:
    1. Parse configuration filename from command line arguments
    2. Load and validate YAML configuration file
    3. Build training and optional evaluation environments (with vectorization support)
    4. Instantiate the specified RL algorithm (DDPG, TD3, SAC, PPO) with configured hyperparameters
    5. Launch appropriate trainer (single or multi-process) based on configuration

Supported Algorithms:
    - DDPG (Deep Deterministic Policy Gradient)
    - Custom DDPG (Modified DDPG implementation)
    - TD3 (Twin Delayed Deep Deterministic Policy Gradient)
    - SAC (Soft Actor-Critic)
    - PPO (Proximal Policy Optimization)

Code Documentation Legend:
    - TODO: Future improvements or refactoring opportunities
    - NOTE: Important behavior or design decisions
    - WARNING: Critical runtime considerations
    - QUESTION: Areas requiring further investigation
"""
import numpy as np
import torch
import os
import sys
import logging
import datetime
from pathlib import Path
sys.path.append(os.path.dirname((os.path.dirname(os.path.dirname(os.path.realpath(__file__)))) ) )

from src.runners.off_policy import ofp_single_process_trainer, ofp_multiprocess_trainer

from src.runners.on_policy import onp_single_process_trainer, onp_multiprocess_trainer

from src.environment.tools import parse_args, parse_config, write_line_to_file
from src.environment.multiprocessing import  make_train_env, make_eval_env
from src.algorithms import DDPG, Custom_DDPG, TD3, SAC, PPO

from torch.utils.tensorboard import SummaryWriter


def main(args: list) -> None:
    """
    Main training function for RIS Duplex RL agents.
    
    Args:
        args: Command line arguments (typically sys.argv[1:])
    """

    config_file_name = parse_args()
    
    config = parse_config(config_file_name)

    # Extract configuration sections
    env_config = getattr(config, "Environment")
    network_config = getattr(config, "Network")
    training_config = getattr(config, "Training_parameters")
    
    # Extract key configuration parameters
    env_seed = env_config["env_seed"]
    debugging = training_config.get("debugging", False)
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    env_name = env_config.get("Environment name", "RIS_Duplex")
    algorithm_name = network_config.get("algorithm", "ddpg")
    experiment_name = f"env_seed{env_seed}_{timestamp}"
    n_rollout_train = training_config.get("n_rollout_threads", 1)
    if debugging:
        log_dir = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) +"/data" + "/debugging") / algorithm_name / experiment_name
        if not log_dir.exists():
            os.makedirs(str(log_dir))
    else:
        log_dir = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) +"/data" + "/raw") \
        / env_name / config_file_name / algorithm_name / experiment_name
        if not log_dir.exists():
            os.makedirs(str(log_dir))

    writer = SummaryWriter(log_dir)

    # Initialize logging and device configuration
    logs_terminal_txt_file = f"{log_dir}/log.txt"
    noise_activated = False

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # choosing the device
    cuda_available = torch.cuda.is_available()

    # Log device availability
    if cuda_available:
        print("\nBoth CUDA & CPU are available\n")
        write_line_to_file(logs_terminal_txt_file, "Both CUDA & CPU are available\n", mode='w')
    else:
        print("\nOnly CPU is available\n")
        write_line_to_file(logs_terminal_txt_file, "Only CPU is available\n", mode='w')

    # CUDA configuration
    # Configure device (CUDA or CPU)
    if training_config.get("cuda", True) and cuda_available:
        write_line_to_file(logs_terminal_txt_file, "Choosing to use CUDA (GPU)...\n")
        print("Choosing to use CUDA (GPU)...\n")
        device = torch.device("cuda:0")  # Use CUDA mask to control which GPU to use
        # torch.set_num_threads(training_config.get("n_training_threads",1))
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
    else:
        write_line_to_file(logs_terminal_txt_file, "Choosing to use CPU...\n")
        print("Choosing to use CPU...\n")
        device = torch.device("cpu")
        torch.set_num_threads(training_config.get("n_training_threads",1))

    # Environment initialization (vectorized depending on n_rollout_threads)
    train_envs = make_train_env(env_config, training_config)

    if training_config.get("conduct_evaluation",False):
        if train_envs.nremotes == 1:
            eval_env = make_eval_env(env_config,training_config)
        else:
            eval_env = make_eval_env(env_config, training_config)
    else:
        eval_env = None

    # Enable optional situation rendering
    use_rendering = training_config.get("rendering",False)
    
    action_noise_scale = training_config.get("action_noise_scale", 0)
    action_noise_activated = action_noise_scale != 0

    # ============================================================================
    # Algorithm Instantiation
    # ============================================================================
    
    off_policy = True
    if algorithm_name.lower() == "ddpg":
        network = DDPG(
            state_dim=train_envs.state_dim,
            action_dim=train_envs.action_dim,
            gamma=network_config["gamma"],
            N_t=train_envs.BS_transmit_antennas,
            K=train_envs.num_users,
            P_max=train_envs.users_max_power,
            actor_linear_layers=network_config.get("actor_linear_layers", [128, 128, 128]),
            critic_linear_layers=network_config.get("critic_linear_layers", [128, 128]),
            device=device,
            optimizer=network_config.get("optimizer", "adam"),
            actor_lr=network_config["actor_lr"],
            critic_lr=network_config["critic_lr"],
            tau=network_config.get('tau', 0.0001),
            critic_tau=network_config.get("critic_tau", 0.0001),
            buffer_size=training_config.get("buffer_size"),
            actor_frequency_update=network_config.get('actor_frequency_update', 1),
            critic_frequency_update=network_config.get('critic_frequency_update', 1),
            action_noise_scale=action_noise_scale
        )
    
    elif algorithm_name.lower() == "custom_ddpg":
        network = Custom_DDPG(state_dim = train_envs.state_dim, action_dim = train_envs.action_dim,
                              gamma = network_config["gamma"],
                              N_t = train_envs.BS_transmit_antennas, K = train_envs.num_users,
                              P_max = train_envs.users_max_power,
                              actor_linear_layers = network_config.get("actor_linear_layers",[128,128,128]),
                              critic_linear_layers = network_config.get("critic_linear_layers",[128,128]),
                              device = device,
                              optimizer = network_config.get("optimizer","adam"),
                              actor_lr = network_config["actor_lr"], critic_lr = network_config["critic_lr"],
                              tau = network_config.get('tau',0.00001),
                              critic_tau = network_config.get("critic_tau", 0.00001),
                              buffer_size = training_config.get("buffer_size"),
                              actor_frequency_update = network_config.get('actor_frequency_update',1),
                              critic_frequency_update = network_config.get('critic_frequency_update',1),
                              action_noise_scale  = action_noise_scale,
                              )
        
    elif algorithm_name.lower() == "td3":
        network = TD3(state_dim = train_envs.state_dim, action_dim = train_envs.action_dim,
                      actor_linear_layers = network_config.get("actor_linear_layers",[128,128,128]),
                      critic_linear_layers = network_config.get("critic_linear_layers",[128,128]),
                      device = device,
                      optimizer = network_config.get("optimizer","adam"),
                      gamma = network_config["gamma"],
                      N_t = train_envs.BS_transmit_antennas, K = train_envs.num_users, 
                      P_max = train_envs.users_max_power,
                      actor_lr = network_config["actor_lr"], critic_lr = network_config["critic_lr"],
                      tau = network_config.get('tau',0.001),
                      critic_tau = network_config.get("critic_tau", 0.0001),
                      buffer_size = training_config.get("buffer_size"),
                      actor_frequency_update = network_config.get('actor_frequency_update',1),
                      critic_frequency_update = network_config.get('critic_frequency_update',1),
                      action_noise_scale  = action_noise_scale,
                      use_per = network_config.get("user_per",True),
                      per_alpha= network_config.get("per_alpha",0.6),
                      per_beta_start = network_config.get("per_beta_start", 0.4),
                      per_beta_frames = network_config.get("per_beta_frames ", 100000),
                      per_epsilon = network_config.get("per_beta_frames ", 1e-6),
                      )
    
    elif algorithm_name.lower() == "sac":
        network = SAC(state_dim = train_envs.state_dim, action_dim = train_envs.action_dim,gamma = network_config["gamma"],
                              N_t = train_envs.BS_transmit_antennas, K = train_envs.num_users, 
                              P_max = train_envs.users_max_power,
                              actor_linear_layers = network_config.get("actor_linear_layers",[128,128,128]),
                              critic_linear_layers = network_config.get("critic_linear_layers",[128,128]),
                              device = device,
                              optimizer = network_config.get("optimizer","adam"),
                              actor_lr = network_config["actor_lr"], critic_lr = network_config["critic_lr"],
                              tau = network_config.get('tau',0.0001),
                              critic_tau = network_config.get("critic_tau", 0.0001),
                              buffer_size = training_config.get("buffer_size"),
                              actor_frequency_update = network_config.get('actor_frequency_update',1),
                              critic_frequency_update = network_config.get('critic_frequency_update',1),
                              use_per = network_config.get("user_per",False),
                              per_alpha = network_config.get("per_alpha",0.6),
                              per_beta_start = network_config.get("per_beta_start", 0.4),
                              per_beta_frames = network_config.get("per_beta_frames ", 100000),
                              per_epsilon = network_config.get("per_beta_frames ", 1e-6),
                              )
    
    elif algorithm_name.lower() == "ppo":
        off_policy = False
        network = PPO(state_dim = train_envs.state_dim, action_dim = train_envs.action_dim,gamma = network_config["gamma"],
                              N_t = train_envs.BS_transmit_antennas, K = train_envs.num_users, 
                              P_max = train_envs.users_max_power,
                              n_rollout_envs = n_rollout_train,
                              actor_linear_layers = network_config.get("actor_linear_layers",[128,128,128]),
                              critic_linear_layers = network_config.get("critic_linear_layers",[128,128]),
                              device = device,
                              optimizer = network_config.get("optimizer","adam"),
                              actor_lr = network_config["actor_lr"], critic_lr = network_config["critic_lr"],
                              rollout_size= n_rollout_train * network_config.get("rollout_size",256),clip_range= network_config.get("clip_range",0.01),
                              ppo_epochs = network_config.get("ppo_epochs",5), minibatch_size = network_config.get("batch_size",16),
                              )

    # ============================================================================
    # Configuration Logging and Setup
    # ============================================================================

    # Log timestamp and configuration details
    write_line_to_file(logs_terminal_txt_file, f"\nTimestamp: {timestamp}\n")

    config_dict = config.__dict__
    print(f"\n Configuration: {config.__dict__}\n")

    # Format configuration details for logging
    log_message = "=" * 60 + "\n"
    log_message += "CONFIGURATION DETAILS\n"
    log_message += "=" * 60 + "\n"
    # Environment Configuration
    log_message += "\nEnvironment Configuration:\n"
    log_message += "-" * 30 + "\n"
    env_config = config_dict.get('Environment', {})
    for key, value in env_config.items():
        log_message += f"  {key}: {value}\n"

    # Network Configuration
    log_message += "\nNetwork Configuration:\n"
    log_message += "-" * 30 + "\n"
    network_config = config_dict.get('Network', {})
    for key, value in network_config.items():
        log_message += f"  {key}: {value}\n"

    # Training Parameters
    log_message += "\nTraining Parameters:\n"
    log_message += "-" * 30 + "\n"
    training_params = config_dict.get('Training_parameters', {})
    for key, value in training_params.items():
        log_message += f"  {key}: {value}\n"
    log_message += "=" * 60 + "\n"
    write_line_to_file(logs_terminal_txt_file, log_message)


    # ============================================================================
    # Training Process Launch
    # ============================================================================
    
    # Select appropriate trainer based on algorithm type and environment configuration
    if train_envs.nremotes == 1 and off_policy:
        ofp_single_process_trainer(training_envs=train_envs, network=network, training_config=training_config,
                                log_dir=log_dir,
                                writer=writer, action_noise_activated=action_noise_activated,
                                batch_instead_of_buff=bool(training_config.get("batch_instead_of_buff", False)),
                                eval_env=eval_env,
                                use_rendering=use_rendering)
    elif off_policy:
        ofp_multiprocess_trainer(training_envs=train_envs, network=network, training_config=training_config,
                                log_dir=log_dir,
                                writer=writer, action_noise_activated=action_noise_activated,
                                batch_instead_of_buff=bool(training_config.get("batch_instead_of_buff", False)),
                                eval_env=eval_env,
                                use_rendering=use_rendering)
        
    elif train_envs.nremotes == 1 and not off_policy:
        onp_single_process_trainer(training_envs=train_envs, network=network, training_config=training_config,
                                log_dir=log_dir,
                                writer=writer, action_noise_activated=action_noise_activated,
                                batch_instead_of_buff=bool(training_config.get("batch_instead_of_buff", False)),
                                eval_env=eval_env,
                                use_rendering=use_rendering)
        
    else:
        onp_multiprocess_trainer(training_envs=train_envs, network=network, training_config=training_config,
                                log_dir=log_dir,
                                writer=writer, action_noise_activated=action_noise_activated,
                                batch_instead_of_buff=bool(training_config.get("batch_instead_of_buff", False)),
                                eval_env=eval_env,
                                use_rendering=use_rendering)


if __name__ == "__main__":
    # Configure logging and start training
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main(sys.argv[1:])