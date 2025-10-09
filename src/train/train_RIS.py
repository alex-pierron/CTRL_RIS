"""
Entrypoint to train RIS Duplex RL agents.

Workflow:
 - Parse config filename from CLI and load YAML configuration
 - Build train and optional eval environments (vectorized if requested)
 - Instantiate algorithm (DDPG, TD3, SAC) with configured hyperparameters
 - Launch single or multi-process trainer accordingly

Better Comments legend:
 - TODO: future improvements or refactors (non-functional here)
 - NOTE: important behavior or design intent
 - !: important runtime remark
 - ?: questioning a choice or highlighting an assumption
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


def main(args):

    config_file_name = parse_args()
    
    config = parse_config(config_file_name)

    env_config, network_config, training_config = getattr(config,"Environment"), getattr(config, "Network"), getattr(config, "Training_parameters")
    
    env_seed = env_config["env_seed"]
    
    debbuging =  training_config.get("debbuging", False) 
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    env_name = env_config.get("Environment name", "RIS_Duplex")
    algorithm_name = network_config.get("algorithm", "ddpg")
    experiment_name = f"env_seed{env_seed}_{timestamp}"
    n_rollout_train = training_config.get("n_rollout_threads", 1)
    if debbuging:
        log_dir = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) +"/data" + "/debugging") / algorithm_name / experiment_name
        if not log_dir.exists():
            os.makedirs(str(log_dir))
    else:
        log_dir = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) +"/data" + "/raw") \
        / env_name / config_file_name / algorithm_name / experiment_name
        if not log_dir.exists():
            os.makedirs(str(log_dir))

    writer = SummaryWriter(log_dir)

    logs_terminal_txt_file = f"{log_dir}/log.txt"

    noise_activated = False

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # choosing the device
    cuda_available = torch.cuda.is_available()

    if cuda_available:
        print(f"\n Both CUDA & CPU are available \n")
        write_line_to_file(logs_terminal_txt_file, f" Both CUDA & CPU are available \n", mode='w')
    else:
        print(f"\n Only CPU is available \n")
        write_line_to_file(logs_terminal_txt_file, f" Only CPU is available \n", mode='w')

    # cuda
    if training_config.get("cuda", True) and cuda_available:
        write_line_to_file(logs_terminal_txt_file, " Choose to use CUDA (GPU)... \n")
        print(" Chose to use CUDA (GPU)... \n")
        device = torch.device("cuda:0")  # use cude mask to control using which GPU
        #torch.set_num_threads(training_config.get("n_training_threads",1))
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
    else:
        write_line_to_file(logs_terminal_txt_file," Choose to use CPU... \n")
        print(" Chose to use CPU... \n")
        device = torch.device("cpu")
        torch.set_num_threads(training_config.get("n_training_threads",1))

    # NOTE: Environment initialization (vectorized depending on n_rollout_threads)
    train_envs = make_train_env(env_config, training_config)

    if training_config.get("conduct_evaluation",False):
        if train_envs.nremotes == 1:
            eval_env = make_eval_env(env_config,training_config)
        else:
            eval_env = make_eval_env(env_config, training_config)
    else:
        eval_env = None

    # NOTE: Enable optional situation rendering
    use_rendering = training_config.get("rendering",False)
    
    action_noise_scale = training_config.get("action_noise_scale", 0)
    action_noise_activated = action_noise_scale != 0

    off_policy = True
    if algorithm_name.lower() == "ddpg":
        network = DDPG(state_dim = train_envs.state_dim, action_dim = train_envs.action_dim,
                              gamma = network_config["gamma"],
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
                              action_noise_scale  = action_noise_scale)
    
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

    # * Recording information of the configuration file and printing it in the console

    write_line_to_file(logs_terminal_txt_file, f" \n Timestamp is {timestamp} \n")

    config_dict = config.__dict__
    print(f" Configuration is: {config.__dict__} \n")

    log_message = "Configuration Details:\n"

    # Environment Section
    log_message += "\nEnvironment Configuration:\n"
    env_config = config_dict.get('Environment', {})
    for key, value in env_config.items():
        log_message += f"  {key}: {value}\n"

    # Network Section
    log_message += "\nNetwork Configuration:\n"
    network_config = config_dict.get('Network', {})
    for key, value in network_config.items():
        log_message += f"  {key}: {value}\n"

    # Training Parameters Section
    log_message += "\nTraining Parameters:\n"
    training_params = config_dict.get('Training_parameters', {})
    for key, value in training_params.items():
        log_message += f"  {key}: {value}\n"

    write_line_to_file(logs_terminal_txt_file, log_message)


    # * Launching the process
    
    if train_envs.nremotes == 1 and off_policy:
        ofp_single_process_trainer(training_envs = train_envs, network = network, training_config = training_config,
                                log_dir=log_dir,
                                writer=writer, action_noise_activated= action_noise_activated,
                                batch_instead_of_buff = bool(training_config.get("batch_instead_of_buff", False)),
                                eval_env = eval_env,
                                use_rendering = use_rendering)
    elif off_policy:
        ofp_multiprocess_trainer(training_envs = train_envs, network = network, training_config = training_config,
                                log_dir=log_dir,
                                writer=writer, action_noise_activated= action_noise_activated,
                                batch_instead_of_buff = bool(training_config.get("batch_instead_of_buff", False)),
                                eval_env = eval_env,
                                use_rendering = use_rendering)
        
    elif train_envs.nremotes == 1 and not off_policy:
        onp_single_process_trainer(training_envs = train_envs, network = network, training_config = training_config,
                                log_dir=log_dir,
                                writer=writer, action_noise_activated= action_noise_activated,
                                batch_instead_of_buff = bool(training_config.get("batch_instead_of_buff", False)),
                                eval_env = eval_env,
                                use_rendering = use_rendering)
        
    else:
        onp_multiprocess_trainer(training_envs = train_envs, network = network, training_config = training_config,
                                log_dir=log_dir,
                                writer=writer, action_noise_activated= action_noise_activated,
                                batch_instead_of_buff = bool(training_config.get("batch_instead_of_buff", False)),
                                eval_env = eval_env,
                                use_rendering = use_rendering)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main(sys.argv[1:])