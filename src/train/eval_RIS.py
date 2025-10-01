import numpy as np
import torch
import os
import sys
import logging
import time, datetime
from copy import deepcopy
from pathlib import Path
sys.path.append(os.path.dirname((os.path.dirname(os.path.dirname(os.path.realpath(__file__)))) ) )

from src.runners import *
from src.environment.tools import parse_args, parse_config, write_line_to_file, SituationRenderer
from src.environment.multiprocessing import DummyVecEnv, SubprocVecEnv, make_train_env
from src.algorithms import Article_DDPG, DDPG, Custom_DDPG, TD3

from torch.utils.tensorboard import SummaryWriter


def main(args):

    config_file_name = parse_args()
    
    config = parse_config(config_file_name)

    env_config, network_config, training_config = getattr(config,"Environment"), getattr(config, "Network"), getattr(config, "Training_parameters")
    
    actor_learning_rate = network_config["actor_lr"]
    env_seed = env_config["env_seed"]
    critic_clamping_max_limit = network_config.get("critic_clamping_max_limit", 100)
    
    debbuging =  training_config.get("debbuging", "False") 
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    env_name = env_config.get("Environment name", "RIS_Duplex")
    algorithm_name = network_config.get("algorithm", "Article DDPG")
    experiment_name = f"env_seed{env_seed}_{timestamp}"

    if debbuging:
        log_dir = f"./logs/Debbuging/{algorithm_name}/{experiment_name}"
    else:
        log_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) +"/logs" + "/results") \
        / env_name / config_file_name / algorithm_name / experiment_name
        if not log_dir.exists():
            os.makedirs(str(log_dir))

    writer = SummaryWriter(log_dir)

    logs_terminal_txt_file = f"{log_dir}/log.txt"

    noise_activated = False

    # cuda
    if training_config.get("cuda", True) and torch.cuda.is_available():
        logging.info("choose to use gpu...")
        device = torch.device("cuda:0")  # use cude mask to control using which GPU
        torch.set_num_threads(training_config.get("n_training_threads",1))
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
    else:
        logging.info("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(training_config.get("n_training_threads",1))

    # env init
    envs = make_train_env(env_config, training_config)

    # To plot animation of the phase shif matrix if needed if needed
    if training_config.get("rendering",False):
        renderer =SituationRenderer(M = envs.M, L = envs.L, lambda_h = env_config.get('lambda_h', 0.1),
                                    max_step_per_episode = training_config.get("max_steps_per_episode", 20000),
                                    BS_position = np.array([0, 0]),
                                    RIS_position = np.array(env_config.get('RIS_position', [20, 100])),
                                    users_position=envs.get_users_positions()[0],
                                    eavesdroppers_position= envs.get_eavesdroppers_positions()[0],
                                    eavesdropper_moving=env_config.get('moving_eavesdropper', False))
    else:
        renderer = None

    if algorithm_name.lower() == "article_ddpg":

        network = Article_DDPG(state_dim = envs.state_dim, action_dim = envs.action_dim,
                              gamma = network_config["gamma"],
                              N_t = envs.BS_transmit_antennas, K = envs.num_users, P_max = envs.users_max_power,
                              actor_lr = network_config["actor_lr"], critic_lr = network_config["critic_lr"],
                              tau = network_config.get('tau',0.00001),
                              critic_tau = network_config.get("critic_tau", 0.00001),
                              present_reward=network_config.get("present_reward",False),
                              buffer_size = training_config.get("buffer_size"),
                              actor_frequency_update = network_config.get('actor_frequency_update',1),
                              critic_frequency_update = network_config.get('critic_frequency_update',1),
                              critic_clamping_max_limit = critic_clamping_max_limit)
        
    elif algorithm_name.lower() == "custom_ddpg":
        network = Custom_DDPG(state_dim = envs.state_dim, action_dim = envs.action_dim,
                              gamma = network_config["gamma"],
                              N_t = envs.BS_transmit_antennas, K = envs.num_users, P_max = envs.users_max_power,
                              actor_lr = network_config["actor_lr"], critic_lr = network_config["critic_lr"],
                              tau = network_config.get('tau',0.00001),
                              critic_tau = network_config.get("critic_tau", 0.00001),
                              present_reward=network_config.get("present_reward",False),
                              buffer_size = training_config.get("buffer_size"),
                              actor_frequency_update = network_config.get('actor_frequency_update',1),
                              critic_frequency_update = network_config.get('critic_frequency_update',1),
                              critic_clamping_max_limit = critic_clamping_max_limit)

    elif algorithm_name.lower() == "ddpg":
        network = DDPG(state_dim = envs.state_dim, action_dim = envs.action_dim,
                              gamma = network_config["gamma"],
                              N_t = envs.BS_transmit_antennas, K = envs.num_users, P_max = envs.users_max_power,
                              actor_lr = network_config["actor_lr"], critic_lr = network_config["critic_lr"],
                              tau = network_config.get('tau',0.0001),
                              critic_tau = network_config.get("critic_tau", 0.0001),
                              present_reward=network_config.get("present_reward",False),
                              buffer_size = training_config.get("buffer_size"),
                              actor_frequency_update = network_config.get('actor_frequency_update',1),
                              critic_frequency_update = network_config.get('critic_frequency_update',1),
                              critic_clamping_max_limit = critic_clamping_max_limit)
        
    elif algorithm_name.lower() == "td3":
        network = TD3(state_dim = envs.state_dim, action_dim = envs.action_dim,
                              gamma = network_config["gamma"],
                              N_t = envs.BS_transmit_antennas, K = envs.num_users, P_max = envs.users_max_power,
                              actor_lr = network_config["actor_lr"], critic_lr = network_config["critic_lr"],
                              tau = network_config.get('tau',0.001),
                              critic_tau = network_config.get("critic_tau", 0.0001),
                              present_reward=network_config.get("present_reward",False),
                              buffer_size = training_config.get("buffer_size"),
                              actor_frequency_update = network_config.get('actor_frequency_update',1),
                              critic_frequency_update = network_config.get('critic_frequency_update',1),
                              critic_clamping_max_limit = critic_clamping_max_limit)
        

    config_2 = {
        "env_args": env_config, 
        "network_args":network_config, 
        "training_args": training_config,
        "envs": envs,
        "device": device,
    }

    write_line_to_file(logs_terminal_txt_file, f"{timestamp}\n", mode='w')
    write_line_to_file(logs_terminal_txt_file, f"{config.__dict__}\n")


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # choosing the device
    print(f"Available device is {device}")
    write_line_to_file(logs_terminal_txt_file, f"Available device is {device}")


    noise_activated = bool(env_config.get("noise_activated", False))
    if envs.nremotes == 1:
        single_process_evaluator(envs = envs, network = network, training_config = training_config,
                              log_dir=log_dir,
                              writer=writer, noise_activated= noise_activated,
                              batch_instead_of_buff = bool(training_config.get("batch_instead_of_buff", False)),
                              renderer = renderer )
    else:
        multiprocess_evaluator(envs = envs, network = network, training_config = training_config,
                              log_dir=log_dir,
                              writer=writer, noise_activated= noise_activated,
                              batch_instead_of_buff = bool(training_config.get("batch_instead_of_buff", False)),
                              renderer = renderer)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main(sys.argv[1:])