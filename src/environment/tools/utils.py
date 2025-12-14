import numpy as np
import yaml
import argparse
from pathlib import Path
import numpy as np
import os

# Get the current working directory
current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

# Define the configs folder using os.path.join
configs_folder = os.path.join(current_dir, 'src', 'config_files')

def load_yaml_config(file_name):
    # Construct the full file path
    file_path = configs_folder / file_name
    # Check if the file exists
    if not file_path.exists():
        raise FileNotFoundError(f"The configuration file '{file_name}' does not exist in the 'configs' folder.")
    # Load the YAML configuration file
    with file_path.open('r') as file:
        config = yaml.safe_load(file)
    return config


def get_root_dir():
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    # Get the parent directory
    root_dir = os.path.abspath(os.path.join(current_dir, '..'))
    return root_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Script Configuration")
    # Configuration file
    configuration_file_group = parser.add_argument_group("Configuration file parameters")
    configuration_file_group.add_argument("--config_file", type=str, default="default", help="Config file to use.")
    # Parse the config_file argument
    args, _ = parser.parse_known_args()
    # Return the config_file string
    return args.config_file


def parse_config(filename: str):
    """Parse config file.

    Args:
        config (str): config file name

    Returns:
        (EnvConfig): a custom class which parsing dict into object.
    """
    filepath = os.path.join(get_root_dir(), 'src/config_files', f'{filename}.yaml')
    assert os.path.exists(filepath), \
        f'config path {filepath} does not exist. Please pass in a string that represents the file path to the config yaml.'
    with open(filepath, 'r', encoding='utf-8') as f:
        config_data = yaml.load(f, Loader=yaml.FullLoader)
    return type('EnvConfig', (object,), config_data)


def write_line_to_file(file_path, line, mode='a'):
    """
    Writes a single line to a specified file.

    Parameters:
    file_path (str): The path to the file.
    line (str): The line to write to the file.
    mode (str): The mode to open the file ('w' for write, 'a' for append).
    """
    try:
        with open(file_path, mode) as file:
            file.write(line + '\n')
    except IOError as e:
        print(f"An error occurred while writing to the file: {e}")


def print_and_write(file_path, message, mode='a'):
    """
    Prints and Writes a message to a specified file.

    Parameters:
    file_path (str): The path to the file.
    message (str): The message to write to the file.
    mode (str): The mode to open the file ('w' for write, 'a' for append).
    """
    print(message)
    write_line_to_file(file_path, message, mode=mode)


def select_functions(config_key, authorized_list, authorized_dict, env_config):
    """
    Selects functions based on the configuration and authorized list.

    Parameters:
    - config_key: The key to retrieve the list of functions from the environment configuration.
    - authorized_list: A list of authorized function names.
    - authorized_dict: A dictionary mapping authorized function names to their corresponding functions.
    - env_config: The environment configuration object.

    Returns:
    - A dictionary of selected functions.

    Raises:
    - ValueError: If any of the chosen functions are not in the authorized list.
    """
    if config_key == 'decisive_reward_functions':
        default = ["baseline_reward"]
    elif config_key == "informative_reward_functions":
        default = []
    elif config_key == "fairness_functions":
        default = ["jain_fairness"]
    else:
        raise ValueError(f"Wrong config key used during intialiazing the selection of rewards/fairness functions")
    
    # Get the functions chosen by the user from the environment configuration
    functions_chosen_by_user = env_config.get(config_key, default)

    # Convert the authorized list to a NumPy array
    authorized_array = np.array(authorized_list)

    # Convert the user's chosen functions to lowercase and to a NumPy array
    chosen_by_user_array = np.array([func.lower() for func in functions_chosen_by_user])

    # Check if all elements in chosen_by_user_array are in authorized_array
    checking_function_validity = np.all(np.isin(chosen_by_user_array, authorized_array))

    if checking_function_validity:
        # Construct the dictionary containing only the functions chosen by the user
        chosen_function_dict = {func: authorized_dict[func] for func in chosen_by_user_array if func in authorized_dict}
        return chosen_function_dict
    else:
        # Raise a ValueError with a descriptive message
        invalid_functions = chosen_by_user_array[~np.isin(chosen_by_user_array, authorized_array)]
        raise ValueError(f"Invalid function(s) chosen: {', '.join(invalid_functions)}. Authorized functions are: {', '.join(authorized_array)}.")
