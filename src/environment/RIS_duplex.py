"""
Gym environment for a RIS duplex communication system.

Responsibilities:
- Initialize physical parameters and topology from a config dict
- Generate Rician channels (LoS + NLoS) and keep them updated
- Build the observation state for RL agents
- Apply actions to update RIS phases `Theta` and BS beamforming `W`
- Compute SINR-related metrics, rewards, and fairness indices
- Provide mobility and power pattern utilities for analysis

Better Comments legend used throughout the file:
- TODO: future improvements or refactors (non-functional)
- NOTE: important behavior or design intent
- !: important runtime remark
- ?: questioning a choice or highlighting an assumption
"""
import os
import sys
import torch
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .tools import PositionGenerator, select_functions
from copy import deepcopy
from .rewards import *
from .physics import *
from .mobility import *
from time import time as time_for_seed

class RIS_Duplex(gym.Env):
    """
    RIS_Duplex Environment

    This class defines a custom Gym environment for simulating a Reconfigurable Intelligent Surface (RIS)
    duplex communication system. It includes various parameters for configuring the environment, such as
    the number of antennas, users, RIS elements, as well as power and noise settings.

    Attributes:
        N_t (int): Number of transmitting antennas at the base station.
        N_r (int): Number of receiving antennas at the base station.
        K (int): Number of legitimate users.
        L (int): Number of passive eavesdroppers.
        M (int): Number of reflective elements in the RIS.
        P_max (float): Maximum transmit power at the base station in Watts.
        P_users (np.ndarray): Maximum transmit power for each user in Watts.
        lambda_h (float): Wavelength parameter for the environment.
        d_h (np.ndarray): Inter-element spacing in the ULAs.
        rician_factor (float): Rician factor for the environment.
        sigma_k_squared (float): Noise power density for the downlink signal in Watts for the users.
        delta_k_squared (float): Noise power density for the uplink signal in Watts for the users.
        mu_d_l_squared (float): Noise power density for the downlink signal in Watts for the eavesdroppers.
        mu_u_l_squared (float): Noise power density for the uplink signal in Watts for the eavesdroppers.
        SI_coef (float): Self-interference coefficient.
        delta_squared (float): Noise power density in dBm.
        kappa (np.ndarray): Hardware impairment coefficients.
        verbose (bool): Verbosity flag.
        moving_eavesdroppers (bool): Flag indicating if eavesdroppers are moving.
        users_spawn_limits (np.ndarray): Spawn limits for users.
        eavesdropper_spawn_limits (np.ndarray): Spawn limits for eavesdroppers.
        RIS_position (np.ndarray): Position of the RIS.
        render_mode (str): Render mode for the environment.
        size (list): Size of the environment in meters.
        seed (int): Seed for random number generation.
        numpy_rng (np.random.Generator): Random number generator.
        position_generator (PositionGenerator): Generator for user and eavesdropper positions.
        Theta (np.ndarray): RIS phase shift matrix.
        Phi (np.ndarray): Phase noise matrix.
        W (np.ndarray): Transmit power matrix from BS to users.
        window_size (int): Size of the PyGame window.
        beamforming_matrix_dim (int): Dimension of the beamforming matrix.
        phase_shift_matrix_action_dim (int): Dimension of the phase shift matrix action.
        action_dim (int): Total dimension of the action space.
        action_space (gym.spaces.Box): Action space definition.
        state_dim (int): Dimension of the state space.
        observation_space (gym.spaces.Dict): Observation space definition.
        BS_position (np.ndarray): Position of the base station.
        state (np.ndarray): Current state of the environment.
        done (bool): Flag indicating if the episode is done.
        num_episode (int): Current episode number.
        num_step (int): Current step number.
        previous_rate_part (np.ndarray): Previous rate part of the state.
        max_sinr_b_k (dict): Maximum SINR for BS to user.
        max_sinr_s_k (dict): Maximum SINR for user to BS.
        sinr_b_k (float): SINR for BS to user.
        sinr_s_k (float): SINR for user to BS.
        max_sinr_e_d_k_l (dict): Maximum SINR for eavesdropper downlink.
        max_sinr_e_u_k_l (float): Maximum SINR for eavesdropper uplink.
        user_reward_detail (np.ndarray): Detailed reward information for each user.
    """

    # ---------------------------------------------- Initialization and Reset --------------------------------------------------------------
    def __init__(self, environment_config_dict: dict):
        """Initialize the environment from a configuration dictionary.

        Args:
            environment_config_dict (dict): Parameters controlling topology,
                power/noise, seeds, and reward settings. Values are read and
                stored; behavior is unchanged by this refactor.
        """
        # NOTE: Parse and store environment configuration once
        self.env_config = environment_config_dict

        # =====================================================================
        # Topology & sizes
        # =====================================================================
        self.N_t = self.env_config['num_BS_antennas'][0]
        self.N_r = self.env_config['num_BS_antennas'][1]
        self.K = self.env_config['num_users']

        self._num_eavesdroppers = self.env_config.get('num_eavesdroppers', 0)
        self.eavesdropper_active = (self._num_eavesdroppers > 0)

        self._M = self.env_config['num_RIS_elements']

        # =====================================================================
        # Power & noise
        # =====================================================================
        self.P_max = dbm_to_watts(self.env_config['BS_max_power'])
        self.P_users = np.ones(self.K) * self.env_config.get('user_transmit_power', 100) * 1e-3  # in W
        self._lambda_h = self.env_config.get('lambda_h', 0.1) # in meter
        self._d_h = self.env_config.get('d_h', self._lambda_h / 2) * np.ones(4) # in meter
        self.los_only = self.env_config.get('los_only', False) 
        self.rician_factor = self.env_config.get('rician_factor', 10)
        self.channel_bandwidth = self.env_config.get('channel_bandwidth', 10)  # in MHz
        self.delta_squared = self.env_config.get('noise_power_density', -174)  # in dBm/Hz
        self.delta_k_squared = dBm_Hz_to_Watts(self.delta_squared, self.channel_bandwidth)  # in W
        self.sigma_k_squared = dBm_Hz_to_Watts(self.delta_squared, self.channel_bandwidth)  # in W

        # Initialize comprehensive information storage using dictionaries
        self.user_info = {}
        self.eavesdropper_info = {}
        
        # Initialize user information dictionaries
        for k in range(self.K):
            self.user_info[k] = {
                'downlink': {
                    'sinr_ratio': 0.0,
                    'sinr_db': -np.inf,
                    'signal_power_watts': 0.0,
                    'signal_power_dbm': -np.inf,
                    'interference_noise_watts': 0.0,
                    'interference_noise_dbm': -np.inf,
                    'cumulative_signal_watts': 0.0,
                    'cumulative_interference_watts': 0.0,
                    'min_signal_watts': np.inf,
                    'max_signal_watts': -np.inf,
                    'avg_signal_dbm': -np.inf,
                    'avg_sinr_ratio': 0.0,
                    'avg_sinr_db': -np.inf
                },
                'uplink': {
                    'sinr_ratio': 0.0,
                    'sinr_db': -np.inf,
                    'signal_power_watts': 0.0,
                    'signal_power_dbm': -np.inf,
                    'interference_noise_watts': 0.0,
                    'interference_noise_dbm': -np.inf,
                    'cumulative_signal_watts': 0.0,
                    'cumulative_interference_watts': 0.0,
                    'min_signal_watts': np.inf,
                    'max_signal_watts': -np.inf,
                    'avg_signal_dbm': -np.inf,
                    'avg_sinr_ratio': 0.0,
                    'avg_sinr_db': -np.inf
                }
            }
        
        # Initialize eavesdropper information dictionaries
        for l in range(self._num_eavesdroppers):
            self.eavesdropper_info[l] = {
                'downlink': {
                    'sinr_ratios': np.zeros(self.K),  # SINR for each user
                    'signal_powers_watts': np.zeros(self.K),
                    'signal_powers_dbm': np.full(self.K, -np.inf),
                    'interference_noise_watts': np.zeros(self.K),
                    'interference_noise_dbm': np.full(self.K, -np.inf),
                    'cumulative_signal_watts': np.zeros(self.K),
                    'cumulative_interference_watts': np.zeros(self.K),
                    'min_signal_watts': np.full(self.K, np.inf),
                    'max_signal_watts': np.full(self.K, -np.inf),
                    'avg_signal_dbm': np.full(self.K, -np.inf),
                    'avg_sinr_ratios': np.zeros(self.K),
                    'avg_sinr_db': np.full(self.K, -np.inf)
                },
                'uplink': {
                    'sinr_ratios': np.zeros(self.K),  # SINR for each user
                    'signal_powers_watts': np.zeros(self.K),
                    'signal_powers_dbm': np.full(self.K, -np.inf),
                    'interference_noise_watts': np.zeros(self.K),
                    'interference_noise_dbm': np.full(self.K, -np.inf),
                    'cumulative_signal_watts': np.zeros(self.K),
                    'cumulative_interference_watts': np.zeros(self.K),
                    'min_signal_watts': np.full(self.K, np.inf),
                    'max_signal_watts': np.full(self.K, -np.inf),
                    'avg_signal_dbm': np.full(self.K, -np.inf),
                    'avg_sinr_ratios': np.zeros(self.K),
                    'avg_sinr_db': np.full(self.K, -np.inf)
                }
            }
        
        # Legacy arrays for backward compatibility
        self.sinr_downlink_users = np.zeros(self.K)
        self.sinr_downlink_signals = np.zeros(self.num_users)
        self.sinr_downlink_interfs = np.zeros(self.num_users)
        self.sinr_uplink_users = np.zeros(self.K)
        self.uplink_signal_strength = np.zeros(self.K)
        self.sinr_downlink_details = {}
        self.downlink_signal_strength = np.zeros(self.K)
        self.downlink_sinr_average = np.zeros(self.K)
        self.min_downlink_signal_strength = np.full(self.K, np.inf)
        self.max_downlink_signal_strength = np.full(self.K, -np.inf)

        self.mu_d_l_squared = self.env_config.get('mu_d_l_squared', 3.981e-14)
        self.mu_u_l_squared = self.env_config.get('mu_u_l_squared', 3.981e-14)

        self.mu_1 = self.env_config.get('mu_1', 2)
        self.mu_2 = self.env_config.get('mu_2', 2)
        self.SI_coef = self.env_config.get('SI_coefficient', 1)
        self.kappa = np.array(self.env_config.get('HWI_coefficients', [0.01] * 4))
        self.verbose = self.env_config.get('verbose', False)

        # =====================================================================
        # Geometry & runtime controls
        # =====================================================================
        self._RIS_position = np.array(self.env_config.get('RIS_position', [20, 100]))
        self.render_mode = self.env_config.get('render_mode', None)
        self.size = self.env_config.get('size', [200, 200])

        # =====================================================================
        # Seeding & RNG
        # =====================================================================
        self.random_random_seed = self.env_config.get("random_random_seed", False)
        self.seed = self.env_config.get('env_seed', 42)
        if self.random_random_seed:
            self.numpy_rng = np.random.default_rng( int(time_for_seed()) )
        else:
            self.numpy_rng = np.random.default_rng(self.seed)

        self.user_position_changing = self.env_config.get('users_position_changing', True)

        self.target_secrecy_rate = self.env_config.get('target_secrecy_rate', 1.5)
        self.target_date_rate = self.env_config.get('target_date_rate', 2.5)

        self.p_f = self.env_config.get('p_f', 2)
        self.additional_state_info = self.env_config.get('additional_state_info', False)

        self.alpha_fairness = self.env_config.get('alpha_fairness', 0.5)
        self.downlink_capacity_reward_threshold = self.env_config.get('downlink_capacity_reward_threshold', 0.75)
        self.uplink_capacity_reward_threshold = self.env_config.get('uplink_capacity_reward_threshold', 0.15)

        self.informative_rewards = {}
        self.decisive_rewards = {}

        # Select reward functions
        self.decisive_reward_functions = select_functions(
            'decisive_reward_functions', authorized_reward_list, authorized_reward_dict, self.env_config
        )
        self.informative_reward_functions = select_functions(
            'informative_reward_functions', authorized_reward_list, authorized_reward_dict, self.env_config
        )
        self.qos_p = self.env_config.get('qos_p', 2)
        self.print_info = self.env_config.get('print_info_env', False)

        if "baseline_reward" not in self.informative_reward_functions or self.decisive_reward_functions:
            self.compute_extra_basic_reward = True
        else:
            self.compute_extra_basic_reward = False

        # Select fairness functions
        self.fairness_functions = select_functions(
            'fairness_functions', authorized_fairness_list, authorized_fairness_dict, self.env_config
        )
        self.decisive_current_rewards = {}  # Dict to store all info for multiple rewards for the agent
        self.informative_current_rewards = {}  # Dict for informative metrics (not used for training)
        self.basic_reward_per_user = np.zeros(self.K)
        self.basic_reward_total = 0
        self.current_fairness = 0
        self.previous_fairness = 0

        # =====================================================================
        # Spatial domain: spawn limits & mobility
        # =====================================================================
        self.users_spawn_limits = np.array(self.env_config.get('user_spawn_limits', [[100, 200], [0, 100]]))
        self.eavesdropper_spawn_limits = np.array(self.env_config.get('eavesdropper_spawn_limits', [[100, 200], [0, 100]]))

        self.eaves_position_changing = self.env_config.get('eaves_position_changing', True)
        self._moving_eavesdroppers = self.env_config.get('moving_eavesdroppers', False)

        if self._moving_eavesdroppers:
            modes = {"random_walk": 0, "brownian": 1, "levy_flight": 2}
            self.mobility_model_eavesdroppers = modes[self.env_config.get('mobility_model_eavesdroppers', "brownian")]
        # NOTE: Position generator encapsulates user/eaves spawn logic
        min_distance = self.env_config.get('min_distance', 2)
        
        self.position_generator = PositionGenerator(
            self.num_users,
            self.users_spawn_limits,
            num_eavesdroppers=self._num_eavesdroppers,
            RIS_position=self._RIS_position,
            numpy_generator=self.numpy_rng,
            min_distance = min_distance,
        )

        self.angles = np.linspace(0, 2 * np.pi, 360)

        # =====================================================================
        # Action & observation structures
        # =====================================================================
        self._Theta = np.eye(self._M, dtype=complex)
        self._Phi = np.eye(self._M, self._M, dtype =complex)
        self._W = np.ones((self.N_t, self.K), dtype=complex)
        self.window_size = 600
        self.beamforming_matrix_dim = 2 * self.N_t * self.K
        self.phase_shift_matrix_action_dim = 2 * self._M
        self._action_dim = 2 * self._M + 2 * self.N_t * self.K

        action_low_bound = -np.inf + np.zeros(self.action_dim)
        action_low_bound[-2 * self._M: -self._M] = -1
        action_low_bound[-self._M:] = 0
        action_high_bound = deepcopy(-action_low_bound)
        action_high_bound[-self._M:] = 2 * np.pi

        self._action_space = spaces.Box(
            low=action_low_bound, high=action_high_bound, shape=(self.action_dim,), dtype=np.float64
        )

        #! FOR DEBUGGING PURPOSE
        self.bjornson = self.env_config.get('bjornson', False)

        self.debugging = self.env_config.get('debugging', False)  # downlink, in W
        if self.debugging:
            self.test_point_for_user = np.array(self.env_config.get('test_point_for_user', [122,88]))
            self.test_point_for_BS = np.array(self.env_config.get('test_point_for_BS', [20,20]))

        self.users_positions = self.position_generator.generate_random_users_positions()

        if self.eavesdropper_active:
            self.eavesdroppers_positions = self.position_generator.generate_random_eavesdroppers_positions()

        # Use two seeds to make sure users are frozen (can be optimized in future work)
        if not self.user_position_changing:
            fixed_position = self.env_config.get('users_fixed_positions', None)
            if fixed_position is not None:
                self.users_positions = fixed_position
            else:
                if self.random_random_seed:  
                    self.second_seed = int(time_for_seed())
                else:
                    self.second_seed = self.env_config.get('user_seed', 42)
                self.second_numpy_rng = np.random.default_rng(self.second_seed)
                self.second_position_generator = PositionGenerator(
                    self.num_users,
                    self.users_spawn_limits,
                    self.RIS_position,
                    numpy_generator=self.second_numpy_rng
                )
                self.users_positions = self.second_position_generator.generate_random_users_positions()

        if not self.eaves_position_changing:
            fixed_position = self.env_config.get('eaves_fixed_positions', None)
            if fixed_position is not None:
                self.eavesdroppers_positions = fixed_position

            else:  
                self.eavesdroppers_positions = self.position_generator.generate_new_eavesdroppers_positions(users_positions=self.users_positions) 


        # =====================================================================
        # State dimensions
        # =====================================================================
        if self.additional_state_info:
            self.previous_rate_dim = 2 * self.K + 2 * self.K + 3  # More fine-grained info
        else:
            self.previous_rate_dim = 4  # Limited info

        if self.eavesdropper_active:
            cascaded_channel_dim = 2 * self.K**2 + 4 * self.K * self._num_eavesdroppers + 2 * self.N_r * self.K 
        else:
            cascaded_channel_dim = 2 * self.K**2 + 4 + 2 * self.N_r * self.K

        phase_noise_dim = self._M
        BS_transmit_power_dim = 2 * self.K
        previous_received_power_dim = 2 * self.K

        self._state_dim = (
            self.previous_rate_dim + cascaded_channel_dim + phase_noise_dim
            + self.action_dim + BS_transmit_power_dim + previous_received_power_dim
        )

        self._observation_space = spaces.Dict(
            {
                "previous rate": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(self.previous_rate_dim,), dtype=np.float64
                ),
                "cascaded channel": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(cascaded_channel_dim,), dtype=np.float64
                ),
                "phase noise": spaces.Box(
                    low=-np.pi / 2, high=np.pi / 2, shape=(self._M,), dtype=np.float64
                ),
                "previous action taken": spaces.Box(
                    low=action_low_bound, high=action_high_bound, shape=(self.action_dim,), dtype=np.float64
                ),
                "BS transmit power": spaces.Box(
                    low=0, high=np.inf, shape=(BS_transmit_power_dim,), dtype=np.float64
                ),
                "previous received power from users": spaces.Box(
                    low=0, high=np.inf, shape=(previous_received_power_dim,), dtype=np.float64
                ),
            }
        )

        # =====================================================================
        # Runtime counters & ephemeral state
        # =====================================================================
        self._BS_position = np.array([0, 0])
        self.state = np.zeros(self.state_dim)
        self.done = None
        self._num_episode = 0
        self._num_step = 0
        self.previous_rate_part = np.zeros(self.previous_rate_dim)
        
        # =====================================================================
        # Performance optimization caches
        # =====================================================================
        self._channel_cache = {}
        self._last_user_positions = None
        self._last_eaves_positions = None
        self._cached_gains = None
        self._cached_matrix_products = {}
        self._last_theta_phi = None
        self._last_W = None
        self._cached_sinr = {}
        self._sinr_cache_valid = False

        self.test = 0

        # Debugging
        self.max_sinr_b_k = {"SINR_in_db": -np.inf}
        self.max_downlink_interf_k = -np.inf * np.ones(self.K)
        self.max_sinr_s_k = {"SINR_in_db": -np.inf}
        self.max_sinr_e_d_k_l = {"SINR_in_db": -np.inf}
        self.max_sinr_e_u_k_l = -np.inf
        self.sinr_b_k = -np.inf
        self.sinr_s_k = -np.inf
        self.sinr_e_d_k_l = -np.inf
        self.sinr_e_u_k_l = -np.inf
        self.user_reward_detail = np.zeros(self.K, dtype=object)

        # NOTE: Initialize first episode
        self.reset(seed=self.seed)
        pass

    # ---------------------------------------------- Properties --------------------------------------------------------------

    @property
    def M(self):
        """Number of RIS elements."""
        return self._M

    @property
    def d_h(self):
        """Inter-element spacing in the RIS, supposed to be a ULA (Uniform Linear Array)."""
        return self._d_h

    @property
    def lambda_h(self):
        """Wavelength parameter."""
        return self._lambda_h

    @property
    def RIS_position(self):
        """Position of the RIS."""
        return self._RIS_position

    @property
    def channel_matrices(self):
        """Channel matrices dictionary."""
        return self._channel_matrices

    def gains_transmitter_ris_receiver(self):
        """Channel matrices dictionary."""
        return self._gains_transmitter_ris_receiver

    @property
    def BS_position(self):
        """Position of the base station."""
        return self._BS_position

    @property
    def moving_eavesdroppers(self):
        """Whether eavesdroppers are moving."""
        return self._moving_eavesdroppers

    @property
    def state_dim(self):
        """Dimension of the state space."""
        return self._state_dim

    @property
    def num_eavesdroppers(self):
        """Number of eavesdroppers."""
        return self._num_eavesdroppers

    @property
    def action_dim(self):
        """Dimension of the action space."""
        return self._action_dim

    @property
    def num_users(self):
        """Number of users."""
        return self.K

    @property
    def maximum_power(self):
        """Maximum transmit power."""
        return self.P_max

    @property
    def BS_transmit_antennas(self):
        """Number of BS transmit antennas."""
        return self.N_t

    @property
    def num_step(self):
        """Current step number."""
        return self._num_step

    @property
    def W(self):
        """Beamforming matrix."""
        return self._W

    @property
    def Phi(self):
        """Phase noise matrix."""
        return self._Phi

    @property
    def Theta(self):
        """RIS phase shift matrix."""
        return self._Theta

    @property
    def Theta_Phi(self):
        """Product of Theta and Phi."""
        return self._Theta_Phi

    @property
    def num_episode(self):
        """Current episode number."""
        return self._num_episode

    @property
    def get_episode_action_noise(self):
        """Returns the action noise for the current episode."""
        return self.episode_action_noise

    @property
    def observation_space(self) -> gym.Space:
        """Returns the observation space of the environment."""
        return self._observation_space

    @property
    def action_space(self) -> gym.Space:
        """Returns the action space of the environment."""
        return self._action_space

    # ---------------------------------------------- Reset and Initialization --------------------------------------------------------------

    def reset(self, seed=None, chosen_difficulty_config=None):
        """Reset environment state at the start of a new episode.

        Args:
            seed: Optional RNG seed for this reset.
            chosen_difficulty_config: Optional curriculum tuple to constrain
                position generation and scenario difficulty.
        """
        if chosen_difficulty_config is not None:
            # NOTE: Curriculum Learning: update spatial generation constraints
            self.position_generator.update_generation_condition(
                chosen_difficulty_config[0],
                chosen_difficulty_config[1],
                chosen_difficulty_config[2],
                chosen_difficulty_config[3],
                chosen_difficulty_config[4],
                chosen_difficulty_config[5],
            )

        self._initialize_reset(seed)

        # Generate random user positions
        if self.user_position_changing:
            if chosen_difficulty_config:
                self.users_positions = self.position_generator.generate_new_users_positions()
                if self.eavesdropper_active and self.eaves_position_changing:
                    self.eavesdroppers_positions = self.position_generator.generate_new_eavesdroppers_positions(
                        users_positions=self.users_positions
                    )
            else:
                self.users_positions = self.position_generator.generate_random_users_positions()
                if self.eavesdropper_active and self.eaves_position_changing:
                    self.eavesdroppers_positions = self.position_generator.generate_random_eavesdroppers_positions()

        # Initialize channels and state
        self._initialize_channels()
        self._initialize_state()

    def determined_reset(self, user_position, seed=None):
        """Reset to a determined state with explicit user positions.

        Args:
            user_position (array-like): Positions to use for users.
            seed: Optional RNG seed.
        """
        self._initialize_reset(seed)
        self.users_positions = np.array(user_position)
        self._initialize_channels()
        self._initialize_state()

    def _initialize_reset(self, seed):
        """Common initialization logic shared by reset paths."""
        super().reset(seed=seed)
        self._num_episode += 1
        self.previous_action = np.zeros(self.action_dim)
        self.current_actions = np.zeros(self.action_dim)
        self.episode_action_noise = torch.zeros(self.action_dim)
        self.previous_actions = np.zeros(self.action_dim)
        self.previous_G_1D = np.zeros(shape=(self.K, self.K), dtype=complex)
        self.previous_rate_part = np.zeros(self.previous_rate_dim)
        self._num_step = 0

        # Reset comprehensive information storage
        for k in range(self.K):
            self.user_info[k]['downlink'].update({
                'sinr_ratio': 0.0,
                'sinr_db': -np.inf,
                'signal_power_watts': 0.0,
                'signal_power_dbm': -np.inf,
                'interference_noise_watts': 0.0,
                'interference_noise_dbm': -np.inf,
                'cumulative_signal_watts': 0.0,
                'cumulative_interference_watts': 0.0,
                'min_signal_watts': np.inf,
                'max_signal_watts': -np.inf,
                'avg_signal_dbm': -np.inf,
                'avg_sinr_ratio': 0.0,
                'avg_sinr_db': -np.inf
            })
            self.user_info[k]['uplink'].update({
                'sinr_ratio': 0.0,
                'sinr_db': -np.inf,
                'signal_power_watts': 0.0,
                'signal_power_dbm': -np.inf,
                'interference_noise_watts': 0.0,
                'interference_noise_dbm': -np.inf,
                'cumulative_signal_watts': 0.0,
                'cumulative_interference_watts': 0.0,
                'min_signal_watts': np.inf,
                'max_signal_watts': -np.inf,
                'avg_signal_dbm': -np.inf,
                'avg_sinr_ratio': 0.0,
                'avg_sinr_db': -np.inf
            })
        
        # Reset eavesdropper information
        for l in range(self._num_eavesdroppers):
            self.eavesdropper_info[l]['downlink'].update({
                'sinr_ratios': np.zeros(self.K),
                'signal_powers_watts': np.zeros(self.K),
                'signal_powers_dbm': np.full(self.K, -np.inf),
                'interference_noise_watts': np.zeros(self.K),
                'interference_noise_dbm': np.full(self.K, -np.inf),
                'cumulative_signal_watts': np.zeros(self.K),
                'cumulative_interference_watts': np.zeros(self.K),
                'min_signal_watts': np.full(self.K, np.inf),
                'max_signal_watts': np.full(self.K, -np.inf),
                'avg_signal_dbm': np.full(self.K, -np.inf),
                'avg_sinr_ratios': np.zeros(self.K),
                'avg_sinr_db': np.full(self.K, -np.inf)
            })
            self.eavesdropper_info[l]['uplink'].update({
                'sinr_ratios': np.zeros(self.K),
                'signal_powers_watts': np.zeros(self.K),
                'signal_powers_dbm': np.full(self.K, -np.inf),
                'interference_noise_watts': np.zeros(self.K),
                'interference_noise_dbm': np.full(self.K, -np.inf),
                'cumulative_signal_watts': np.zeros(self.K),
                'cumulative_interference_watts': np.zeros(self.K),
                'min_signal_watts': np.full(self.K, np.inf),
                'max_signal_watts': np.full(self.K, -np.inf),
                'avg_signal_dbm': np.full(self.K, -np.inf),
                'avg_sinr_ratios': np.zeros(self.K),
                'avg_sinr_db': np.full(self.K, -np.inf)
            })

        # Legacy arrays for backward compatibility
        self.sinr_downlink_users = np.zeros(self.K)
        self.sinr_downlink_signals = np.zeros(self.num_users)
        self.sinr_downlink_interfs = np.zeros(self.num_users)
        self.sinr_downlink_details = {}
        self.downlink_signal_strength = np.zeros(self.K)
        self.downlink_sinr_average = np.zeros(self.K)
        self.min_downlink_signal_strength = np.full(self.K, np.inf)
        self.max_downlink_interf_k = -np.inf * np.ones(self.K)
        self.max_downlink_signal_strength = np.full(self.K, -np.inf)

        
        self.sinr_uplink_users = np.zeros(self.K)
        self.uplink_signal_strength = np.zeros(self.K) 

        # For curriculum learning episode success/failure
        self.downlink_episode_success_condition = np.zeros(self.K, dtype=np.float16)
        self.uplink_episode_success_condition = np.zeros(self.K, dtype=np.float16)
        if self.eavesdropper_active:
            self.best_eavesdropping_episode_success_condition = np.zeros(self.K, dtype=np.float16)
            """self.downlink_eavesdropping_episode_success_condition = np.zeros(self.K, dtype=np.float16)
            self.uplink_eavesdropping_episode_success_condition = np.zeros(self.K, dtype=np.float16)"""
        self.done = False

    def _initialize_channels(self):
        """Initialize channels and cached matrices for the episode."""
        self._channel_matrices = {}
        self.F = np.zeros((self.N_r, self.K), dtype=complex)

        self._gains_transmitter_ris_receiver = np.zeros(self.K + self._num_eavesdroppers)
        self._gains_transmitter_ris = 0
        # Compute BS -> RIS channel
    
        self._channel_matrices["H_BS_RIS"] = rician_fading_channel(
            transmitter_position=self._BS_position,
            receiver_position=self._RIS_position,
            W_h_t=self.N_t, W_h_r=self._M,
            d_h_tx=self._d_h[0], d_h_rx=self._d_h[1],
            lambda_h=self._lambda_h,
            epsilon_h=self.rician_factor,
            numpy_generator=self.numpy_rng, bjornson = self.bjornson,los_only = self.los_only
        )

        if self.debugging:
            
            self_second_np_rng = np.random.default_rng(123)
            self._channel_matrices["H_BS_RIS_Test"] = rician_fading_channel(
            transmitter_position=self._BS_position,
            receiver_position=self._RIS_position,
            W_h_t=self.N_t, W_h_r=self._M,
            d_h_tx=self._d_h[0], d_h_rx=self._d_h[1],
            
            lambda_h=self._lambda_h,
            epsilon_h=self.rician_factor,
            numpy_generator=self_second_np_rng ,bjornson = True,
            )
        
            self_second_np_rng = np.random.default_rng(123)

            self._channel_matrices["H_BS_Test_Point_BS"] = np.array([rician_fading_channel(
                transmitter_position=self._BS_position,
                receiver_position=self.test_point_for_BS,
                W_h_t=self.N_t, W_h_r=1,
                d_h_tx=self._d_h[1], d_h_rx=self._d_h[2],
                lambda_h=self._lambda_h,
                epsilon_h=self.rician_factor,
                numpy_generator=self_second_np_rng, nlos_only = True ,bjornson = True,
            )])

        # Compute RIS -> BS channel
        self._channel_matrices["H_RIS_BS"] = rician_fading_channel(
            transmitter_position=self._RIS_position,
            receiver_position=self._BS_position,
            W_h_t=self._M, W_h_r=self.N_r,
            d_h_tx=self._d_h[1], d_h_rx=self._d_h[0],
            lambda_h=self._lambda_h,
            epsilon_h=self.rician_factor,
            numpy_generator=self.numpy_rng, bjornson = self.bjornson,los_only = self.los_only
        )

        # Draw the first phase noise matrix
        self.phases = self.numpy_rng.uniform(-90, 90, self._M)
        self.phases = np.deg2rad((360 + self.phases) % 360)
        self._Phi = np.diag(np.exp(1j * self.phases))
        self._Theta = np.ones_like(self._Phi)

        # Further initialization
        # NOTE: Initialize composite matrices and derived terms
        self.compute_theta_phi()
        self.compute_WWH()
        self.compute_all_channels()

        # Compute decoding matrix after all channels are available
        self.compute_decoding_matrix()

    def _initialize_state(self):
        """Initialize current and next state buffers for the episode."""
        self.current_state = self.get_state()
        self.next_state = np.zeros_like(self.current_state)

    def compute_theta_phi(self):
        """Update cached products of `Theta` and `Phi` used across formulas."""
        diagonal_elements = np.diag(self._Theta @ self._Phi) 
        self._Theta_Phi = np.diag(diagonal_elements / np.abs(diagonal_elements)) 
        self.Phi_H_Theta_H = self._Phi.conj().T @ self._Theta.conj().T
        #self.Phi_H_Theta_H = self._Phi.T @ self._Theta.T
        pass

    def compute_WWH(self):
        """Update cached `W @ W^H` and its diagonal; reused in SINR terms."""
        self.WWH = self._W @ self._W.conj().T
        self.diag_matrix_WWH = np.diag(np.diag(self.WWH)).real
        pass
    
    def _cache_matrix_products(self):
        """Cache expensive matrix products used in state building."""
        # Check if matrices have changed
        theta_phi_changed = (self._last_theta_phi is None or 
                            not np.array_equal(self._last_theta_phi, self._Theta_Phi))
        W_changed = (self._last_W is None or 
                    not np.array_equal(self._last_W, self._W))
        
        if theta_phi_changed or W_changed:
            # Cache BS-RIS-LU channel products
            h_d = self._channel_matrices["H_RIS_Users"].squeeze(axis=1)  # Shape: (K,M)
            self._cached_matrix_products["G_1D"] = h_d @ self._Theta_Phi @ self._channel_matrices["H_BS_RIS"] @ self._W
            
            # Cache LU-RIS-BS channel products  
            h_u = self._channel_matrices["H_Users_RIS"].squeeze(axis=2).T  # Shape: (M, K)
            self._cached_matrix_products["LU_BS_RIS"] = self._channel_matrices["H_RIS_BS"] @ self._Theta_Phi @ h_u
            
            # Cache eavesdropper products if active
            if self.eavesdropper_active:
                G_2D = self._channel_matrices["H_RIS_Eaves_downlink"].squeeze(axis=1)  # Shape: (L, M)
                self._cached_matrix_products["BS_RIS_EAVES"] = G_2D @ self._Theta_Phi @ self._channel_matrices["H_BS_RIS"] @ self._W
                
                g_u = self._channel_matrices["H_RIS_Eaves_uplink"].squeeze(axis=1)  # Shape: (L, M)
                self._cached_matrix_products["LU_BS_EAVES"] = g_u @ self._Theta_Phi @ h_u
            
            # Update cache markers
            self._last_theta_phi = self._Theta_Phi.copy()
            self._last_W = self._W.copy()
    
    def _update_user_info_downlink(self, k, sinr_ratio, sinr_db, signal, interference_noise):
        """Optimized method to update user info for downlink."""
        # Pre-compute common values
        signal_dbm = watts_to_dbm(signal)
        interference_dbm = watts_to_dbm(interference_noise)
        
        # Update user info dictionary
        self.user_info[k]['downlink'].update({
            'sinr_ratio': float(sinr_ratio),
            'sinr_db': float(sinr_db),
            'signal_power_watts': float(signal),
            'signal_power_dbm': float(signal_dbm),
            'interference_noise_watts': float(interference_noise),
            'interference_noise_dbm': float(interference_dbm)
        })
        
        # Update cumulative statistics
        self.user_info[k]['downlink']['cumulative_signal_watts'] += signal
        self.user_info[k]['downlink']['cumulative_interference_watts'] += interference_noise
        
        # Update min/max signal tracking
        if signal < self.user_info[k]['downlink']['min_signal_watts']:
            self.user_info[k]['downlink']['min_signal_watts'] = signal
        if signal > self.user_info[k]['downlink']['max_signal_watts']:
            self.user_info[k]['downlink']['max_signal_watts'] = signal
        
        # Update average signal strength in dBm
        if self._num_step > 0:
            avg_signal_watts = self.user_info[k]['downlink']['cumulative_signal_watts'] / self._num_step
            self.user_info[k]['downlink']['avg_signal_dbm'] = float(watts_to_dbm(avg_signal_watts))
            
            # Update average SINR
            avg_sinr_ratio = self.user_info[k]['downlink']['cumulative_signal_watts'] / self.user_info[k]['downlink']['cumulative_interference_watts']
            self.user_info[k]['downlink']['avg_sinr_ratio'] = float(avg_sinr_ratio)
            self.user_info[k]['downlink']['avg_sinr_db'] = float(watts_to_db(avg_sinr_ratio))
    
    def _cache_sinr_calculations(self):
        """Cache SINR calculations to avoid redundant computations."""
        if not self._sinr_cache_valid:
            # Cache downlink SINR for all users
            self._cached_sinr['downlink'] = np.zeros(self.K)
            for k in range(self.K):
                w_k = self._W[:, k].reshape(-1, 1)
                signal = np.squeeze(
                    np.abs(np.sqrt(self._gains_transmitter_ris_receiver[k]) * 
                           self._channel_matrices["H_RIS_Users"][k] @ 
                           self._Theta_Phi @ 
                           self._channel_matrices["H_BS_RIS"] @ w_k)**2
                )
                interference_noise = Gamma_B_k(
                    k, self._W, self.WWH, self._Theta_Phi, self.Phi_H_Theta_H,
                    self._gains_transmitter_ris_receiver,
                    self._channel_matrices["H_BS_RIS"], self._channel_matrices["H_RIS_Users"],
                    self.kappa[-1], self._channel_matrices["H_Users_RIS"],
                    self.P_users, self.kappa[0], self.SI_coef, self.sigma_k_squared
                )
                self._cached_sinr['downlink'][k] = signal / interference_noise
            
            # Cache uplink SINR for all users
            if self.P_max != 0:
                self._cached_sinr['uplink'] = np.zeros(self.K)
                for k in range(self.K):
                    if self.P_users[k] == 0:
                        self._cached_sinr['uplink'][k] = 0
                        continue
                    f_u_k = self.F[:, k].reshape(-1, 1)
                    signal = np.squeeze(
                        self.P_users[k] * np.abs(np.sqrt(self._gains_transmitter_ris_receiver[k]) * 
                            f_u_k.T @ self._channel_matrices["H_RIS_BS"] @
                            self._Theta_Phi @ self._channel_matrices["H_Users_RIS"][k])**2
                    )
                    interference_noise = Gamma_S_k(
                        self.K, k, self._Theta_Phi, self._gains_transmitter_ris_receiver,
                        self._channel_matrices["H_Users_RIS"],
                        self._channel_matrices["H_RIS_BS"], f_u_k,
                        self.P_users, self.kappa[0], self.delta_k_squared
                    )
                    self._cached_sinr['uplink'][k] = signal / interference_noise
            
            self._sinr_cache_valid = True

    # ?  ----------------------------------------------     COMPUTATION OF THE CHANNELS AND THE SIGNALS     --------------------------------------------------------------
    
    def compute_all_channels(self):
        """Compute time-varying channels for current positions.

        Populates `self._channel_matrices` with:
        - H_RIS_Users, H_BS_Users, H_RIS_Eaves_downlink, H_RIS_Eaves_uplink, H_Users_RIS
        NOTE: BS<->RIS channels are set at reset since BS/RIS are fixed.
        """
        # Check if positions have changed to avoid redundant computations
        positions_changed = (
            self._last_user_positions is None or 
            not np.array_equal(self._last_user_positions, self.users_positions) or
            (self.eavesdropper_active and (
                self._last_eaves_positions is None or 
                not np.array_equal(self._last_eaves_positions, self.eavesdroppers_positions)
            ))
        )
        
        #* managing downlink channels first 
        if not positions_changed and self._cached_gains is not None:
            # Reuse cached gains
            self._gains_transmitter_ris_receiver = self._cached_gains.copy()
        else:
            # Compute gains only when positions have changed
            self._gains_transmitter_ris_receiver[:self.K] = np.array([
                calculate_gain_transmitter_ris_receiver(
                    transmitter_position = self._BS_position,
                    receiver_position = self.users_positions[k],
                    RIS_position = self._RIS_position,
                    RIS_Cells = self.M,
                    lambda_h=self._lambda_h,
                ) for k in range(self.K) ])
            
            A_m = (self._lambda_h/4)**2
            d_t = np.linalg.norm(self._BS_position - self._RIS_position)
            gain_transmitter_to_RIS = ( A_m / (4 * np.pi * (d_t **2) ) )  
            self._gains_transmitter_ris = np.sqrt(gain_transmitter_to_RIS)

            if self.eavesdropper_active:
                self._gains_transmitter_ris_receiver[self.K:] = np.array([
                        calculate_gain_transmitter_ris_receiver(
                            transmitter_position = self._BS_position,
                            receiver_position = self.eavesdroppers_positions[eavesdropper],
                            RIS_position = self._RIS_position,
                        RIS_Cells = self.M,
                        lambda_h=self._lambda_h,) for eavesdropper in range(self._num_eavesdroppers) ])
            
            # Cache the gains and positions for next time
            self._cached_gains = self._gains_transmitter_ris_receiver.copy()
            self._last_user_positions = self.users_positions.copy()
            if self.eavesdropper_active:
                self._last_eaves_positions = self.eavesdroppers_positions.copy()

        if self.debugging:
            # Compute RIS -> Users channels using list comprehension
            self._channel_matrices["H_RIS_Test_Point"] = np.array(
                rician_fading_channel(
                    transmitter_position=self._RIS_position,
                    receiver_position=self.test_point_for_user,
                    W_h_t=self._M, W_h_r=1,
                    d_h_tx=self._d_h[1], d_h_rx=self._d_h[2],
                    lambda_h=self._lambda_h,
                    epsilon_h=self.rician_factor,
                    numpy_generator=self.numpy_rng, bjornson = self.bjornson,
                ))  # Shape: (1, M)

        # Compute BS -> Users channels using list comprehension
        self._channel_matrices["H_BS_Users"] = np.array([
            rician_fading_channel(
                transmitter_position=self._BS_position,
                receiver_position=self.users_positions[k],
                W_h_t=self.N_t, W_h_r=1,
                d_h_tx=self._d_h[1], d_h_rx=self._d_h[2],
                lambda_h=self._lambda_h,
                epsilon_h=self.rician_factor,
                numpy_generator=self.numpy_rng,bjornson = self.bjornson,los_only = self.los_only,
            ) for k in range(self.K)
        ])  # Shape: (K, 1, M)

        # Compute RIS -> Users channels using list comprehension
        self._channel_matrices["H_RIS_Users"] = np.array([
            rician_fading_channel(
                transmitter_position=self._RIS_position,
                receiver_position=self.users_positions[k],
                W_h_t=self._M, W_h_r=1,
                d_h_tx=self._d_h[1], d_h_rx=self._d_h[2],
                lambda_h=self._lambda_h,
                epsilon_h=self.rician_factor,
                numpy_generator=self.numpy_rng, bjornson = self.bjornson,los_only = self.los_only,
            ) for k in range(self.K)
        ])  # Shape: (K, 1, M)

        # Compute RIS -> Eavesdroppers channels for the downlink
        self._channel_matrices["H_RIS_Eaves_downlink"] = np.array([
            rician_fading_channel(
                transmitter_position=self._RIS_position,
                receiver_position=self.eavesdroppers_positions[l],
                W_h_t=self._M, W_h_r=1,
                d_h_tx=self._d_h[1], d_h_rx=self._d_h[3],
                lambda_h=self._lambda_h,
                epsilon_h=self.rician_factor,
                numpy_generator=self.numpy_rng, bjornson = self.bjornson,los_only = self.los_only,
            ) for l in range(self._num_eavesdroppers)
        ])  # Shape: (L, 1, M)

        #* managing uplink channels 
        # Compute Users -> RIS channels
        self._channel_matrices["H_Users_RIS"] = np.array([
            rician_fading_channel(
                transmitter_position=self.users_positions[k],
                receiver_position=self._RIS_position,
                W_h_t=1, W_h_r=self._M,
                d_h_tx=self._d_h[2], d_h_rx=self._d_h[1],
                lambda_h=self._lambda_h,
                epsilon_h=self.rician_factor,
                numpy_generator=self.numpy_rng, bjornson = self.bjornson,los_only = self.los_only,
            ) for k in range(self.K)
        ])  # Shape: (K, M, 1)

        # Compute RIS -> Eavesdroppers channels for the uplink
        self._channel_matrices["H_RIS_Eaves_uplink"] = np.array([
            rician_fading_channel(
                transmitter_position=self._RIS_position,
                receiver_position=self.eavesdroppers_positions[l],
                W_h_t=self._M, W_h_r=1, 
                d_h_tx=self._d_h[1], d_h_rx=self._d_h[3], 
                lambda_h=self._lambda_h,
                epsilon_h=self.rician_factor,
                numpy_generator=self.numpy_rng, bjornson = self.bjornson,los_only = self.los_only,
            ) for l in range(self._num_eavesdroppers)
        ])  # Shape: (L, 1, M)


    def compute_eavesdropper_channels(self):
        """Compute RIS->eaves channels for both uplink and downlink."""
        # Compute RIS -> Eavesdroppers channels for the donwlink
        H_RIS_Eaves_downlink = []
        for l in range(self._num_eavesdroppers):
            H_RIS_Eaves_downlink.append( rician_fading_channel ( transmitter_position = self._RIS_position,
                                                                receiver_position = self.eavesdroppers_positions[l],
                                                                W_h_t = self._M, W_h_r = 1,
                                                                d_h_tx = self._d_h[1], d_h_rx = self._d_h[3],
                                                                lambda_h = self._lambda_h,
                                                                epsilon_h = self.rician_factor,bjornson = self.bjornson,los_only = self.los_only,
                                                                numpy_generator = self.numpy_rng) ) 
        self._channel_matrices["H_RIS_Eaves_downlink"] = np.array(H_RIS_Eaves_downlink)  # Shape: (L, 1, M)
        
        # Compute RIS -> Eavesdroppers channels for the uplink
        H_RIS_Eaves_uplink = []
        for l in range(self._num_eavesdroppers):
            H_RIS_Eaves_uplink.append( rician_fading_channel ( transmitter_position = self._RIS_position,
                                                              receiver_position = self.eavesdroppers_positions[l],
                                                              W_h_t = self._M, W_h_r = 1, d_h_tx = self._d_h[1],
                                                              d_h_rx = self._d_h[3],
                                                              lambda_h =  self._lambda_h,
                                                              epsilon_h = self.rician_factor,bjornson = self.bjornson, los_only = self.los_only,
                                                              numpy_generator = self.numpy_rng ) ) 
        self._channel_matrices["H_RIS_Eaves_uplink"] = np.array(H_RIS_Eaves_uplink)  # Shape: (L, 1, M)

        pass


    def compute_decoding_matrix(self):
        """Compute the uplink combining matrix `F` for the current step."""
        H_u = self._channel_matrices["H_RIS_BS"]
        for i in range(self.K):
            combining_vector = H_u  @ self._Theta @self._Phi @ self._channel_matrices["H_Users_RIS"][i]
            self.F[:,i] = np.squeeze(combining_vector)
        pass
    
    # ?  ----------------------------------------------     ACTION MANAGEMENT/PROCESSING FUNCTIONS    ----------------------------------------------

    def process_raw_actions(self, actor_actions):
        """Convert raw actor output tensor into `Theta` and `W`.

        Args:
            actor_actions (array): Raw output of the model for actions to take.
        """
        current_actions = actor_actions.numpy()
        self.current_actions = current_actions

        #* updataing Theta
        normalized_theta_real = current_actions[2 * self.N_t * self.K::2] 
        normalized_theta_imag = current_actions[2 * self.N_t * self.K + 1::2] 
        self._Theta = np.diag(normalized_theta_real + 1j * normalized_theta_imag)
        #self._Theta = np.eye(self._M, self._M, dtype=complex)

        #* updating W
        
        W_flattened_real = current_actions[:2 * self.N_t * self.K:2]
        W_flattened_imag = current_actions[1:2 * self.N_t * self.K:2]
        temporary_W = W_flattened_real + 1j * W_flattened_imag

        # Directly reshape using column-major ordering
        self._W = temporary_W.reshape(self.K, self.N_t).T
        pass


    # ?  ----------------------------------------------     STATE MANAGEMENT FUNCTIONS    ------------------------------------------------------------------------------------------------------


    def compute_previous_rate_part(self):
        """
        Compute the rate part at the (t-1)-th time slot. It is composed of:
        1. The sum rate at the legitimate users
        2. The sum rate at the BS
        3. The sum rate at the eavesdroppers ( = 0 here)
        4. The SSR.

        These terms are stored inside the array `self.previous_rate_part`.
        This function is used to create the state.
        """
        if self._num_step > 0:
            # Vectorized computation for legitimate user rates and BS rates
            SINR_B_k_values = np.array([self.sinr_downlink_users[i] for i in range(self.K)])
            SINR_S_k_values = np.array([self.sinr_uplink_users[i] for i in range(self.K)])
            
            rates_legitimate_users = np.log2(1 + SINR_B_k_values)  # Rates at legitimate users
            rates_BS = np.log2(1 + SINR_S_k_values)  # Rates at the BS

            # Compute rates for eavesdroppers and SSR
            SSR_terms = np.zeros(self.K)
            if self.eavesdropper_active:
                eavesdroppers_rates = np.zeros((self.K, self._num_eavesdroppers))
                sinr_matrix = np.array([
                    [self.SINR_E_d_k_l(i, l) for l in range(self._num_eavesdroppers)]
                    for i in range(self.K)
                ])
                eavesdroppers_rates = np.log2(1 + sinr_matrix) + rates_BS[:, np.newaxis]
                SSR_terms = rates_legitimate_users + rates_BS - np.max(eavesdroppers_rates, axis=1)
                sum_eavesdroppers_rates = np.sum(eavesdroppers_rates)
            else:
                sum_eavesdroppers_rates = 0
                SSR_terms = rates_legitimate_users + rates_BS

            #* Trying something with additional (optional) observations infos
            # ! only here for testing purpose
            self.previous_rate_part[2] = sum_eavesdroppers_rates  # Sum rate at the eavesdroppers
            self.previous_rate_part[3] = np.sum(np.maximum(0, SSR_terms))  # SSR
            self.previous_fairness = self.current_fairness
            if not self.additional_state_info:
                self.previous_rate_part[0] = np.sum(rates_legitimate_users)
                self.previous_rate_part[1] = np.sum(rates_BS)
                self.previous_rate_part[2] = sum_eavesdroppers_rates  # Sum rate at the eavesdroppers
                self.previous_rate_part[3] = np.sum(np.maximum(0, SSR_terms))  # SSR
            else:
                self.previous_rate_part[:self.K] =  rates_legitimate_users 
                self.previous_rate_part[self.K:2 *self.K] = rates_BS
                self.previous_rate_part[2 * self.K] = self.previous_fairness
                self.previous_rate_part[2 * self.K + 1] = 0  # Sum rate at the eavesdroppers
                self.previous_rate_part[2 * self.K + 2] = np.sum(np.maximum(0, SSR_terms))  # SSR
                self.previous_rate_part[2 * self.K + 3:] = np.array(self.users_positions).flatten()

        pass


    def get_state(self):
        """Function that generates the state for the environment at the t-th time slot.
        It is composed of 6 parts: \\
        1°/ the rate part at the (t-1)-th time slot (see method compute_rate_part) \\
        2°/ the cascaded channel part at the t-th time slot. \\
        3°/ the phase noise part at the t-th time slot. \\
        4°/ the action at the (t-1)-th time slot. \\
        5°/ the transmit power of the BS. \\
        6°/ the received power of the legitimate user at the (t-1)-th time slot.

        The state is an attribut array of dimensions self.state_dim.
        """
        # Cache matrix products for efficiency
        self._cache_matrix_products()
        
        state = np.zeros(self.state_dim)
        #* starting the state by the rate  part at the (t-1)-th time slot
        state[0:self.previous_rate_dim] = self.previous_rate_part
        #* continuing the state with the cascaded channel part
        start_index, end_index = self.previous_rate_dim, self.previous_rate_dim + self.K *self.K

        # beginning with the BS-RIS-legitimate users channel G_1d - USE CACHED
        G_1D = self._cached_matrix_products["G_1D"]
        BS_RIS_LU_FLAT = G_1D.flatten()
        state[start_index:end_index] = np.real(BS_RIS_LU_FLAT) # correctly managing indexing inside the state array
        start_index, end_index = end_index, end_index + self.K *self.K
        state[start_index:end_index]  = np.imag(BS_RIS_LU_FLAT)

        # Continuing with the BS-RIS-Eavesdropper channel G_2D - USE CACHED
        if self.eavesdropper_active:
            BS_RIS_EAVES = self._cached_matrix_products["BS_RIS_EAVES"]
            BS_RIS_EAVES_FLAT =  BS_RIS_EAVES.flatten()
            start_index, end_index = end_index, end_index + self.K * self._num_eavesdroppers
            state[start_index:end_index] = np.real(BS_RIS_EAVES_FLAT)
            start_index, end_index = end_index, end_index + self.K * self._num_eavesdroppers
            state[start_index:end_index] = np.imag(BS_RIS_EAVES_FLAT)
        else:
            start_index, end_index = end_index, end_index + 1
            state[start_index:end_index] = 0
            start_index, end_index = end_index, end_index + 1 
            state[start_index:end_index] = 0
        
        # Continuing with the Legitimate Users-RIS-BS channel G_1u - USE CACHED
        LU_BS_RIS = self._cached_matrix_products["LU_BS_RIS"]  # Shape: (N_r, K)
        LU_BS_RIS_FLAT = LU_BS_RIS.flatten()
        start_index, end_index = end_index, end_index + self.N_r *self.K
        state[start_index:end_index] = np.real(LU_BS_RIS_FLAT) # correctly managing indexing inside the state array
        start_index, end_index = end_index, end_index + self.N_r *self.K
        state[start_index:end_index]  = np.imag(LU_BS_RIS_FLAT)

        # Continuing with the Legitimate Users-RIS-Eavesdroppers channel G_2u - USE CACHED
        if self.eavesdropper_active :
            LU_BS_EAVES = self._cached_matrix_products["LU_BS_EAVES"]  # Shape: (L, K)
            LU_BS_EAVES_FLAT = LU_BS_EAVES.flatten()
            start_index, end_index = end_index, end_index + self._num_eavesdroppers *self.K
            state[start_index:end_index] = np.real(LU_BS_EAVES_FLAT) # correctly managing indexing inside the state array
            start_index, end_index = end_index, end_index + self._num_eavesdroppers *self.K
            state[start_index:end_index]  = np.imag(LU_BS_EAVES_FLAT)
        else:
            start_index, end_index = end_index, end_index + 1
            state[start_index:end_index] = 0
            start_index, end_index = end_index, end_index+1 
            state[start_index:end_index] = 0

        #* managing the phase noise part
        start_index, end_index = end_index, end_index + self._M
        state[start_index:end_index]  = self.phases

        #* managing the action at the (t-1)-th time slot
        #TODO write the code once the action choosing system is done 
        start_index, end_index = end_index, end_index + 2 * self._M + 2 * self.N_t *self.K
        state[start_index:end_index]  = self.previous_actions

        #* managing the transmit power of the BS - OPTIMIZED
        # Pre-compute W^H @ W more efficiently
        W_norms = np.sum(np.abs(self._W)**2, axis=0)  # Shape: (K,)
        w_results = np.zeros(2 * self.K)
        w_results[0::2] = W_norms  # Real part norms
        w_results[1::2] = 0.0      # Imaginary part norms (W is real in this context)
        new_end_index = end_index + 2 * self.K
        state[end_index:new_end_index] = w_results
        end_index = new_end_index

        #* managing the received power of the legitimate user at the (t-1)-th time slot - OPTIMIZED
        # Compute norms more efficiently
        G_norms = np.sum(np.abs(self.previous_G_1D)**2, axis=0)  # Shape: (K,)
        g_results = np.zeros(2 * self.K)
        g_results[0::2] = G_norms  # Real part norms
        g_results[1::2] = 0.0      # Imaginary part norms (already included in abs)

        # Assign to state array
        new_end_index = end_index + 2 * self.K
        state[end_index:new_end_index] = g_results
        end_index = new_end_index
        
        self.previous_G_1D = deepcopy(G_1D) #* Updating the value of previous G_1D

        #* adding previous fairness
        #state[-1] = self.previous_fairness
        return state


    # ?  ----------------------------------------------     STEP FUNCTIONS    ----------------------------------------------------------------------------------------------------

    def step(self, state, new_actions):
        self._num_step += 1
        self.state = state
        self.compute_previous_rate_part()
        #* Transition the environment by updating all terms impacted by the change of Theta and W
        self.transitioning(new_actions)
        self.compute_new_SINRs()
        self.compute_reward()
        self.previous_actions = new_actions
        self.next_state = self.get_state()

        if self._moving_eavesdroppers: #* managing if a moving eavesdropper model is currently being used
            self.move_eavesdroppers(grid_mobility)
        # PPO difference: keep API the same but clarify that episodes do not signal done here; trainers manage fixed rollout lengths.
        return self.state, self.current_actions, self.current_reward, self.next_state


    def transitioning(self,new_actions):
        """Function to operate the transition of the environment between timestep t and t+1. This function must be called after an action is taken at time t before changing the environment accordingly.
        """
        self.process_raw_actions(new_actions)
        self.compute_theta_phi()
        self.compute_WWH()
        self.compute_decoding_matrix()
        pass


    # ?  ----------------------------------------------     SINR FUNCTIONS    ------------------------------------------------------------------------------------------------------


    def compute_new_SINRs(self):
        """Recompute all SINR metrics for the current timestep.

        This updates:
        - Downlink SINR for all legitimate users
        - Uplink SINR for all legitimate users
        - Eavesdropper downlink/uplink SINR matrices when eavesdroppers are active
        """
        # Use cached SINR calculations for better performance
        self._cache_sinr_calculations()
        
        # Update arrays with cached values
        self.sinr_downlink_users = self._cached_sinr['downlink'].copy()
        if self.P_max != 0:
            self.sinr_uplink_users = self._cached_sinr['uplink'].copy()
        
        # Still need to compute eavesdropper SINRs as they're not cached yet
        if self.eavesdropper_active:
            self._SINR_eavesdropper_downlink_all()
            self._SINR_eavesdropper_uplink_all()
        
        # Invalidate cache for next step
        self._sinr_cache_valid = False
        pass


    def _SINR_downlink_all_users(self):
        """
        Compute SINR for all users and log statistics.

        Returns:
            dict: SINR ratios per user
        """
        num_users = self._W.shape[1]
        self.sinr_downlink_users = np.zeros(self.num_users)
        self.sinr_downlink_details = {}

        # Precompute useful signals
        """signal_at_BS = np.trace(np.diag(np.diag(self._W @ self._W.conj().T)).real)

        total_signal_before_ris= np.sum(np.abs(self._gains_transmitter_ris * self._channel_matrices["H_BS_RIS"] @ self._W) ** 2)

        signal_just_after_ris = np.sum(np.abs(self._gains_transmitter_ris * self._Theta @ self._Phi  @ self._channel_matrices["H_BS_RIS"] @ self._W) ** 2)"""

        for k in range(num_users):
            w_k = self._W[:, k].reshape(-1, 1)
        
            signal = np.squeeze(
            np.abs( np.sqrt(self._gains_transmitter_ris_receiver[k]) * self._channel_matrices["H_RIS_Users"][k] @ self._Theta @ self._Phi @ self._channel_matrices["H_BS_RIS"] @ w_k) ** 2
            )

            # Interference + noise
            interference_noise = Gamma_B_k(
            k, self._W, self.WWH, self._Theta_Phi, self.Phi_H_Theta_H,
            self._gains_transmitter_ris_receiver,
            self._channel_matrices["H_BS_RIS"], self._channel_matrices["H_RIS_Users"],
            self.kappa[-1], self._channel_matrices["H_Users_RIS"],
            self.P_users, self.kappa[0], self.SI_coef, self.sigma_k_squared
            )
    
            # Compute SINR
            sinr_ratio = signal / interference_noise
            sinr_db = watts_to_db(sinr_ratio)
            
            # Optimized verbose logging - only compute when needed
            if self.verbose:
                self._update_user_info_downlink(k, sinr_ratio, sinr_db, signal, interference_noise)

                # Legacy arrays for backward compatibility
                self.downlink_signal_strength[k] += signal
                self.min_downlink_signal_strength[k] = min(signal, self.min_downlink_signal_strength[k])

                if signal > self.max_downlink_signal_strength[k]:
                    self.max_downlink_interf_k[k] = interference_noise
                    self.max_downlink_signal_strength[k] = signal

                avg_signal_dbm = watts_to_dbm(self.downlink_signal_strength[k] / self._num_step)

            self.sinr_downlink_users[k] = sinr_ratio
            self.sinr_downlink_signals[k] += signal
            self.sinr_downlink_interfs[k] += interference_noise
            self.downlink_sinr_average[k] += sinr_ratio

            if self.verbose:
                sinr_average_ratio = self.downlink_sinr_average[k]/self._num_step

                self.sinr_downlink_details[k] = {
                    "SINR_in_db": float(sinr_db),
                    "Average_SINR_in_db": float(watts_to_db(sinr_average_ratio)),
                    "Average_SINR_ratio": float(sinr_average_ratio),
                    "ratio": float(sinr_ratio),
                    "direct_signal": float(signal),
                    "interference_noise": float(interference_noise),
                    "Average interference_noise": float(watts_to_dbm(self.sinr_downlink_interfs[k]/self._num_step)),
                    "min_signal_dbm": float(watts_to_dbm(self.min_downlink_signal_strength[k])),
                    "max_signal_dbm": float(watts_to_dbm(self.max_downlink_signal_strength[k])),
                    "avg_signal_dbm": float(avg_signal_dbm)
                }

                # Track maximum SINR globally
                if sinr_db > self.max_sinr_b_k.get("SINR_in_db", -np.inf):
                    self.max_sinr_b_k = self.sinr_downlink_details[k]

            pass
        


    def _SINR_uplink_all_users(self):
        """
        Compute the Signal-to-Interference-plus-Noise Ratio at the Base Station
        for all users (uplink).

        Returns:
            np.ndarray: SINR ratios per user
        """
        
        self.sinr_uplink_users = np.zeros(self.K)
        self.sinr_s_details = {}
        for k in range(self.K):
            f_u_k = self.F[:, k].reshape(-1, 1)
            # If user has no transmit power, SINR is zero
            if self.P_users[k] == 0:
                self.sinr_uplink_users[k] = 0
                
                # Update user information dictionary for zero power case
                self.user_info[k]['uplink'].update({
                    'sinr_ratio': 0.0,
                    'sinr_db': -np.inf,
                    'signal_power_watts': 0.0,
                    'signal_power_dbm': -np.inf,
                    'interference_noise_watts': 0.0,
                    'interference_noise_dbm': -np.inf
                })
                
                if self.verbose:
                    self.sinr_s_details[k] = {
                        "SINR_in_db": -np.inf,
                        "ratio": 0.0,
                        "direct_signal": 0.0,
                        "interference_noise": 0.0
                    }
                continue

            # Compute direct signal power at the BS for user k
            signal = np.squeeze(
                self.P_users[k] *
                np.abs(np.sqrt(self._gains_transmitter_ris_receiver[k]) * 
                    f_u_k.T @
                    self._channel_matrices["H_RIS_BS"] @
                    self._Theta_Phi @
                    self._channel_matrices["H_Users_RIS"][k]
                ) ** 2
            )
            # Compute interference + noise power for user k
            interference_noise = Gamma_S_k(
                self.K, k, self._Theta_Phi,self._gains_transmitter_ris_receiver,
                self._channel_matrices["H_Users_RIS"],
                self._channel_matrices["H_RIS_BS"], f_u_k,
                self.P_users, self.kappa[0],
                self.delta_k_squared
            )
            # Compute SINR
            sinr_ratio = signal / interference_noise
            sinr_db = watts_to_db(sinr_ratio)
            
            if self.verbose:
                # Update comprehensive user information dictionary
                self.user_info[k]['uplink'].update({
                    'sinr_ratio': float(sinr_ratio),
                    'sinr_db': float(sinr_db),
                    'signal_power_watts': float(signal),
                    'signal_power_dbm': float(watts_to_dbm(signal)),
                    'interference_noise_watts': float(interference_noise),
                    'interference_noise_dbm': float(watts_to_dbm(interference_noise))
                })
                # Update cumulative statistics
                self.user_info[k]['uplink']['cumulative_signal_watts'] += signal
                self.user_info[k]['uplink']['cumulative_interference_watts'] += interference_noise
                # Update min/max signal tracking
                if signal < self.user_info[k]['uplink']['min_signal_watts']:
                    self.user_info[k]['uplink']['min_signal_watts'] = signal
                if signal > self.user_info[k]['uplink']['max_signal_watts']:
                    self.user_info[k]['uplink']['max_signal_watts'] = signal
                # Update average signal strength in dBm
                if self._num_step > 0:
                    avg_signal_watts = self.user_info[k]['uplink']['cumulative_signal_watts'] / self._num_step
                    self.user_info[k]['uplink']['avg_signal_dbm'] = float(watts_to_dbm(avg_signal_watts))
                    # Update average SINR
                    avg_sinr_ratio = self.user_info[k]['uplink']['cumulative_signal_watts'] / self.user_info[k]['uplink']['cumulative_interference_watts']
                    self.user_info[k]['uplink']['avg_sinr_ratio'] = float(avg_sinr_ratio)
                    self.user_info[k]['uplink']['avg_sinr_db'] = float(watts_to_db(avg_sinr_ratio))

            # Legacy arrays for backward compatibility
            self.sinr_uplink_users[k] = sinr_ratio
            self.uplink_signal_strength[k] += signal

            # Store details if verbose mode is enabled
            if self.verbose:
                self.sinr_s_details[k] = {
                    "SINR_in_db": float(sinr_db),
                    "ratio": float(sinr_ratio),
                    "direct_signal": float(signal),
                    "interference_noise": float(interference_noise)
                }
                # Track the maximum SINR seen at the BS
                if sinr_db > self.max_sinr_s_k.get("SINR_in_db", -np.inf):
                    self.max_sinr_s_k = self.sinr_s_details[k]
        pass        



    def SINR_E_d_k_l(self, k: int, l: int):
        """Return precomputed downlink SINR observed at eavesdropper l for user k.

        Args:
            k (int): Index of the legitimate user of interest.
            l (int): Index of the eavesdropper.

        Returns:
            float: SINR ratio at eavesdropper l on the downlink signal intended for user k.
        """
        # Ensure values are computed for current step
        if not hasattr(self, "sinr_eavesdropper_downlink") or self.sinr_eavesdropper_downlink.shape != (self.K, self._num_eavesdroppers):
            self._SINR_eavesdropper_downlink_all()
        return self.sinr_eavesdropper_downlink[k, l]
    

    def SINR_E_u_k_l(self, k: int, l: int):
        """Return precomputed uplink SINR observed at eavesdropper l for user k.

        Args:
            k (int): Index of the legitimate user of interest.
            l (int): Index of the eavesdropper.

        Returns:
            float: SINR ratio at eavesdropper l on the uplink signal from user k.
        """
        if not hasattr(self, "sinr_eavesdropper_uplink") or self.sinr_eavesdropper_uplink.shape != (self.K, self._num_eavesdroppers):
            self._SINR_eavesdropper_uplink_all()
        return self.sinr_eavesdropper_uplink[k, l]



    def _SINR_eavesdropper_downlink_all(self):
        """Compute downlink SINR at all eavesdroppers for every user.

        Results are stored in `self.sinr_eavesdropper_downlink` with shape (K, L),
        where K is the number of users and L is the number of eavesdroppers.
        """
        if not self.eavesdropper_active or self._num_eavesdroppers == 0:
            self.sinr_eavesdropper_downlink = np.zeros((self.K, 0))
            return
        self.sinr_eavesdropper_downlink = np.zeros((self.K, self._num_eavesdroppers))
        for k in range(self.K):
            w_k = self._W[:, k].reshape(-1, 1)
            for l in range(self._num_eavesdroppers):
                # Direct signal power received at eavesdropper l for user k
                signal_power_at_eaves = np.squeeze(
                    np.abs(
                        np.sqrt(self._gains_transmitter_ris_receiver[self.K + l])
                        * self._channel_matrices["H_RIS_Eaves_downlink"][l]
                        @ self._Theta_Phi
                        @ self._channel_matrices["H_BS_RIS"]
                        @ w_k
                    )
                    ** 2
                )
                # Interference plus noise at eavesdropper l for user k
                interference_plus_noise = Gamma_E_d_k_l(
                    k,
                    l,
                    self._W,
                    self._Theta_Phi,
                    self.Phi_H_Theta_H,
                    self.diag_matrix_WWH,
                    self._gains_transmitter_ris_receiver,
                    self._channel_matrices["H_BS_RIS"],
                    self._channel_matrices["H_RIS_Eaves_downlink"],
                    self.kappa[-1],
                    self._channel_matrices["H_Users_RIS"],
                    self._channel_matrices["H_RIS_Eaves_uplink"],
                    self.P_users,
                    self.kappa[0],
                    self.mu_d_l_squared,
                )
                
                # Compute SINR
                sinr_ratio = signal_power_at_eaves / interference_plus_noise
                sinr_db = watts_to_db(sinr_ratio)

                if self.verbose:
                    # Update comprehensive eavesdropper information dictionary
                    self.eavesdropper_info[l]['downlink']['sinr_ratios'][k] = sinr_ratio
                    self.eavesdropper_info[l]['downlink']['signal_powers_watts'][k] = signal_power_at_eaves
                    self.eavesdropper_info[l]['downlink']['signal_powers_dbm'][k] = watts_to_dbm(signal_power_at_eaves)
                    self.eavesdropper_info[l]['downlink']['interference_noise_watts'][k] = interference_plus_noise
                    self.eavesdropper_info[l]['downlink']['interference_noise_dbm'][k] = watts_to_dbm(interference_plus_noise)
                    
                    # Update cumulative statistics
                    self.eavesdropper_info[l]['downlink']['cumulative_signal_watts'][k] += signal_power_at_eaves
                    self.eavesdropper_info[l]['downlink']['cumulative_interference_watts'][k] += interference_plus_noise
                    
                    # Update min/max signal tracking
                    if signal_power_at_eaves < self.eavesdropper_info[l]['downlink']['min_signal_watts'][k]:
                        self.eavesdropper_info[l]['downlink']['min_signal_watts'][k] = signal_power_at_eaves
                    if signal_power_at_eaves > self.eavesdropper_info[l]['downlink']['max_signal_watts'][k]:
                        self.eavesdropper_info[l]['downlink']['max_signal_watts'][k] = signal_power_at_eaves
                    
                    # Update average signal strength in dBm
                    if self._num_step > 0:
                        avg_signal_watts = self.eavesdropper_info[l]['downlink']['cumulative_signal_watts'][k] / self._num_step
                        self.eavesdropper_info[l]['downlink']['avg_signal_dbm'][k] = watts_to_dbm(avg_signal_watts)
                        
                        # Update average SINR
                        avg_sinr_ratio = self.eavesdropper_info[l]['downlink']['cumulative_signal_watts'][k] / self.eavesdropper_info[l]['downlink']['cumulative_interference_watts'][k]
                        self.eavesdropper_info[l]['downlink']['avg_sinr_ratios'][k] = avg_sinr_ratio
                        self.eavesdropper_info[l]['downlink']['avg_sinr_db'][k] = watts_to_db(avg_sinr_ratio)
                
                # Legacy array for backward compatibility
                self.sinr_eavesdropper_downlink[k, l] = sinr_ratio



    def _SINR_eavesdropper_uplink_all(self):
        """Compute uplink SINR at all eavesdroppers for every user.

        Results are stored in `self.sinr_eavesdropper_uplink` with shape (K, L),
        where K is the number of users and L is the number of eavesdroppers.
        """
        if not self.eavesdropper_active or self._num_eavesdroppers == 0:
            self.sinr_eavesdropper_uplink = np.zeros((self.K, 0))
            return
        self.sinr_eavesdropper_uplink = np.zeros((self.K, self._num_eavesdroppers))
        for k in range(self.K):
            if self.P_users[k] == 0:
                # Update eavesdropper information for zero power case
                for l in range(self._num_eavesdroppers):
                    self.eavesdropper_info[l]['uplink']['sinr_ratios'][k] = 0.0
                    self.eavesdropper_info[l]['uplink']['signal_powers_watts'][k] = 0.0
                    self.eavesdropper_info[l]['uplink']['signal_powers_dbm'][k] = -np.inf
                    self.eavesdropper_info[l]['uplink']['interference_noise_watts'][k] = 0.0
                    self.eavesdropper_info[l]['uplink']['interference_noise_dbm'][k] = -np.inf
                continue
            for l in range(self._num_eavesdroppers):
                # Direct signal power received at eavesdropper l from user k
                signal_power_at_eaves = np.squeeze(
                    np.abs(
                        np.sqrt(self._gains_transmitter_ris_receiver[self.K + l])
                        * self._channel_matrices["H_RIS_Eaves_uplink"][l]
                        @ self._Theta_Phi
                        @ self._channel_matrices["H_Users_RIS"][k]
                        * np.sqrt(self.P_users[k])
                    )
                    ** 2
                )
                # Interference plus noise at eavesdropper l for user k
                interference_plus_noise = Gamma_E_u_k_l(
                    k,
                    l,
                    self._W,
                    self._Theta_Phi,
                    self.Phi_H_Theta_H,
                    self.diag_matrix_WWH,
                    self._gains_transmitter_ris_receiver,
                    self._channel_matrices["H_BS_RIS"],
                    self._channel_matrices["H_RIS_Eaves_downlink"],
                    self.kappa[-1],
                    self._channel_matrices["H_Users_RIS"],
                    self._channel_matrices["H_RIS_Eaves_uplink"],
                    self.P_users,
                    self.kappa[0],
                )
                self.sinr_eavesdropper_uplink[k, l] = signal_power_at_eaves / interference_plus_noise

                # Compute SINR
                sinr_ratio = signal_power_at_eaves / interference_plus_noise
                sinr_db = watts_to_db(sinr_ratio)
                
                # Update comprehensive eavesdropper information dictionary
                self.eavesdropper_info[l]['uplink']['sinr_ratios'][k] = sinr_ratio
                self.eavesdropper_info[l]['uplink']['signal_powers_watts'][k] = signal_power_at_eaves
                self.eavesdropper_info[l]['uplink']['signal_powers_dbm'][k] = watts_to_dbm(signal_power_at_eaves)
                self.eavesdropper_info[l]['uplink']['interference_noise_watts'][k] = interference_plus_noise
                self.eavesdropper_info[l]['uplink']['interference_noise_dbm'][k] = watts_to_dbm(interference_plus_noise)
                
                # Update cumulative statistics
                self.eavesdropper_info[l]['uplink']['cumulative_signal_watts'][k] += signal_power_at_eaves
                self.eavesdropper_info[l]['uplink']['cumulative_interference_watts'][k] += interference_plus_noise
                
                # Update min/max signal tracking
                if signal_power_at_eaves < self.eavesdropper_info[l]['uplink']['min_signal_watts'][k]:
                    self.eavesdropper_info[l]['uplink']['min_signal_watts'][k] = signal_power_at_eaves
                if signal_power_at_eaves > self.eavesdropper_info[l]['uplink']['max_signal_watts'][k]:
                    self.eavesdropper_info[l]['uplink']['max_signal_watts'][k] = signal_power_at_eaves
                
                # Update average signal strength in dBm
                if self._num_step > 0:
                    avg_signal_watts = self.eavesdropper_info[l]['uplink']['cumulative_signal_watts'][k] / self._num_step
                    self.eavesdropper_info[l]['uplink']['avg_signal_dbm'][k] = watts_to_dbm(avg_signal_watts)
                    
                    # Update average SINR
                    avg_sinr_ratio = self.eavesdropper_info[l]['uplink']['cumulative_signal_watts'][k] / self.eavesdropper_info[l]['uplink']['cumulative_interference_watts'][k]
                    self.eavesdropper_info[l]['uplink']['avg_sinr_ratios'][k] = avg_sinr_ratio
                    self.eavesdropper_info[l]['uplink']['avg_sinr_db'][k] = watts_to_db(avg_sinr_ratio)
                
                # Legacy array for backward compatibility
                self.sinr_eavesdropper_uplink[k, l] = sinr_ratio




    # ?  ----------------------------------------------     REWARD FUNCTIONS    ------------------------------------------------------------------------------------------------------
    


    def eavesdropper_reward(self):
        """
        Computes the total eavesdropper reward, which is the sum of the maximum
        eavesdropper rates across all users.

        Returns:
            float: The sum of the maximum eavesdropper rates across all k users.
        """
        if not self.eavesdropper_active:
            return 0
        
        eavesdroppers_reward = 0

        self.all_eavesdroppers_rewards = np.zeros(self.K)
        self.detailed_eavesdroppers_rewards = np.zeros(self.K, dtype=object)
        # Iterate over all users k
        for k in range(self.K):  # Assuming self.K represents the total number of users
            # Calculate R_E_k_all_l for user k
            R_E_k_all_l = np.array([np.log2(1 + self.SINR_E_d_k_l(k, l)) + np.log2(1 + self.SINR_E_u_k_l(k,l)) for l in range(self._num_eavesdroppers)])

            # Add the maximum value for this user to the total reward
            indice_max = np.argmax(R_E_k_all_l)
            max_eave_reward_for_user_k = R_E_k_all_l[indice_max] 
            eavesdroppers_reward +=  max_eave_reward_for_user_k

            if self.verbose:
                self.all_eavesdroppers_rewards[k] =  deepcopy(max_eave_reward_for_user_k)
                self.detailed_eavesdroppers_rewards[k] = {"Total interception":np.float16(max_eave_reward_for_user_k),
                "Downlink interception":np.float16(np.log2(1 + self.SINR_E_d_k_l(k, indice_max))),
                "Uplink interception": np.float16(np.log2(1 + self.SINR_E_u_k_l(k,indice_max))) }

        return eavesdroppers_reward
    


    def compute_reward(self):
        """
        Computes the rewards for both decisive and informative reward functions.

        This method calculates the rewards for each user based on the defined reward functions.
        It handles both decisive and informative rewards, aggregating the results accordingly.

        Attributes:
            self.users_current_rewards (np.ndarray): An array to store the current rewards for each user.
            self.current_reward (float): A variable to store the total current reward.
            self.informative_current_rewards (dict): A dictionary to store the current informative rewards.
            self.decisive_current_rewards (dict): A dictionary to store the current decisive rewards.

        Helper Function:
            apply_reward_function(reward_name, reward_function, reward_type):
                Applies a given reward function and updates the corresponding rewards dictionary.

                Args:
                    reward_name (str): The name of the reward function.
                    reward_function (callable): The reward function to be applied.
                    reward_type (str): The type of reward ('decisive' or 'informative').

                Returns:
                    tuple: A tuple containing the reward per user and the total reward.

        Process:
            1. Initializes `self.users_current_rewards` and `self.current_reward` to zero.
            2. Defines the `apply_reward_function` helper function to apply reward functions and update rewards dictionaries.
            3. Applies each decisive reward function using `apply_reward_function` and aggregates the results.
            4. Applies each informative reward function using `apply_reward_function` and aggregates the results.
            5. Updates `self.users_current_rewards` and `self.current_reward` with the aggregated decisive rewards.
            6. Updates `self.informative_current_rewards` with the aggregated informative rewards.

        Note:
            - The `compute_basic_reward` and `compute_fqsos_reward` functions are expected to be defined elsewhere in the class.
            - The `self.decisive_reward_functions` and `self.informative_reward_functions` dictionaries should contain the reward functions to be applied.
            - The `self.verbose` attribute controls the verbosity of the reward calculations.
        """
        self.users_current_rewards = np.zeros(self.K)
        self.users_current_basic_rewards = np.zeros(self.K)
        self.current_reward = 0
        # Pre-compute basic reward once to avoid duplicate calls
        self.basic_reward_per_user, self.basic_reward_total, basic_reward_details = compute_basic_reward(
            self.K, self._num_eavesdroppers, self.sinr_downlink_users, self.sinr_uplink_users,
                        self.SINR_E_d_k_l, self.SINR_E_u_k_l, self.eavesdropper_active,
                        verbose = True
        )
        self.user_reward_detail = basic_reward_details  

        #* Used to further determine if the episode is a success, failure or severe failure for Curriculum Learning approach.
        self.downlink_episode_success_condition += np.array([inner_dict['Downlink reward'] for inner_dict in basic_reward_details.values()])
        self.uplink_episode_success_condition +=  np.array([inner_dict['Uplink reward'] for inner_dict in basic_reward_details.values()])

        if self.eavesdropper_active:
            self.best_eavesdropping_episode_success_condition +=  np.array([inner_dict['Eavesdropping reward'] for inner_dict in basic_reward_details.values()])
            
        # Define a helper function to apply each reward function
        def apply_reward_function(reward_function, current_rewards_dict):
            if reward_function == compute_basic_reward:
                # Use pre-computed basic reward
                current_rewards_dict["basic_reward_details"] = basic_reward_details
                current_rewards_dict["basic_reward_per_user"] = self.basic_reward_per_user
                current_rewards_dict["total_basic_reward"] = self.basic_reward_total  # Use the total from pre-computation
                return self.basic_reward_per_user, self.basic_reward_total

            elif reward_function == compute_qos_reward:
                reward_per_user, total_reward = reward_function(
                    self.K, self._num_eavesdroppers,self.mu_1,self.mu_2,
                    self.target_secrecy_rate,
                    self.target_date_rate,
                    self.sinr_downlink_users, self.sinr_uplink_users,
                    self.SINR_E_d_k_l, self.SINR_E_u_k_l,
                    self.eavesdropper_active
                )
                current_rewards_dict["qos_reward_per_user"] = reward_per_user
                current_rewards_dict["total_qos_reward"] = total_reward
                return reward_per_user, total_reward
            
            elif reward_function == compute_minmax_reward:
                reward_per_user, total_reward = reward_function(
                    self.K, self._num_eavesdroppers,
                    self.sinr_downlink_users, self.sinr_uplink_users,
                    self.SINR_E_d_k_l, self.SINR_E_u_k_l, self.eavesdropper_active,
                )
                current_rewards_dict["minmax_reward_per_user"] = reward_per_user
                current_rewards_dict["total_minmax_reward"] = total_reward
                return reward_per_user, total_reward
        
            
            elif reward_function == compute_minmax_smoothed_reward:
                reward_per_user, total_reward = reward_function(
                    self.K, self._num_eavesdroppers,
                    self.sinr_downlink_users, self.sinr_uplink_users,
                    self.SINR_E_d_k_l, self.SINR_E_u_k_l, self.eavesdropper_active,
                    minmax_smoothed_p = self.p_f
                )
                current_rewards_dict["minmax_smoothed_reward_per_user"] = reward_per_user
                current_rewards_dict["total_minmax_smoothed_reward"] = total_reward
                return reward_per_user, total_reward
            # Add other reward functions here if needed
            return np.zeros(self.K), 0

        # Apply each decisive reward function and aggregate results
        decisive_rewards_results = []
        for reward_name, reward_function in self.decisive_reward_functions.items():
            result = apply_reward_function(reward_function, self.decisive_current_rewards)
            decisive_rewards_results.append(result)
            self.decisive_rewards[reward_name] = {
                "reward_per_user": result[0],
                "total_reward": result[1]
            }
        self.users_current_rewards = np.sum([result[0] for result in decisive_rewards_results], axis=0)
        self.current_reward = np.sum([result[1] for result in decisive_rewards_results])

        # Apply each informative reward function and aggregate results
        informative_rewards_results = []
        for reward_name, reward_function in self.informative_reward_functions.items():
            result = apply_reward_function(reward_function, self.informative_current_rewards)
            informative_rewards_results.append(result)
            self.informative_rewards[reward_name] = {
                "reward_per_user": result[0],
                "total_reward": result[1]
            }
        self.informative_current_rewards["users_current_rewards"] = np.sum([result[0] for result in informative_rewards_results], axis=0)
        self.informative_current_rewards["current_reward"] = np.sum([result[1] for result in informative_rewards_results])

        self.current_fairness = self.user_jain_fairness()
    

    def user_jain_fairness(self):
        if not self.basic_reward_per_user.size:
             return round(1/self.K,ndigits=4)   
        sum_rewards = np.sum(self.basic_reward_per_user)
        sum_squares = np.sum(self.basic_reward_per_user**2)
        if sum_squares == 0:
            return round(1/self.K,ndigits=4)  
        jain_index = (sum_rewards ** 2) / (self.K * sum_squares)
        return round(jain_index,ndigits=4) 
         

    def compute_all_fairness(self):
        self.current_fairness_values = {}
        for fairness_name, fairness_function in self.fairness_functions.items():
            if fairness_function == jain_fairness_index:
                self.current_fairness_value[fairness_name] = fairness_function(self.users_current_rewards)
        pass

    
    #? ----------------------------------------------     MANAGING THE MOVEMENT OF EAVESDROPPERS    ------------------------------------------------------------------------------------------------------

    def move_eavesdroppers(self, mobility_pattern):
        """Move the eavesdroppers according to the specified mobility pattern.

        Args:
            mobility_pattern (function): Function that defines the movement of the eavesdroppers.
        """
        self.eavesdroppers_positions = mobility_pattern(self.eavesdroppers_positions,
                                                        mobility_model = self.mobility_model_eavesdroppers,
                                                        obstacles = self.users_positions,
                                                        limits = self.eavesdropper_spawn_limits.flatten(),
                                                       random_numpy_generator = self.numpy_rng)
        self.compute_eavesdropper_channels()
        pass


     # ?  ----------------------------------------------     Power Patterns functions   ------------------------------------------------------------------------------------------------------

    def RIS_W_compute_power_pattern(self):
        """
        Precompute data for a single frame.

        Parameters:
        - Theta_Phi: Phase shifts.

        Returns:
        - Power pattern.
        """
        power_patterns = np.zeros((self.K, 360))
        for k in range(self.K):

            if self.debugging:
                Sended_Signal_Downlink_user_k =  self._W[:,k]
            else:
                Sended_Signal_Downlink_user_k =  self._W[:,k]

            E_total = np.zeros_like(self.angles, dtype=complex)
            for i, theta in enumerate(self.angles):
                spatial_phase = (2 * np.pi / self.lambda_h) * self._d_h[0] * np.arange(self.N_t) * np.sin(-theta)
                E_total[i] = np.sum(Sended_Signal_Downlink_user_k * np.exp(1j * spatial_phase))
            power_patterns[k] = np.abs(E_total) ** 2
        return power_patterns


    def RIS_downlink_compute_power_patterns(self):
        """
        Precompute data for a single frame.

        Parameters:
        - Theta_Phi: Phase shifts.

        Returns:
        - Power pattern.
        """
        power_patterns = np.zeros((self.K, 360))
        BS_RIS_channel = self._channel_matrices["H_BS_RIS"]
        for k in range(self.K):

            if self.debugging:
                Reflected_Signal_Downlink_user_k = self._Theta @ BS_RIS_channel @ self._W[:,k]
            else:
                Reflected_Signal_Downlink_user_k = self._Theta @ BS_RIS_channel @ self._W[:,k]

            E_total = np.zeros_like(self.angles, dtype=complex)
            for i, theta in enumerate(self.angles):
                spatial_phase = (2 * np.pi / self.lambda_h) * self._d_h[1] * np.arange(self.M) * np.sin(-theta)
                E_total[i] = np.sum(Reflected_Signal_Downlink_user_k * np.exp(1j * spatial_phase))
            power_patterns[k] = deepcopy(np.abs(E_total) ** 2)
        return power_patterns
    

    def RIS_uplink_compute_power_patterns(self):
        """
        Precompute data for a single frame.

        Parameters:
        - Theta_Phi: Phase shifts.

        Returns:
        - Power pattern.
        """
        power_patterns = np.zeros((self.K, 360))
        User_RIS_channels = self._channel_matrices["H_Users_RIS"]
        for k in range(self.K):
            Reflected_Signal_Uplink_user_k = (np.sqrt(self.P_users[k])  * self._Theta_Phi @ User_RIS_channels[k])[:,0]
            E_total = np.zeros_like(self.angles, dtype=complex)
            for i, theta in enumerate(self.angles):
                spatial_phase = (2 * np.pi / self.lambda_h) * self._d_h[1] * np.arange(self.M) * np.sin(-theta)
                E_total[i] = np.sum(Reflected_Signal_Uplink_user_k * np.exp(1j * spatial_phase))
            power_patterns[k] = np.abs(E_total) ** 2
        return power_patterns