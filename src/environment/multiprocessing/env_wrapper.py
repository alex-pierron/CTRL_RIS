"""
Vectorized environment wrappers used in RIS Duplex experiments.

This module offers two vectorization strategies, inspired by OpenAI Baselines, to
batch multiple environment instances and interact with them using a unified API:

- DummyVecEnv: runs environments sequentially in the main process; useful for
  debugging or when `num_envs == 1` to avoid IPC overhead.
- SubprocVecEnv: runs environments in subprocesses and communicates through
  pipes; recommended when `num_envs > 1` and environment `step()` is expensive.

It also provides small helpers to construct train/eval environments from config
dicts: `make_train_env` and `make_eval_env`.

Better Comments legend used throughout the file:
- TODO: future improvements or refactors (non-functional)
- NOTE: important behavior or design intent
- !: important runtime remark
- ?: questioning a choice or highlighting an assumption
"""
import os
import contextlib
import numpy as np
from abc import ABC, abstractmethod
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection
from multiprocessing.connection import wait
from copy import deepcopy
from multiprocessing import TimeoutError
import logging
from src.environment import RIS_Duplex

# ============================================================================
# Utilities
# ============================================================================

class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)


@contextlib.contextmanager
def clear_mpi_env_vars():
    """Temporarily clear MPI-related environment variables.

    NOTE: `from mpi4py import MPI` calls MPI_Init by default. If the child
    process inherits MPI environment variables, MPI may treat it as an MPI
    process and hang. This context manager clears such variables while
    starting subprocesses.
    """
    removed_environment = {}
    for k, v in list(os.environ.items()):
        for prefix in ['OMPI_', 'PMI_']:
            if k.startswith(prefix):
                removed_environment[k] = v
                del os.environ[k]
    try:
        yield
    finally:
        os.environ.update(removed_environment)

class VecEnv(ABC):
    """
    An abstract asynchronous, vectorized environment.
    Used to batch data from multiple copies of an environment, so that
    each observation becomes an batch of observations, and expected action is a batch of actions to
    be applied per-environment.
    """
    closed = False

    def __init__(self, num_envs, observation_space, action_space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space

    @abstractmethod
    def reset(self):
        """
        Reset all the environments and return an array of
        observations, or a dict of observation arrays.

        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.
        """
        pass

    @abstractmethod
    def step_async(self, states, actions):
        """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.

        You should not call this if a step_async run is
        already pending.
        """
        pass

    @abstractmethod
    def step_wait(self):
        """
        Wait for the step taken with step_async().

        Returns (obs, rews, dones, infos):
         - obs: an array of observations, or a dict of
                arrays of observations.
         - rews: an array of rewards
         - dones: an array of "episode done" booleans
         - infos: a sequence of info objects
        """
        pass

    def close_extras(self):
        """
        Clean up the extra resources, beyond what's in this base class.
        Only runs when not self.closed.
        """
        pass

    def close(self):
        if self.closed:
            return
        self.close_extras()
        self.closed = True

    def step(self, states, actions):
        """
        Step the environments synchronously.

        This is available for backwards compatibility.
        """
        # existing step
        self.step_async(states, actions)
        return self.step_wait()
    

class DummyVecEnv(VecEnv):
    """
    VecEnv that does runs multiple environments sequentially, that is,
    the step and reset commands are send to one environment at a time.
    Useful when debugging and when num_env == 1 (in the latter case,
    avoids communication overhead)
    """
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        super().__init__(len(self.envs), env.observation_space, env.action_space)
        self.nremotes = 1
        self.actions = None
        self.state = None
        self.action_dim = getattr(self.envs[0], "action_dim")
        self.state_dim = getattr(self.envs[0], "state_dim")
        self.num_users = getattr(self.envs[0], "K")
        self.num_eavesdroppers = getattr(self.envs[0], "num_eavesdroppers")
        self.users_max_power = getattr(self.envs[0], "P_max")
        self.BS_transmit_antennas = getattr(self.envs[0], "N_t")
        self.BS_position = getattr(self.envs[0], "BS_position")
        self.lambda_h = getattr(self.envs[0], "lambda_h")
        self.d_h = getattr(self.envs[0], "d_h")
        self.RIS_position = getattr(self.envs[0], "RIS_position")
        self.moving_eavesdroppers = getattr(self.envs[0], "moving_eavesdroppers")
        self.M = getattr(self.envs[0], "M")
        self.verbose = getattr(self.envs[0], "verbose")
        self.env_config = self.envs[0].env_config
    def num_step(self):
        return getattr(self.envs[0], "num_step")
    
    def num_episode(self):
        return getattr(self.envs[0], "num_episode")
    
    def get_states(self):
        """Return the current state for each underlying environment."""
        return [env.get_state() for env in self.envs]
    
    def get_W(self):
        """Return the current BS beamforming matrix `W` from the first env."""
        return [getattr(env, "W") for env in self.envs][0]
    
    def get_Theta(self):
        """Return the current RIS phase-shift matrix `Theta` from the first env."""
        return [getattr(env, "Theta") for env in self.envs][0]
    
    
    def get_channel_matrices(self):
        """Return the channel matrices dict from the first env."""
        return [getattr(env, "channel_matrices") for env in self.envs][0]
    

    def get_phis(self):
        """Return the phase-noise matrix `Phi` from the first env."""
        return [getattr(env, "Phi") for env in self.envs][0]
    
    def get_theta_phis(self):
        """Return the product `Theta_Phi` from the first env."""
        return [getattr(env, "Theta_Phi") for env in self.envs][0]

    def episode_action_noise(self):
        """Return callable(s) that expose episode-level action noise."""
        return [getattr(env, "get_episode_action_noise") for env in self.envs]
    
    def get_eavesdropper_rewards(self):
        """Return the eavesdropper reward from the first env."""
        return [env.eavesdropper_reward() for env in self.envs][0]
    

    def get_additionnal_informations(self):
        """Return per-user reward breakdown from the first env."""
        return [env.user_reward_detail for env in self.envs][0]

    def get_users_positions(self):
        """Return current users positions from the first env."""
        return [env.users_positions for env in self.envs][0]
    
    def get_downlink_sum_for_success_conditions(self):
        """Return aggregated downlink success metrics for curriculum signals."""
        return np.reshape([env.downlink_episode_success_condition for env in self.envs][0], (1,self.num_users))

    def get_uplink_sum_for_success_conditions(self):
        """Return aggregated uplink success metrics for curriculum signals."""
        return np.reshape([env.uplink_episode_success_condition for env in self.envs][0], (1,self.num_users))

    def get_best_eavesdropper_sum_for_success_conditions(self):
        """Return aggregated best-eavesdropper metrics for curriculum signals."""
        return np.reshape([env.best_eavesdropping_episode_success_condition for env in self.envs][0], (1,self.num_users))
    
    
    def get_eavesdroppers_positions(self):
        """Return current eavesdroppers positions from the first env."""
        return [env.eavesdroppers_positions for env in self.envs][0]
    
    def get_W_power_patterns(self):
        """Return precomputed BS array power patterns from the first env."""
        return [env.RIS_W_compute_power_pattern() for env in self.envs][0]
    
    def get_downlink_power_patterns(self):
        """Return precomputed RIS downlink power patterns from the first env."""
        return [env.RIS_downlink_compute_power_patterns() for env in self.envs][0]
    
    def get_uplink_power_patterns(self):
        """Return precomputed RIS uplink power patterns from the first env."""
        return [env.RIS_uplink_compute_power_patterns() for env in self.envs][0]
    
    def get_user_jain_fairness(self):
        """Return current Jain fairness index computed in the first env."""
        return [env.user_jain_fairness() for env in self.envs][0]
    
    def get_basic_reward(self):
        """Return latest total basic reward from the first env."""
        return [env.basic_reward_total for env in self.envs][0]
    
    def get_user_info(self):
        """Return comprehensive user information dictionary from the first env."""
        return [env.get_user_info() for env in self.envs][0]
    
    def get_eavesdropper_info(self):
        """Return comprehensive eavesdropper information dictionary from the first env."""
        return [env.get_eavesdropper_info() for env in self.envs][0]
    
    def get_user_communication_rates(self):
        """Return communication rates (downlink, uplink, total) for all users from the first env."""
        return [env.get_user_communication_rates() for env in self.envs][0]
    
    def get_eavesdropper_communication_rates(self):
        """Return communication rates (downlink, uplink, total) for all eavesdroppers from the first env."""
        return [env.get_eavesdropper_communication_rates() for env in self.envs][0]
    
    def get_user_signal_strengths(self):
        """Return min/max signal strengths for all users from the first env."""
        return [env.get_user_signal_strengths() for env in self.envs][0]
    
    def get_eavesdropper_signal_strengths(self):
        """Return min/max signal strengths for all eavesdroppers from the first env."""
        return [env.get_eavesdropper_signal_strengths() for env in self.envs][0]
    
    def step_async(self, state, actions):
        self.actions = actions
        self.state = state

    def step_wait(self):
        results = [env.step(self.state, self.actions) for env in self.envs]
        states, current_actions, current_rewards, next_states = map(list, zip(*results))
        self.actions = None
        return self._flatten(states), self._flatten(current_actions), self._flatten(current_rewards), self._flatten(next_states)
    
    
    def scripted_actions(self, script_function, opponents_actions):
        """
        Calculate scripted actions in the environments synchronously.

        This is available for backwards compatibility.
        """
        results = [env.apply_script_rendering(script_function, opponents_actions) for env in self.envs]
        results = np.array(results)[0]
        return results
    
    def reset(self, difficulty_config = None ):
        """Reset all underlying environments.

        Args:
            difficulty_config: Optional curriculum tuple passed to each env.
        Returns:
            Flattened observations from all environments.
        """
        if difficulty_config:
            obss = [env.reset(chosen_difficulty_config = difficulty_config[0]) for env in self.envs]
        else:
            obss = [env.reset() for env in self.envs]
        return self._flatten(obss)

    def close(self):
        for env in self.envs:
            env.close()

    def render(self, mode, filepath):
        if mode == 'txt':
            self.envs[0].render(mode, filepath)

    @classmethod
    def _flatten(cls, v):
        assert isinstance(v, (list, tuple))
        assert len(v) > 0

        if isinstance(v[0], dict):
            return {k: np.stack([v_[k] for v_ in v]) for k in v[0].keys()}
        else:
            return np.stack(v)
        


def worker(remote: Connection, parent_remote: Connection, env_fn_wrappers):
    """Subprocess worker hosting one or more environment instances.

    The worker receives commands via a `Pipe` and returns results. Each worker
    may run several envs in series (`in_series > 1`).

    Args:
        remote: Connection used by the subprocess to send/receive data.
        parent_remote: Connection used by the parent process; closed here.
        env_fn_wrappers: Wrapper holding a split of env factory functions.
    """
    def step_env(env, state, action):
        state, current_action, current_reward, next_state = env.step(state, action)
        return state, current_action, current_reward, next_state

    def reset_env(env, chosen_difficulty_config):
        env.reset(chosen_difficulty_config = chosen_difficulty_config)
        pass

    def get_state(env):
        return env.get_state()
    
    def get_user_jain_fairness(env):
        return env.user_jain_fairness()
    
    def get_users_positions(env):
        return env.users_positions 
    
    def get_eavesdroppers_positions(env):
        return env.eavesdroppers_positions
    
    def get_num_step(env):
        return getattr(env, "num_step")
    
    def get_num_episode(env):
        return getattr(env, "num_episode")
    
    def get_episode_action_noise(env):
        return env.get_episode_action_noise() 
    
    def execute_script(env,script_function, opponent_action):
        return env.apply_script(script_function, opponent_action)
    
    def get_eavesdroppers_rewards(env):
        return env.eavesdropper_reward()
    
    def get_basic_reward(env):
        return env.basic_reward_total
    
    def get_user_info(env):
        return env.get_user_info()
    
    def get_eavesdropper_info(env):
        return env.get_eavesdropper_info()
    
    def get_user_communication_rates(env):
        return env.get_user_communication_rates()
    
    def get_eavesdropper_communication_rates(env):
        return env.get_eavesdropper_communication_rates()
    
    def get_user_signal_strengths(env):
        return env.get_user_signal_strengths()
    
    def get_eavesdropper_signal_strengths(env):
        return env.get_eavesdropper_signal_strengths()
    
    def get_decisive_rewards(env):
        return env.decisive_rewards
    
    def get_informative_rewards(env):
        return env.informative_rewards
    
    def get_theta_phis(env):
        return getattr(env, "Theta_Phi")
    
    def get_downlink_success_condition(env):
        return env.downlink_episode_success_condition
    
    def get_uplink_success_condition(env):
        return env.uplink_episode_success_condition
    
    def get_best_eavesdropper_sum_for_success_condition(env):
        return env.best_eavesdropping_episode_success_condition
    
    parent_remote.close()
    try:
        envs = [env_fn_wrapper() for env_fn_wrapper in env_fn_wrappers.x]
        print("Worker started environments:", envs)
    except Exception as e:
        print("Error creating environments in worker:", e)
    try:
        while True:
            cmd, data = remote.recv()
            
            if cmd == 'step':
                states, actions = data
                remote.send([step_env(env, state, action) for env, state, action in zip(envs, states, actions)])
            
            elif cmd == 'reset':
                chosen_difficulty_config = data
                if chosen_difficulty_config is not None:
                    remote.send([reset_env(env,chosen_difficulty_config) for env in envs])
                else:
                    remote.send([env.reset() for env in envs])
            
            elif cmd == 'close':
                remote.close()
                break
            
            elif cmd =="get_env_config":
                remote.send(CloudpickleWrapper(envs[0].env_config))

            elif cmd =="get_num_RIS_elements":
                remote.send(CloudpickleWrapper(envs[0].M))

            elif cmd =="get_BS_position":
                remote.send(CloudpickleWrapper(envs[0].BS_position))
            
            elif cmd =="get_d_h":
                remote.send(CloudpickleWrapper(envs[0].d_h))

            elif cmd =="get_lambda_h":
                remote.send(CloudpickleWrapper(envs[0].lambda_h))

            elif cmd =="get_RIS_position":
                remote.send(CloudpickleWrapper(envs[0].RIS_position))

            elif cmd =="get_users_positions":
                remote.send([get_users_positions(env) for env in envs])

            elif cmd =="get_eavesdroppers_positions":
                remote.send([get_eavesdroppers_positions(env) for env in envs])

            elif cmd =="get_moving_eavesdroppers":
                remote.send(CloudpickleWrapper(envs[0].moving_eavesdroppers))

            elif cmd =="get_state":
                remote.send([get_state(env) for env in envs])

            elif cmd =="get_user_jain_fairness":
                remote.send([get_user_jain_fairness(env) for env in envs])
            
            elif cmd =="get_episode_action_noise":
                remote.send([get_episode_action_noise(env) for env in envs])
            
            elif cmd == "get_basic_reward":
                remote.send([get_basic_reward(env) for env in envs])

            elif cmd == "get_user_info":
                remote.send([get_user_info(env) for env in envs])

            elif cmd == "get_eavesdropper_info":
                remote.send([get_eavesdropper_info(env) for env in envs])

            elif cmd == "get_user_communication_rates":
                remote.send([get_user_communication_rates(env) for env in envs])

            elif cmd == "get_eavesdropper_communication_rates":
                remote.send([get_eavesdropper_communication_rates(env) for env in envs])

            elif cmd == "get_user_signal_strengths":
                remote.send([get_user_signal_strengths(env) for env in envs])

            elif cmd == "get_eavesdropper_signal_strengths":
                remote.send([get_eavesdropper_signal_strengths(env) for env in envs])

            elif cmd =="get_decisive_rewards":
                remote.send([get_decisive_rewards(env) for env in envs])
            
            elif cmd =="get_informative_rewards":
                remote.send([get_informative_rewards(env) for env in envs])

            elif cmd =="get_eavesdroppers_rewards":
                remote.send([get_eavesdroppers_rewards(env) for env in envs])

            elif cmd =="get_theta_phis":
                remote.send([get_theta_phis(env) for env in envs])

            elif cmd == "get_downlink_sum_for_success_conditions":
                remote.send([get_downlink_success_condition(env) for env in envs])

            elif cmd == "get_uplink_sum_for_success_conditions":
                remote.send([get_uplink_success_condition(env) for env in envs])

            elif cmd == "get_best_eavesdropper_sum_for_success_conditions":
                remote.send([get_best_eavesdropper_sum_for_success_condition(env) for env in envs])
            elif cmd == 'get_state_dim':
                remote.send(CloudpickleWrapper(envs[0].state_dim))

            elif cmd == 'get_action_dim':
                remote.send(CloudpickleWrapper(envs[0].action_dim))

            elif cmd == 'get_num_users':
                remote.send(CloudpickleWrapper(envs[0].num_users))

            elif cmd == 'get_num_eavesdropper':
                remote.send(CloudpickleWrapper(envs[0].num_eavesdroppers))

            elif cmd == 'get_users_max_power':
                remote.send(CloudpickleWrapper(envs[0].maximum_power))
            
            elif cmd == 'get_BS_transmit_antennas':
                remote.send(CloudpickleWrapper(envs[0].BS_transmit_antennas))

            elif cmd == 'num_step':
                remote.send(CloudpickleWrapper(get_num_step(envs[0])))

            elif cmd == 'num_episode':
                remote.send(CloudpickleWrapper(get_num_episode(envs[0])))

            elif cmd == 'get_spaces':
                remote.send(CloudpickleWrapper((envs[0].observation_space, envs[0].action_space)))
            
            elif cmd == 'scripted_actions':
                script_function, opponent_actions = data 
                results = [execute_script(env, script_function, opponent_actions) for env in envs]
                remote.send(results)
            
            elif cmd == 'get_spaces':
                remote.send(CloudpickleWrapper((envs[0].observation_space, envs[0].action_space)))
            elif cmd == 'get_num_agents':
                remote.send(CloudpickleWrapper((getattr(envs[0], "num_agents", 1))))
            else:
                raise NotImplementedError
    except KeyboardInterrupt:
        print('SubprocVecEnv worker: got KeyboardInterrupt')
    finally:
        for env in envs:
            pass # TODO:: replace with a proper env.close() method



class SubprocVecEnv(VecEnv):
    """
    VecEnv that runs multiple environments in parallel in subproceses and communicates with them via pipes.
    Recommended to use when num_envs > 1 and step() can be a bottleneck.
    """
    def __init__(self, env_fns, context='spawn', in_series=1):
        """
        Initialize the SubprocVecEnv.

        Args:
            env_fns (iterable): Iterable of callables - functions that create environments to run in subprocesses.
                Need to be cloud-pickleable.
            context (str, optional): The context in which the subprocesses are created. Defaults to 'spawn'.
            in_series (int, optional): Number of environments to run in series in a single process. Defaults to 1.
                For example, when len(env_fns) == 12 and in_series == 3, it will run 4 processes, each running 3 envs in series.
        """
        self.waiting = False
        self.closed = False
        self.in_series = in_series
        nenvs = len(env_fns)
        self.nenvs = nenvs
        assert nenvs % in_series == 0, "Number of envs must be divisible by number of envs to run in series"
        self.nremotes = nenvs // in_series
        env_fns = np.array_split(env_fns, self.nremotes)
        # create Pipe connections to send/recv data from subprocesses,
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(self.nremotes)])
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            with clear_mpi_env_vars():
                p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv().x
        super().__init__(nenvs, observation_space, action_space)

        self.remotes[0].send(('get_num_RIS_elements', None))
        self.M = self.remotes[0].recv().x

        self.remotes[0].send(('get_state_dim', None))
        self.state_dim = self.remotes[0].recv().x

        self.remotes[0].send(('get_action_dim', None))
        self.action_dim = self.remotes[0].recv().x

        self.remotes[0].send(('get_env_config', None))
        self.env_config = self.remotes[0].recv().x

        self.remotes[0].send(('get_num_users', None))
        self.num_users = self.remotes[0].recv().x

        self.remotes[0].send(('get_num_eavesdropper', None))
        self.num_eavesdroppers = self.remotes[0].recv().x

        self.remotes[0].send(('get_users_max_power', None))
        self.users_max_power = self.remotes[0].recv().x

        self.remotes[0].send(('get_BS_transmit_antennas', None))
        self.BS_transmit_antennas = self.remotes[0].recv().x

        self.remotes[0].send(('get_BS_position', None))
        self.BS_position = self.remotes[0].recv().x
        
        self.remotes[0].send(('get_d_h', None))
        self.d_h = self.remotes[0].recv().x

        self.remotes[0].send(('get_lambda_h', None))
        self.lambda_h = self.remotes[0].recv().x

        self.remotes[0].send(('get_RIS_position', None))
        self.RIS_position = self.remotes[0].recv().x

        self.remotes[0].send(('get_moving_eavesdroppers', None))
        self.moving_eavesdroppers = self.remotes[0].recv().x
    
    def step_async(self, states, actions):
        """
        Asynchronously send states and actions to the environments.

        Args:
            states: The states corresponding to each environment.
            actions: The actions to be sent to the environments.
        """
        self._assert_not_closed()
        
        # Split states and actions according to the number of remote environments
        states = np.array_split(states, self.nremotes)
        actions = np.array_split(actions, self.nremotes)
        
        # Send states and actions to the corresponding environments
        for remote, state, action in zip(self.remotes, states, actions):
            remote.send(('step', (state, action)))

        self.waiting = True


    def step_wait(self):
        """
        Wait for the environments to complete their steps and return the results.

        Returns:
            Tuple: A tuple containing the flattened observations, rewards, dones, and infos.
        """
        self._assert_not_closed()
        results = [remote.recv() for remote in self.remotes]
        results = self._flatten_series(results)  # [[tuple] * in_series] * nremotes => [tuple] * nenvs
        self.waiting = False
        states, current_actions, current_rewards, next_states = zip(*results)
        return self._flatten(states), self._flatten(current_actions), self._flatten(current_rewards),self._flatten(next_states)

    def get_users_positions(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('get_users_positions', None))
        # Collect results from all environments
        users_positions = [remote.recv() for remote in self.remotes]
        return users_positions
    
    def get_eavesdroppers_positions(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('get_eavesdroppers_positions', None))
        # Collect results from all environments
        eavesdroppers_positions = [remote.recv() for remote in self.remotes]
        return eavesdroppers_positions
    
    def get_eavesdroppers_rewards(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('get_eavesdroppers_rewards', None))
        # Collect results from all environments
        eavesdroppers_rewards = [remote.recv() for remote in self.remotes]
        return eavesdroppers_rewards
    
    def get_basic_reward(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('get_basic_reward', None))
        basic_reward = np.array([remote.recv() for remote in self.remotes])
        return basic_reward
    
    def get_user_info(self):
        """Return comprehensive user information dictionary from all environments."""
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('get_user_info', None))
        # Collect results from all environments
        user_info_list = [remote.recv() for remote in self.remotes]
        return user_info_list
    
    def get_eavesdropper_info(self):
        """Return comprehensive eavesdropper information dictionary from all environments."""
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('get_eavesdropper_info', None))
        eavesdropper_info_list = [remote.recv() for remote in self.remotes]
        return eavesdropper_info_list
    
    def get_user_communication_rates(self):
        """Return communication rates (downlink, uplink, total) for all users from all environments."""
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('get_user_communication_rates', None))
        rates_list = [remote.recv() for remote in self.remotes]
        return rates_list
    
    def get_eavesdropper_communication_rates(self):
        """Return communication rates (downlink, uplink, total) for all eavesdroppers from all environments."""
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('get_eavesdropper_communication_rates', None))
        rates_list = [remote.recv() for remote in self.remotes]
        return rates_list
    
    def get_user_signal_strengths(self):
        """Return min/max signal strengths for all users from all environments."""
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('get_user_signal_strengths', None))
        strengths_list = [remote.recv() for remote in self.remotes]
        return strengths_list
    
    def get_eavesdropper_signal_strengths(self):
        """Return min/max signal strengths for all eavesdroppers from all environments."""
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('get_eavesdropper_signal_strengths', None))
        strengths_list = [remote.recv() for remote in self.remotes]
        return strengths_list
    
    def get_decisive_rewards(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('get_decisive_rewards', None))
        decisive_rewards = [remote.recv() for remote in self.remotes]
        return decisive_rewards
    
    def get_informative_rewards(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('get_informative_rewards', None))
        informative_rewards = [remote.recv() for remote in self.remotes]
        return informative_rewards
    
    def get_theta_phis(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('get_theta_phis', None))
        # Collect results from all environments
        theta_phis = [remote.recv() for remote in self.remotes]
        return theta_phis
    
    def get_downlink_sum_for_success_conditions(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('get_downlink_sum_for_success_conditions', None))
        # Collect results from all environments
        downlink_sum_for_success_conditions = [remote.recv() for remote in self.remotes]
        return downlink_sum_for_success_conditions
    
    def get_uplink_sum_for_success_conditions(self):
        """Collects the uplinks signals for all users and for each environment (2d array). Used to further compute wether or not the success condition for a specific environment episode is filled.

        Returns:
            _type_: _description_
        """
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('get_uplink_sum_for_success_conditions', None))
        # Collect results from all environments
        downlink_sum_for_success_conditions = [remote.recv() for remote in self.remotes]
        return downlink_sum_for_success_conditions

    def get_best_eavesdropper_sum_for_success_conditions(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('get_best_eavesdropper_sum_for_success_conditions', None))
        # Collect results from all environments
        best_eavesdropper_sum_for_success_conditions = [remote.recv() for remote in self.remotes]
        return best_eavesdropper_sum_for_success_conditions

    def get_states(self):
        """
        Synchronously fetch states from all environments.
        """
        self.get_states_async()
        return self.get_states_wait()
    
    def get_states_async(self):
        """
        Asynchronously request states from the environments.
        """
        self._assert_not_closed()
        
        # Send request to all environments
        for remote in self.remotes:
            remote.send(('get_state', None))
        self.waiting = True


    def get_states_wait(self):
        """
        Wait for all environments to return their states.

        Returns:
            np.array: Flattened states from all environments.
        """
        self._assert_not_closed()
        
        # Receive the states from all environments
        results = [remote.recv() for remote in self.remotes]
        
        self.waiting = False
        return self._flatten(results)

    
    def get_jain_fairness_async(self):
        """
        """
        self._assert_not_closed()
        
        # Send request to all environments
        for remote in self.remotes:
            remote.send(('get_user_jain_fairness', None))
        self.waiting = True

    def get_jain_fairness_wait(self):
        """
        """
        self._assert_not_closed()
        
        # Receive the states from all environments
        results = [remote.recv() for remote in self.remotes]
        
        self.waiting = False
        return self._flatten(results)
    

    def get_jain_fairness(self):
        """
        """
        self.get_jain_fairness_async()
        return self.get_jain_fairness_wait()
    

    def reset(self, chosen_difficulties = None):
        """
        Reset the environments.

        Returns:
            
        """
        self._assert_not_closed()
        if chosen_difficulties is None:
            for remote in self.remotes:
                remote.send(('reset', None))
            results = [remote.recv() for remote in self.remotes]
            pass
        else:
            for remote, chosen_difficulty in zip(self.remotes,chosen_difficulties):
                remote.send(('reset', chosen_difficulty))
            results = [remote.recv() for remote in self.remotes]
            pass
        # PPO difference: return nothing as current code expects, PPO trainers call get_states next.
    
    def scripted_actions_async(self, script_function, opponents_actions):
        """
        Send asynchronous commands to execute scripted actions in the environments.

        Args:
            script_function (function): The function representing the scripted actions.
            opponents_actions (numpy.ndarray): Actions of the opponents.

        """
        self._assert_not_closed()
        opponent_action = np.array_split(opponents_actions, self.nremotes) 
        for remote, opponent_action in zip(self.remotes, opponents_actions):
            remote.send(('scripted_actions', (script_function, opponent_action) ) )
        self.waiting = True

    def close_extras(self):
        """
        Close any additional resources used by the SubprocVecEnv.
        """
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()

    def _assert_not_closed(self):
        assert not self.closed, "Trying to operate on a SubprocVecEnv after calling close()"

    @classmethod
    def _flatten(cls, v):
        assert isinstance(v, (list, tuple))
        assert len(v) > 0

        if isinstance(v[0], dict):
            return {k: np.stack([v_[k] for v_ in v]) for k in v[0].keys()}
        else:
            return np.stack(v)

    @classmethod
    def _flatten_series(cls, v):
        assert isinstance(v, (list, tuple))
        assert len(v) > 0
        assert all([len(v_) > 0 for v_ in v])

        return [v__ for v_ in v for v__ in v_]
    

def make_train_env(all_args_env, all_args_training):
    """
    Creates and returns a training environment based on the provided arguments.

    Args:
        all_args_env (dictionnary): A dictionnary containing all the arguments for configuring the environment.

    Returns:
        object: The training environment.

    Raises:
        NotImplementedError: If the specified environment is not supported.
    """
    base_seed = all_args_env.get('env_seed', 42) 
    all_args_env['user_seed'] = deepcopy(base_seed)
    def get_env_fn(rank):
        """
        Returns an initialization function for the environment.

        Args:
            rank (int): The rank of the process.

        Returns:
            function: An initialization function that creates and seeds the environment.

        Raises:
            NotImplementedError: If the specified environment is not supported.
        """

        def init_env():
            """
            Initializes the environment based on the specified environment name.

            Returns:
                env: The initialized environment object.

            Raises:
                NotImplementedError: If the specified environment name is not supported.
            """
            new_seed = deepcopy(base_seed) + rank * 1000
            if all_args_env.get("env_name","RIS_Duplex_default") == "RIS_Duplex_default":
                all_args_env['env_seed'] = deepcopy(new_seed)
                env = RIS_Duplex(all_args_env)
            else:
                logging.error("Can not support the " + all_args_env.env_name + "environment.")
                raise NotImplementedError
            return env

        return init_env
    
    n_rollout_threads = all_args_training.get("n_rollout_threads", 1)
    if n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])


def make_eval_env(all_args_env, all_args_training):
    """
    Creates and returns an evaluation environment based on the provided arguments.

    Args:
        all_args_env (dict): A dictionary containing all the arguments for configuring the environment.

    Returns:
        object: The evaluation environment.

    Raises:
        NotImplementedError: If the specified environment is not supported.
    """
    modified_all_args_env = deepcopy(all_args_env)
    base_seed = all_args_env.get('eval_env_seed', 420)
    modified_all_args_env['user_seed'] = deepcopy(base_seed)
    modified_all_args_env['users_position_changing'] = True
    modified_all_args_env['eaves_position_changing'] = True
    def get_env_fn(rank):
        """
        Returns an initialization function for the evaluation environment.

        Args:
            rank (int): The rank of the process.

        Returns:
            function: An initialization function that creates and seeds the environment.

        Raises:
            NotImplementedError: If the specified environment is not supported.
        """

        def init_env():
            """
            Initializes the evaluation environment based on the specified scenario name.

            Returns:
                env: The initialized environment object.

            Raises:
                NotImplementedError: If the specified scenario name is not supported.
            """
            new_seed = all_args_env.get('eval_env_seed', 4200) + rank * 1001
            modified_all_args_env['env_seed'] = deepcopy(new_seed)
            if all_args_env.get("env_name","RIS_Duplex_default") == "RIS_Duplex_default":
                modified_all_args_env['env_seed'] = deepcopy(new_seed)
                env = RIS_Duplex(modified_all_args_env)
            else:
                logging.error("Can not support the " + all_args_env.env_name + "environment.")
                raise NotImplementedError
            return env

        return init_env
    n_rollout_threads = all_args_training.get("n_eval_rollout_threads", 1)
    if n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])
