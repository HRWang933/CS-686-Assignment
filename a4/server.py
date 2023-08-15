import threading
import queue
import pickle
import collections
from copy import deepcopy
import os
import sys
import traceback
import json

import numpy as np

from wumpus_env import *
import rl_soln

np.set_printoptions(threshold=np.inf)

EPSILON_GREEDY_TESTS_FILE = 'select_action_epsilon_greedy.json'
SOFTMAX_TESTS_FILE = 'select_action_softmax.json'
OPTIMISTIC_UTILITY_TESTS_FILE = 'select_action_optimistically.json'
Q_LEARNING_TESTS_FILE = 'active_q_learning.json'
SARSA_TESTS_FILE = 'active_sarsa.json'

WUMPUS_ENVS_FILE = 'envs.json'
WUMPUS_ENV_SEED = 0

TEST_DIR = 'tests/'

# Time thresholds
ACTION_SELECTION_TIMEOUT = 2
Q_LEARNING_TIMEOUT = 120
SARSA_TIMEOUT = 120

# Tolerance for elementwise equality for np array outputs
TOLERANCE = 1e-3

def compare_np_arrays(ret, func_exout):
    '''
    Compares numpy arrays, returning true if corresponding elements are less than or equal to the tolerance.
    :param ret: The value returned by a test
    :param func_exout: The correct value of the test
    :return: True if the arrays are close. False otherwise.
    '''
    return np.allclose(ret, func_exout, rtol=0, atol=TOLERANCE)

def load_env_dicts_from_file(envs_file_path):
    '''
    Loads dictionaries for environment initialization from a JSON file.
    :param envs_file_path: The path of a JSON file containing information on constructing the environments.
    :return: A dictionary containing values to be passed to an environment's constructor
    '''
    with open(envs_file_path) as f:
        envs_dict = json.load(f)
    return envs_dict

def dict_to_env(envs_dict, env_name):
    '''
    Instantiates an environment from a dictionary containing values for the constructor's arguments.
    :param envs_dict: Dict containing constructor arguments for various environments
    :param env_name: The identifier of the environment to instantiate
    '''
    env_dict = envs_dict[env_name]
    env = WumpusWorld(env_dict['width'], env_dict['height'], env_dict['entrance'], env_dict['heading'],
                      env_dict['wumpus'], env_dict['pits'], env_dict['gold'], env_dict['max_steps'], env_dict['kappa'])
    env.seed(WUMPUS_ENV_SEED)
    return env

def load_tests(test_file_path):
    '''
    Loads serialized test information as dictionaries.
    :param test_file_path: A path containing serialized test information
    '''
    with open(test_file_path) as f:
        data = json.load(f)
    return data

def save_tests(test_file_path, tests_dict):
    '''
    Serializes test information as a JSON file.
    :param test_file_path: The path to which test information will be saved
    :param tests_dict: A dictionary containing test information
    '''
    with open(test_file_path, 'w') as json_file:
        json.dump(tests_dict, json_file, indent=3)


class Test:

    def __init__(self, gen_soln=False):

        self.envs_dict = load_env_dicts_from_file(os.path.join(TEST_DIR, WUMPUS_ENVS_FILE))
        self.env_deterministic_4x4_textbook = dict_to_env(self.envs_dict, "deterministic_textbook_4x4_env")
        self.env_noisy_4x4_textbook = dict_to_env(self.envs_dict, "noisy_textbook_4x4_env")
        self.env_deterministic_3x3 = dict_to_env(self.envs_dict, "deterministic_3x3_env")
        self.env_noisy_3x3 = dict_to_env(self.envs_dict, "noisy_3x3_env")
        if gen_soln:
            self._gen_soln()

    def _get_env_from_identifier(self, env_name):
        if env_name == "deterministic_textbook_4x4_env":
            env = self.env_deterministic_4x4_textbook
        elif env_name == "noisy_textbook_4x4_env":
            env = self.env_noisy_4x4_textbook
        elif env_name == "deterministic_3x3_env":
            env = self.env_deterministic_3x3
        elif env_name == "noisy_3x3_env":
            env = self.env_noisy_3x3
        else:
            raise Exception('Unknown environment identifier.')
        env.seed(WUMPUS_ENV_SEED)
        return env

    def _gen_soln(self):
        """
        Generates solutions and save them to test files in JSON format.
        """

        print('Generating solutions for SELECT_ACTION_EPSILON_GREEDY tests.')
        epsilon_greedy_tests = load_tests(os.path.join(TEST_DIR, EPSILON_GREEDY_TESTS_FILE))
        for _, test in epsilon_greedy_tests.items():
            np.random.seed(test['seed'])
            Q = np.array(test['Q'])
            a = rl_soln.select_action_epsilon_greedy(test['s'], Q, test['epsilon'])
            test['soln'] = int(a)
        save_tests(os.path.join(TEST_DIR, EPSILON_GREEDY_TESTS_FILE), epsilon_greedy_tests)

        print('Generating solutions for SELECT_ACTION_SOFTMAX tests.')
        softmax_tests = load_tests(os.path.join(TEST_DIR, SOFTMAX_TESTS_FILE))
        for _, test in softmax_tests.items():
            np.random.seed(test['seed'])
            Q = np.array(test['Q'])
            a = rl_soln.select_action_softmax(test['s'], Q, test['T'])
            test['soln'] = int(a)
        save_tests(os.path.join(TEST_DIR, SOFTMAX_TESTS_FILE), softmax_tests)

        print('Generating solutions for SELECT_ACTION_OPTIMISTICALLY tests.')
        optimistic_utility_tests = load_tests(os.path.join(TEST_DIR, OPTIMISTIC_UTILITY_TESTS_FILE))
        for _, test in optimistic_utility_tests.items():
            np.random.seed(test['seed'])
            Q = np.array(test['Q'])
            N_sa = np.array(test['N_sa'])
            a = rl_soln.select_action_optimistically(test['s'], Q, N_sa, test['N_e'], test['R_plus'])
            test['soln'] = int(a)
        save_tests(os.path.join(TEST_DIR, OPTIMISTIC_UTILITY_TESTS_FILE), optimistic_utility_tests)

        print('Generating solutions for Q_LEARNING tests.')
        q_learning_tests = load_tests(os.path.join(TEST_DIR, Q_LEARNING_TESTS_FILE))
        for _, test in q_learning_tests.items():
            env = self._get_env_from_identifier(test['env'])
            Q_init = np.zeros((env.n_states, env.n_actions))
            Q_final, rewards = rl_soln.active_q_learning(env, Q_init, test['n_episodes'], test['action_selection'],
                                          test['discount_factor'], test['alpha'], test['epsilon'], test['T'],
                                          test['N_e'], test['R_plus'])
            test['soln'] = {'Q_final': Q_final.tolist(), 'episode_rewards': rewards.tolist()}
        save_tests(os.path.join(TEST_DIR, Q_LEARNING_TESTS_FILE), q_learning_tests)

        print('Generating solutions for SARSA tests.')
        sarsa_tests = load_tests(os.path.join(TEST_DIR, SARSA_TESTS_FILE))
        for _, test in sarsa_tests.items():
            env = self._get_env_from_identifier(test['env'])
            Q_init = np.zeros((env.n_states, env.n_actions))
            Q_final, rewards = rl_soln.active_sarsa(env, Q_init, test['n_episodes'], test['action_selection'],
                                          test['discount_factor'], test['alpha'], test['epsilon'], test['T'],
                                          test['N_e'], test['R_plus'])
            test['soln'] = {'Q_final': Q_final.tolist(), 'episode_rewards': rewards.tolist()}
        save_tests(os.path.join(TEST_DIR, SARSA_TESTS_FILE), sarsa_tests)


    @staticmethod
    def _run_test(func_name, module, func_in, func_exout, timeout, print_results=False):

        def wrapper_thread(method, q, args):
            try:
                ret = method(*args)
                q.put(ret)
            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except Exception as e:
                traceback.print_exc()
                q.put('Exception occurred')

        try:
            method_to_call = getattr(module, func_name)
        except Exception as e:
            print(repr(e))
            sys.exit(1)  # No method found

        que = queue.Queue()
        t = threading.Thread(target=wrapper_thread, args=(method_to_call, que, func_in))
        t.setDaemon(True)
        t.start()
        t.join(timeout)
        if t.is_alive():
            print('Test not finished!')
            sys.exit(2)  # Timeout
        ret = que.get()
        if isinstance(ret, str):
            if ret == 'Exception occurred':
                sys.exit(1)
        try:
            if func_name in ['active_q_learning', 'active_sarsa']:
                if compare_np_arrays(ret[0], func_exout[0]) and compare_np_arrays(ret[1], func_exout[1]):
                    sys.exit(0)
                else:
                    if print_results:
                        print('Result of test differs from solution.\nRESULT: {}\nSOLUTION: {}'.format(ret, func_exout))
            else:
                if ret == func_exout:
                    sys.exit(0)
                else:
                    if print_results:
                        print('Result of test differs from solution.\nRESULT: {}\nSOLUTION: {}'.format(ret, func_exout))
        except Exception as e:
            print(repr(e))
            sys.exit(1)
        sys.exit(3)

    def test_function(self, func_name, module, test_file_name, test_name, timeout):
        test_file_path = os.path.join(TEST_DIR, test_file_name)
        tests = load_tests(test_file_path)
        test = tests[test_name]
        if func_name in ['active_q_learning', 'active_sarsa']:
            np.random.seed(WUMPUS_ENV_SEED)
            env = self._get_env_from_identifier(test['env'])
            Q_init = np.zeros((env.n_states, env.n_actions))
            func_in = (env, Q_init, test['n_episodes'], test['action_selection'], test['discount_factor'],
                       test['alpha'], test['epsilon'], test['T'], test['N_e'], test['R_plus'])
            func_exout = (np.array(test['soln']['Q_final']), np.array(test['soln']['episode_rewards']))
        elif func_name in ['select_action_epsilon_greedy', 'select_action_softmax', 'select_action_optimistically']:
            np.random.seed(test['seed'])
            if func_name == 'select_action_epsilon_greedy':
                func_in = (test['s'], np.array(test['Q']), test['epsilon'])
            elif func_name == 'select_action_softmax':
                func_in = (test['s'], np.array(test['Q']), test['T'])
            else:
                func_in = (test['s'], np.array(test['Q']), np.array(test['N_sa']), test['N_e'], test['R_plus'])
            func_exout = test['soln']
        else:
            raise NotImplementedError
        self._run_test(func_name, module, func_in, func_exout, timeout, print_results=True)


def parse_command_line(arg):
    if "test_select_action_epsilon_greedy" in arg:
        return "select_action_epsilon_greedy", rl, EPSILON_GREEDY_TESTS_FILE, arg, ACTION_SELECTION_TIMEOUT
    elif "test_select_action_softmax" in arg:
        return "select_action_softmax", rl, SOFTMAX_TESTS_FILE, arg, ACTION_SELECTION_TIMEOUT
    elif "test_select_action_optimistically" in arg:
        return "select_action_optimistically", rl, OPTIMISTIC_UTILITY_TESTS_FILE, arg, ACTION_SELECTION_TIMEOUT
    elif "test_Q_learning" in arg:
        return "active_q_learning", rl, Q_LEARNING_TESTS_FILE, arg, Q_LEARNING_TIMEOUT
    elif "test_sarsa" in arg:
        return "active_sarsa", rl, SARSA_TESTS_FILE, arg, SARSA_TIMEOUT
    else:
        raise NotImplementedError


if __name__ == '__main__':
    t = Test(gen_soln=False)

    try:
        import rl
    except Exception as e:
        print(repr(e))
        sys.exit(1)

    t.test_function(*parse_command_line(sys.argv[1]))