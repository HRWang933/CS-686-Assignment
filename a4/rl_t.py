
import gym
import numpy as np

from utils import *
from wumpus_env import *


def select_action(strategy: str, s: int, Q: np.array, N_sa: np.array, epsilon: float=0.05, T: float=10., N_e: int=5,
                  R_plus: float=999.):
    '''
    Selects an action for the current state according to the desired strategy. This may be of use to you for the
    Q-learning and SARSA functions.
    :param strategy: Action selection strategy for exploration - one of {"optimistic", "softmax", "epsilon_greedy"}
    :param s: The current state
    :param Q: A [n_states, n_actions] array where Q[s, a] is the action value for taking action a in state s
    :param N_sa: A [n_states, n_actions] array indicating the number of times that each state/action pair has been visited
    :param epsilon: The probability for selecting a random action (in [0, 1])
    :param T: The temperature for softmax action selection
    :param N_e: Number of times a state-action pair is visited before expected utility is used instead of optimistic estimates
    :param R_plus: The best possible reward obtainable in any state
    '''

    if strategy == "optimistic":
        return select_action_optimistically(s, Q, N_sa, N_e, R_plus)
    elif strategy == "softmax":
        return select_action_softmax(s, Q, T)
    else:
        return select_action_epsilon_greedy(s, Q, epsilon)


def select_action_epsilon_greedy(s: int, Q: np.array, epsilon: float=0.1) -> int:
    '''
    With probability epsilon, select a random action. Otherwise, select the greedy action.
    :param s: The current state
    :param Q: A [n_states, n_actions] array where Q[s, a] is the action value for taking action a in state s
    :param epsilon: The probability for selecting a random action (in [0, 1])
    '''
    epsilon_P = np.array([epsilon, 1. - epsilon])
    if sample_integer_from_categorical_distribution(epsilon_P) == 0:
        n_actions = Q.shape[1]
        uniform_P = np.ones((n_actions)) / n_actions
        return sample_integer_from_categorical_distribution(uniform_P)
    else:
        return argmax_with_random_tiebreaking(Q[s])


def select_action_softmax(s: int, Q: np.array, T: float=10.0) -> int:
    '''
    Select an action via softmax selection using the Gibbs/Boltzmann distribution. Assumes that the temperature is
    always nonzero.
    :param s: The current state
    :param Q: A [n_states, n_actions] array where Q[s, a] is the action value for taking action a in state s
    :param T: The temperature (T != 0)
    '''

    Q_s_shifted = Q[s] - np.max(Q[s])
    P = np.exp(Q_s_shifted / T) / np.sum(np.exp(Q_s_shifted / T))
    return sample_integer_from_categorical_distribution(P)


def select_action_optimistically(s: int, Q: np.array, N_sa: np.array, N_e: int=5, R_plus: np.float=999.0) -> int:
    '''
    Use optimistic utility estimates to select an action. If the
    :param s: The current state
    :param Q: A [n_states, n_actions] array where Q[s, a] is the action value for taking action a in state s
    :param N_sa: A [n_states, n_actions] array indicating the number of times that each state/action pair has been visited
    :param N_e: Number of times a state-action pair is visited before expected utility is used instead of optimistic estimates
    :param R_plus: The best possible reward obtainable in any state
    '''

    f = np.full((Q.shape[1]), Q[s])
    f[N_sa[s] < N_e] = R_plus
    return argmax_with_random_tiebreaking(f)


def active_q_learning(env: WumpusWorld, Q_init: np.array, n_episodes: int, action_selection: str='optimistic',
                      discount_factor=0.99, alpha: float=0.5, epsilon: float=0.1, T: float=1., N_e: int=3,
                      R_plus: float=1000.) -> (np.array, np.array):
    '''
    Conducts active Q-learning to learn optimal Q-values. Q-values are updated during each step for a fixed number of
    episodes. Use "select_action()" to sample
    :param env: The environment with which the agent interacts
    :param Q_init [env.n_states, env.n_actions]: Initial action values
    :param n_episodes: The number of training episodes during which experience can be collected to learn the Q-values
    :param action_selection: Action selection strategy for exploration - one of {"optimistic", "softmax", "epsilon_greedy"}
    :param discount_factor: Discount factor, in (0, 1]
    :param alpha: Learning rate. alpha > 0.
    :param epsilon: The probability for selecting a random action (in [0, 1])
    :param T: The temperature for softmax action selection
    :param N_e: Number of times a state-action pair is visited before expected utility is used instead of optimistic estimates
    :param R_plus: The best possible reward obtainable in any state
    :return: (Final Q-values after convergence [env.n_states, env.n_actions], Rewards obtained in each episode [n_episodes])
    '''

    # Initialize variables
    n_states = Q_init.shape[0]                                        # Number of possible states
    n_actions = Q_init.shape[1]                                       # Number of possible actions
    Q = Q_init                                                        # Current action values
    R = np.zeros((n_states))                                          # Observed reward function
    N_sa = np.zeros((n_states, n_actions), dtype=int)                 # Number of times state-action pair is observed
    episode_rewards = np.zeros((n_episodes))                          # Total reward per episode

    for e in range(n_episodes):

        s_curr = env.reset()
        terminal = False
        R_total = 0.

        # Conduct the Q-learning loop
        while not terminal:

            # Select action
            a = select_action(action_selection, s_curr, Q, N_sa, epsilon, T, N_e, R_plus)

            # Apply action and observe next state and reward
            s_next, reward, terminal, s_arr = env.step(a)

            # Update reward function
            R[s_curr] = reward
            R_total += reward

            # Update Q value
            Q[s_curr, a] += alpha * (R[s_curr] + discount_factor * np.max(Q[s_next]) - Q[s_curr, a])

            # Update counts
            N_sa[s_curr, a] += 1

            # Update the current state
            s_curr = s_next

        episode_rewards[e] = R_total
    return Q, episode_rewards


def active_sarsa(env: WumpusWorld, Q_init: np.array, n_episodes: int, action_selection: str='optimistic',
                 discount_factor: float=0.99, alpha: float=0.5, epsilon: float=0.1, T: float=1., N_e: int=3,
                 R_plus: float=999.) -> (np.array, np.array):
    '''
    Conducts active SARSA to learn optimal Q-values. Q-values are updated during each step for a fixed number of
    episodes.
    :param env: The environment with which the agent interacts
    :param Q_init: Initial action values
    :param n_episodes: The number of training episodes during which experience can be collected to learn the Q-values
    :param action_selection: Action selection strategy for exploration - one of {"optimistic", "softmax", "epsilon_greedy"}
    :param discount_factor: Discount factor, in (0, 1]
    :param alpha: Learning rate. alpha > 0.
    :param epsilon: The probability for selecting a random action (in [0, 1])
    :param T: The temperature for softmax action selection
    :param N_e: Number of times a state-action pair is visited before expected utility is used instead of optimistic estimates
    :param R_plus: The best possible reward obtainable in any state
    :return: (Final Q-values after convergence [env.n_states, env.n_actions], Rewards obtained in each episode [n_episodes])
    '''

    # Initialize variables
    n_states = Q_init.shape[0]                                        # Number of possible states
    n_actions = Q_init.shape[1]                                       # Number of possible actions
    Q = Q_init                                                        # Current action values
    R = np.zeros((n_states))                                          # Observed reward function
    N_sa = np.zeros((n_states, n_actions), dtype=np.int)              # Number of times state-action pair is observed
    episode_rewards = np.zeros((n_episodes))                          # Total reward per episode

    for e in range(n_episodes):

        s_curr = env.reset()
        a_curr = select_action(action_selection, s_curr, Q, N_sa, epsilon, T, N_e, R_plus)
        terminal = False
        R_total = 0.

        while not terminal:

            # Apply action and observe next state and reward
            s_next, r, terminal, _ = env.step(a_curr)

            # Update reward function
            R[s_curr] = r
            R_total += r

            a_next = select_action(action_selection, s_next, Q, N_sa, epsilon, T, N_e, R_plus)

            # Update Q value
            Q[s_curr, a_curr] += alpha * (R[s_curr] + discount_factor * Q[s_next, a_next] - Q[s_curr, a_curr])

            # Update counts
            N_sa[s_curr, a_curr] += 1

            # Update the current state and action
            s_curr = s_next
            a_curr = a_next

        episode_rewards[e] = R_total
    return Q, episode_rewards