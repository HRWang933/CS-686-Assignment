# version 1.0

from operator import ne
from PIL.ImagePalette import random
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

    #### YOUR CODE HERE ####
    epsilon_np= np.array([epsilon,1-epsilon])
    flag= sample_integer_from_categorical_distribution(epsilon_np)
    if flag==0:
        a_len=len(Q[s])
        a_p=np.repeat(1/a_len,a_len)
        a= sample_integer_from_categorical_distribution(a_p)
    else:
        a=argmax_with_random_tiebreaking(Q[s,:])
    return a


def select_action_softmax(s: int, Q: np.array, T: float=10.0) -> int:
    '''
    Select an action via softmax selection using the Gibbs/Boltzmann distribution. Assumes that the temperature is
    always nonzero.
    :param s: The current state
    :param Q: A [n_states, n_actions] array where Q[s, a] is the action value for taking action a in state s
    :param T: The temperature (T != 0)
    '''

    #### YOUR CODE HERE ####
    # a_np= Q[s,:].copy()
    # a_np/=T
    # e = np.exp(a_np - np.sum(a_np))
    Qs=Q[s] - np.max(Q[s])
    e=np.exp(Qs / T) / np.sum(np.exp(Qs / T))
    a = argmax_with_random_tiebreaking(e)
    return a


def select_action_optimistically(s: int, Q: np.array, N_sa: np.array, N_e: int=5, R_plus: np.float=999.0) -> int:
    '''
    Use optimistic utility estimates to select an action. If the
    :param s: The current state
    :param Q: A [n_states, n_actions] array where Q[s, a] is the action value for taking action a in state s
    :param N_sa: A [n_states, n_actions] array indicating the number of times that each state/action pair has been visited
    :param N_e: Number of times a state-action pair is visited before expected utility is used instead of optimistic estimates
    :param R_plus: The best possible reward obtainable in any state
    '''

    #### YOUR CODE HERE ####
    a_np= Q[s,:].copy()
    for x in range(len(a_np)):
        if N_sa[s][x]<N_e:
            a_np[x]=R_plus
    a = argmax_with_random_tiebreaking(a_np)
    return a


def active_q_learning(env: WumpusWorld, Q_init: np.array, n_episodes: int, action_selection: str='optimistic',
                      discount_factor=0.99, alpha: float=0.5, epsilon: float=0.1, T: float=1., N_e: int=3,
                      R_plus: float=999.) -> (np.array, np.array):
    '''
    Conducts active Q-learning to learn optimal Q-values. Q-values are updated during each step for a fixed number of
    episodes.
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
    #### YOUR CODE HERE ####
    #Q = np.zeros((env.n_states, env.n_actions))
    #print(env._state, n_episodes, action_selection,discount_factor,alpha,epsilon,T,N_e,R_plus)
    Q=Q_init
    episode_rewards = np.zeros((n_episodes))


    N_sa = np.zeros((env.n_states, env.n_actions), dtype=int)
    R = np.zeros((env.n_states))
    for i in range(n_episodes):
        s=env.reset()
        terminal = False
        while not terminal:
            a = select_action(action_selection,s,Q,N_sa,epsilon,T,N_e,R_plus)
            next_state_id, reward, terminal, _=env.step(a)
            R[s] =reward
            episode_rewards[i]+=reward
            # next_a=argmax_with_random_tiebreaking(Q[next_state_id,:])
            # Q[s,a] += alpha*( R[s] + discount_factor*Q[next_state_id,next_a] -Q[s,a])
            Q[s, a] += alpha * (R[s] + discount_factor * np.max(Q[next_state_id]) - Q[s, a])
            N_sa[s,a]+=1
            s=next_state_id
    #print(episode_rewards)
    #print(N_sa)
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

    #### YOUR CODE HERE ####
    #Q = np.zeros((env.n_states, env.n_actions))

    

    episode_rewards = np.zeros((n_episodes))
    Q=Q_init
    #print(env._state, n_episodes, action_selection,discount_factor,alpha,epsilon,T,N_e,R_plus)

    N_sa = np.zeros((env.n_states, env.n_actions))
    R = np.zeros((env.n_states))
    for i in range(n_episodes):
        s=env.reset()
        a = select_action(action_selection,s,Q,N_sa,epsilon,T,N_e,R_plus)
        terminal = False
        while not terminal:
            next_state_id, reward, terminal, _=env.step(a)
            episode_rewards[i]+=reward
            R[s] =reward
            next_a=select_action(action_selection,next_state_id,Q,N_sa,epsilon,T,N_e,R_plus)
            Q[s,a]+=alpha*(R[s]+discount_factor*Q[next_state_id,next_a]-Q[s,a])
            N_sa[s,a]+=1
            s=next_state_id
            a=next_a
    return Q, episode_rewards