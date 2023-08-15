import numpy as np
from rl_t import *
from utils import *
from wumpus_env import *
import matplotlib.pyplot as plt
def run():
    n=50000
    rewards= np.zeros((n))
    for i in [1,2,3]:
        #WumpusWorld.seed(i)
        envv=WumpusWorld()
        #envv.seed(i)

        #_,reward=active_sarsa(env=envv,T=0.1,action_selection="softmax",n_episodes=n,Q_init=np.zeros((envv.n_states, envv.n_actions)))
        #_,reward=active_q_learning(env=envv,T=0.1,action_selection="softmax",n_episodes=n,Q_init=np.zeros((envv.n_states, envv.n_actions)))
        #_,reward=active_sarsa(env=envv,N_e=2,R_plus=999,action_selection="optimistic",n_episodes=n,Q_init=np.zeros((envv.n_states, envv.n_actions)))
        _,reward=active_q_learning(env=envv,N_e=2,R_plus=999,action_selection="optimistic",n_episodes=n,Q_init=np.zeros((envv.n_states, envv.n_actions)))
        rewards += reward
    rewards/=3
    N=50
    rewards_mov_avg = np.convolve(rewards, np.ones(N)/N, mode='valid')
    np.save("rewards_mov_avg",rewards_mov_avg)
    plt.plot(rewards_mov_avg)
    #plt.title("Total rewards per episode with SARSA and softmax exploration")
    #plt.title("Total rewards per episode with Q-learning and softmax exploration")
    #plt.title("Total rewards per episode with SARSA and optimistic utility estimates")
    plt.title("Total rewards per episode with Q-learning and optimistic utility estimates")
    plt.xlabel("Episode number")         
    plt.ylabel("Total undiscounted reward") 
    plt.show()

if __name__ == '__main__':
    run()