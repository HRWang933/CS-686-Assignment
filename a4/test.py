import numpy as np
from rl import *
from utils import *
from wumpus_env import *

def main():
    s=0
    Q=np.array([[-3.6,-9.2,-5.6,2.6,6.5,-0.6,3.3,5.,8.,-3.1],
                 [ 8.8,-0.9,-7.1,2.,-3.,-6.5,1.1,-8.,7.7,-2.8],
                 [ 6.,-8.7,4.6,-4.8,-1.9,-7.8,-4.2,1.6,-0.8,7.7]])
    # N_sa=np.array([[2,3,7,7,6,7,2,9,6,8],
    #                 [4,1,2,3,0,2,6,5,8,7],
    #                 [2,7,7,5,6,9,3,2,3,8]])
    # N_e=2
    # R_plus=10

    a_len=len(Q[s])
    a_p=np.repeat(1/a_len,a_len)

    # x=rl.select_action_optimistically(s,Q,N_sa,N_e,R_plus)
    # print(x)
    env=WumpusWorld(heading=HEADING_EAST)
    env.seed(0)
    id=np.array([1,1,1,0,0,0,0,0])
    print(active_q_learning(env,np.zeros((env.n_states, env.n_actions)),10,"epsilon_greedy"))


main()