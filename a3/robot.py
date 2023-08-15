import numpy as np
from factor import Factor
from typing import List
from grid_env import GridEnvironment
from vea import ve

if __name__ == '__main__':

    '''
    The code below demonstrates how to create a GridEnvironment object that represents the robot localization problem.
    '''

    # Specify width and height, in cells
    width = 16
    height = 4

    # Define which cell coordinates correspond to empty squares in which the robot can be
    empty_cell_coords = [[0,0],[0,1],[0,2],[0,3],[0,5],[0,6],[0,7],[0,8],[0,9],[0,11],[0,12],[0,13],[0,15],
                   [1,2],[1,3],[1,5],[1,8],[1,10],[1,12],
                   [2,1],[2,2],[2,3],[2,5],[2,8],[2,9],[2,10],[2,11],[2,12],[2,15],
                   [3,0],[3,1],[3,3],[3,4],[3,5],[3,7],[3,8],[3,9],[3,10],[3,12],[3,13],[3,14],[3,15]]
    n_empty_cells = len(empty_cell_coords)

    # Set the robot's initial location
    init_cell = np.random.randint(n_empty_cells)

    # Set the robot's sensor error rate
    epsilon = 0.10

    # Initialize the environment.
    env = GridEnvironment(width, height, empty_cell_coords, init_cell, epsilon)

    p = env.trans_probs     # Access the environment's transition matrices (we only have 1 matrix since we have 1 possible action)
    o = env.obs_probs       # Access the environment's observation probability matrix

    # Visualize the robot's current location
    #env.visualize_state()

    # # Visualize the robot's localization belief (a probability distribution over all states)
    # belief = np.random.random((n_empty_cells,))     # Get some random numbers in [0,1]
    # belief = (belief / np.sum(belief)).tolist()     # Normalize the belief to make it a valid probability distribution


    n_observation = 50
    observation = np.random.randint(16, size=n_observation)

    
    #env.visualize_belief(belief)
def generate_observation(number:int)->List[int]:
    np.random.seed()
    observations=[]
    cell=np.random.randint(n_empty_cells)
    for i in range(number):
        observation= np.random.choice(np.arange(0, env.n_observations), p=env.obs_probs[cell])
        observations.append(observation)
        #print(cell,observation)
        cell = np.random.choice(np.arange(0, env.n_states), p=env.trans_probs[0][cell])
        
    return observations

def print_max(belief):
        max_index = np.argmax(belief)
        print("("+str(env.empty_cell_coords[max_index][0])+","+str(env.empty_cell_coords[max_index][1])+") "+str(belief[max_index]))

def probability_at_least(observation:List[int],probability:float):
    belief = o[:,observation[0]]
    belief = (belief / np.sum(belief)).tolist()
    print_max(belief)
    for i in range(1,len(observation)):
        factor_p=Factor(["move","state","newstate"],p)
        factor_o=Factor(["newstate","observation"],o)
        factor_s=Factor(["state"],belief)
        evidence = {"move":0,"observation":observation[i]}
        factor_n=ve([factor_p,factor_o,factor_s],["newstate"],evidence,["move","state"])
        belief=factor_n.values
        #env.visualize_belief(belief)
        print_max(belief)
        for x in belief:
            if x >=probability: 
                print("minimum number of time steps "+str(i))
                return i
avg=[]
for i in range(5000):
    observations=generate_observation(50)
    avg.append(probability_at_least(observations,0.8))
print(np.average(np.array(avg)))




    