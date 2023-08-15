from vea import *
from grid_env import GridEnvironment

def main():
    f1 = Factor(["A", "B"],
            [[0.4, 0.2, 0.5],
                [0.1, 0.9, 0.2]])
    f2 = Factor(["C", "B"],
        [[0.4, 0.2, 0.5],
            [0.1, 0.9, 0.2]])
    f3=multiply(f1,f2)
    f4=normalize(f3)
    print(f1.values.shape)
def main2():
    width = 2
    height = 2

    # Define which cell coordinates correspond to empty squares in which the robot can be
    empty_cell_coords = [[0,0],[0,1],[1,0],[1,1]]
    n_empty_cells = len(empty_cell_coords)

    # Set the robot's initial location
    init_cell = np.random.randint(n_empty_cells)

    # Set the robot's sensor error rate
    epsilon = 0.2

    # Initialize the environment.
    env = GridEnvironment(width, height, empty_cell_coords, init_cell, epsilon)
    print(env.obs_discrepancy(9,2))
    p = env.trans_probs     # Access the environment's transition matrices (we only have 1 matrix since we have 1 possible action)
    
    o = env.obs_probs       # Access the environment's observation probability matrix
    print(o)
main()
