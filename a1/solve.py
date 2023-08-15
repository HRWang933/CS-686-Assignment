
from operator import index, truth
from typing import NewType
from board import *
import copy
import bisect
def a_star(init_board, hfn):
    """
    Run the A_star search algorithm given an initial board and a heuristic function.

    If the function finds a goal state, it returns a list of states representing
    the path from the initial state to the goal state in order and the cost of
    the solution found.
    Otherwise, it returns am empty list and -1.

    :param init_board: The initial starting board.
    :type init_board: Board
    :param hfn: The heuristic function.
    :type hfn: Heuristic
    :return: (the path to goal state, solution cost)
    :rtype: List[State], int
    """
    frontier = []
    explored = []
    expands_conter =0
    init_state = State(init_board,hfn,hfn(init_board),0)
    frontier.append(init_state)
    while frontier :
        # if frontier[0].parent != None:
        #     # frontier.sort(key=lambda x: x.f)
             
        #      #frontier.sort(key=lambda x: x.id)
        #      #frontier.sort(key=lambda x: x.parent.id)
        #      frontier.sort(key=lambda x:(x.f,x.id,x.parent.id))
             
        check_state=frontier.pop(0)
        if check_state.id not in explored:
            explored.append(check_state.id)
            if is_goal(check_state):
                print(expands_conter,end= ' ')
                return get_path(check_state),check_state.depth
            successors = get_successors(check_state)
            expands_conter+=1
            for next_state in successors:
                next_state.f= hfn(next_state.board)+next_state.depth
                #frontier.append(next_state) 
                bisect.insort_left(frontier, next_state)  
                i= frontier.index(next_state)
                llen = len(frontier)
                if i < llen-1:
                     while i < llen-1 and frontier[i].f== frontier[i+1].f and frontier[i].id > frontier[i+1].id:
                        frontier[i],frontier[i+1] = frontier[i+1],frontier[i]
                        i+=1  
                     while i < llen-1 and frontier[i].f== frontier[i+1].f and frontier[i].id == frontier[i+1].id and frontier[i].parent.id > frontier[i+1].parent.id:
                        frontier[i],frontier[i+1] = frontier[i+1],frontier[i]
                        i+=1

                 
    return [],-1
    raise NotImplementedError


def dfs(init_board):
    """
    Run the DFS algorithm given an initial board.

    If the function finds a goal state, it returns a list of states representing
    the path from the initial state to the goal state in order and the cost of
    the solution found.
    Otherwise, it returns am empty list and -1.

    :param init_board: The initial board.
    :type init_board: Board
    :return: (the path to goal state, solution cost)
    :rtype: List[State], int
    """
    frontier = []
    explored = []

    init_state = State(init_board,0,0,0)
    frontier.append(init_state)
    while frontier :
        check_state = frontier.pop()
        if check_state not in explored:
            explored.append(check_state)
            if is_goal(check_state):
                return get_path(check_state),check_state.depth
            successors=get_successors(check_state)
            successors.sort(key=lambda x: x.id, reverse=True)
            for next_state in successors:
                frontier.append(next_state)        
    return [],-1






    raise NotImplementedError


def get_successors(state):
    """
    Return a list containing the successor states of the given state.
    The states in the list may be in any arbitrary order.

    :param state: The current state.
    :type state: State
    :return: The list of successor states.
    :rtype: List[State]
    """
    list = []
    i = 0
    for car in state.board.cars:
        car_front = car.var_coord-1
        car_back = car.var_coord + car.length
        if car.orientation == 'h':
            while  car_front >= 0 :
                if state.board.grid[car.fix_coord][car_front]==".":
                    next = copy.deepcopy(state)
                    next.board.cars[i].set_coord(car_front)
                    next.board = Board(next.board.name,next.board.size,next.board.cars)
                    next.depth+=1
                    next.parent=state
                    next.id= hash(next.board)
                    #next.board.display()
                    list.append(next)
                    car_front-=1
                else:break
            while car_back <= state.board.size -1:
                if state.board.grid[car.fix_coord][car_back]==".":
                    next = copy.deepcopy(state)
                    next.board.cars[i].set_coord(car_back-car.length+1)
                    next.board = Board(next.board.name,next.board.size,next.board.cars)
                    next.depth+=1
                    next.parent=state
                    next.id= hash(next.board)
                    #next.board.display()
                    list.append(next)
                    car_back+=1
                else:break               
        else:
            while car_front >= 0 :
                if state.board.grid[car_front][car.fix_coord]==".":
                    next = copy.deepcopy(state)
                    next.board.cars[i].set_coord(car_front)
                    next.board = Board(next.board.name,next.board.size,next.board.cars)
                    next.depth+=1
                    next.parent=state
                    next.id= hash(next.board)
                    #next.board.display()
                    list.append(next)
                    car_front-=1
                else:break
            while car_back <= state.board.size-1 :
                if state.board.grid[car_back][car.fix_coord]==".":
                    next = copy.deepcopy(state)
                    next.board.cars[i].set_coord(car_back-car.length+1)
                    next.board = Board(next.board.name,next.board.size,next.board.cars)
                    next.depth+=1
                    next.parent=state
                    next.id= hash(next.board)
                    #next.board.display()
                    list.append(next)
                    car_back+=1   
                else:break
        i+=1
    return list
    raise NotImplementedError


def is_goal(state):
    """
    Returns True if the state is the goal state and False otherwise.

    :param state: the current state.
    :type state: State
    :return: True or False
    :rtype: bool
    """
    if state.board.cars[0].var_coord + state.board.cars[0].length == state.board.size:
        return True
    else: return False

    raise NotImplementedError


def get_path(state):
    """
    Return a list of states containing the nodes on the path 
    from the initial state to the given state in order.

    :param state: The current state.
    :type state: State
    :return: The path.
    :rtype: List[State]
    """
    list =[]
    list.append(state)
    #list[-1].board.display()
    #print(list[-1].id)
    while list[-1].parent != None:
        
        list.append(list[-1].parent)
        #list[-1].board.display()
        #print(list[-1].id)
    list.reverse()
    return list
          


    raise NotImplementedError


def blocking_heuristic(board):
    """
    Returns the heuristic value for the given board
    based on the Blocking Heuristic function.

    Blocking heuristic returns zero at any goal board,
    and returns one plus the number of cars directly
    blocking the goal car in all other states.

    :param board: The current board.
    :type board: Board
    :return: The heuristic value.
    :rtype: int
    """
    heuristic_value = 0
    #all goal car is horizantal
    potential_block =board.size -(board.cars[0].var_coord + board.cars[0].length)
    while potential_block!=0:
        # print(str(board.grid[board.cars[0].fix_coord][board.cars[0].var_coord+(board.cars[0].length-1)+potential_block]))
        # gird[y][x]
        if  board.grid[board.cars[0].fix_coord][board.cars[0].var_coord+(board.cars[0].length-1)+potential_block]!= ".":
            heuristic_value += 1
        potential_block -= 1
    if board.cars[0].var_coord + board.cars[0].length == board.size:
        return 0
    return heuristic_value +1       
            




    raise NotImplementedError


def advanced_heuristic(board):
    """
    An advanced heuristic of your own choosing and invention.

    :param board: The current board.
    :type board: Board
    :return: The heuristic value.
    :rtype: int
    """
    def can_car_move(y,x):
        head_block=False
        butt_block=False
        head_point=y
        butt_point=y
        while board.grid[head_point][x]!="^":
            head_point-=1
        if board.grid[head_point-1][x]!=".":
            head_block=True
        while board.grid[butt_point][x]!="v":
            butt_point+=1
        if board.grid[butt_point+1][x]!=".":
            butt_block=True
        if head_block and butt_block:
            return True
        else : return False



    heuristic_value = 0
    adv=0
    #all goal car is horizantal
    potential_block =board.size -(board.cars[0].var_coord + board.cars[0].length)
    while potential_block!=0:
        # print(str(board.grid[board.cars[0].fix_coord][board.cars[0].var_coord+(board.cars[0].length-1)+potential_block]))
        # gird[y][x]
        if  board.grid[board.cars[0].fix_coord][board.cars[0].var_coord+(board.cars[0].length-1)+potential_block]!= ".":
            if can_car_move(board.cars[0].fix_coord,board.cars[0].var_coord+(board.cars[0].length-1)+potential_block):
                adv=1
            heuristic_value += 1
        potential_block -= 1
    if board.cars[0].var_coord + board.cars[0].length == board.size:
        return 0
    return heuristic_value +adv +1       
            
    



    raise NotImplementedError

def test():
     boards = []
     boards = from_file("jams_posted.txt")


     #state1=State(boards[0],0,0,0)
     
     #print(str(is_goal(state1)))
     #dfs(boards[0])
    #  boards[0].display()
    #  print(advanced_heuristic(boards[0]))
    # for state in states[0]:
    #   state.board.display()
    #  for board in boards:
    #      board.display()
    #      print(blocking_heuristic(board))

     for board in boards:
        state=a_star(board,advanced_heuristic)
        print(board.name.strip(),end="\n")  

    #      #  board.display()
    #     #  print(advanced_heuristic(board))
            

    #     # print(state[0][-1].id)
    #     # state[0][-1].board.display()
    #      print(board.name.strip(),end="\n")
    #  boards[39].display()
    #states=a_star(boards[39],advanced_heuristic)
    #print(boards[39].name.strip(),end="\n")
    #  for state in states[0]:
    #      print(state.id,advanced_heuristic(state.board))
    #      state.board.display()
    #  print(states[1])     


test()