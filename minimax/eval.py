import numpy as np
from minimax.ai import State, INF_VALUE

BLACK = -1
EMPTY = 0
WHITE = 1

# Scores for different situations in eval funciton
consecutive_score = (2, 5, 1000, 10000)    # Corresponding scores for 1, 2, 3, 4 consecutive chess
block_count_score = (1, 0.6, 0.2)   # Number of blocks and correspoinding influence factor
empty_score = (1, 0.6, 0.8, 0.9)    # How much influence empty can make for 1, 2, 3, 4 consecutive chess cases

# Weights for current color and opponent color
current_score = (0.6, 1)    # current_color == color --> 1.0, current_color != color --> 0.6

def evaluate_func(state, **kwargs):
    mode = kwargs['mode']
    drops = kwargs['drops']
    last_evaluate = kwargs['last_evaluate']

    global current_color
    current_color = state.color

    if mode=='simple':
        return simple_evaluate(state, BLACK)-simple_evaluate(state, WHITE)
    elif mode=='method1':
        return method1_evaluate(state, BLACK)-method1_evaluate(state, WHITE)


def simple_evaluate(state, color):
    """Evaluate the state using simple method"""
    size = len(state.board)
    value = 0

    # evaluate rows and columns
    for i in range(size):
        row = state.board[i]
        col = state.board[:, i]
        value += simple_line_evaluate(row, color)
        value += simple_line_evaluate(col, color)
        
    # evaluate diagonals
    for i in range(-(size-5), size-4):
        diag = np.diag(state.board, k=i)
        cont_diag = np.diag(np.fliplr(state.board), k=i)
        value += simple_line_evaluate(diag, color)
        value += simple_line_evaluate(cont_diag, color)

    return value


def simple_line_evaluate(line, color):
    """Evaluate a line using simple method"""
    # single chess: 1
    # double chess: 10
    # triple chess: 100
    # quadruple chess: 1000
    # quintuple chess: 10000
    consecutive = 0
    value = 0
    for i in range(len(line)):
        if line[i] == color:
            consecutive += 1
        else:
            if consecutive > 0:
                value += 10 ** (consecutive - 1)
            consecutive = 0
    return value


def method1_evaluate(state, color):
    """Evaluate the state using method 1"""
    size = len(state.board)
    value = 0

    # evaluate rows and columns
    for i in range(size):
        row = state.board[i]
        col = state.board[:, i]
        value += method1_line_evaluate(row, color)
        value += method1_line_evaluate(col, color)
        
    # evaluate diagonals
    for i in range(-(size-5), size-4):
        diag = np.diag(state.board, k=i)
        cont_diag = np.diag(np.fliplr(state.board), k=i)
        value += method1_line_evaluate(diag, color)
        value += method1_line_evaluate(cont_diag, color)

    return value


def method1_line_evaluate(line, color):
    """Evaluate a line using method 1"""
    size = len(line)
    value = 0
    consecutive = 0
    block_count = 2
    empty_inside = False

    for i in range(size):
        cell_color = line[i]
        # Same color
        if cell_color==color:
            consecutive += 1
        # Empty cell
        elif cell_color==EMPTY:
            if consecutive==0:
                block_count = 1
            else:
                if not empty_inside and i+1<size and line[i+1]==color:
                    empty_inside = True
                else:
                    value += calculate_score(consecutive, block_count-1, empty_inside)
                    consecutive = 0
                    block_count = 1
                    empty_inside = False
        # Different color
        else:
            if consecutive>0:
                value += calculate_score(consecutive, block_count, empty_inside)
                consecutive = 0
            block_count = 2
    
    if consecutive>0:
        value += calculate_score(consecutive, block_count, empty_inside)

    # modify weight according to current color
    value *= current_score[current_color == color]
    
    return value


def calculate_score(consecutive, block_count, empty_inside):
    """Calculate the score of a small part of chess, 
    given the number of consecutive chess, number of blocks and whether there is an empty cell inside"""
    # Circustance that's impossible to win
    if block_count==2 and consecutive<5:
        return 0
    
    # Long consective chess, very likely to win
    if consecutive>=5:
        if empty_inside:
            return 10000
        return 100000
    
    idx = consecutive-1
    value = consecutive_score[idx]
    value *= block_count_score[block_count]
    if empty_inside:
        value *= empty_score[idx]
    return int(value)