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
current_score = (0.7, 1)    # current_color == color --> 1.0, current_color != color --> 0.6

def evaluate_func(state, **kwargs):
    mode = kwargs['mode']

    global drops, last_evaluate, init_state
    drops = kwargs['drops']
    last_evaluate = kwargs['last_evaluate']
    init_state = kwargs['init_state']

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
    return method1_init(state, color)

def method1_init(state, color):
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

def method1_recalculate(state, color):
    """Evaluate the state using method 1"""
    value = last_evaluate
    row_idxes, col_idxes, diag_idxes, cont_diag_idxes = set(), set(), set(), set()

    # calculate the index of rows, columns, diagonals and cont diagonals
    for drop in drops:
        i, j = drop[0], drop[1]
        row_idxes.add(i)
        col_idxes.add(j)
        diag_idxes.add(j-i)
        cont_diag_idxes.add(15-(j+i))

    # recalculate row values
    for row_idx in row_idxes:
        init_row = init_state.board[row_idx]
        row = state.board[row_idx]
        value -= method1_line_evaluate(init_row, color)
        value += method1_line_evaluate(row, color)

    # recalculate column values
    for col_idx in col_idxes:
        init_col = init_state.board[:, col_idx]
        col = state.board[:, col_idx]
        value -= method1_line_evaluate(init_col, color)
        value += method1_line_evaluate(col, color)

    # recalculate diagonal values
    for diag_idx in diag_idxes:
        init_diag = np.diag(init_state.board, k=diag_idx)
        diag = np.diag(state.board, k=diag_idx)
        value -= method1_line_evaluate(init_diag, color)
        value += method1_line_evaluate(diag, color)
    
    # recalculate cont diagonal values
    for cont_diag_idx in cont_diag_idxes:
        init_cont_diag = np.diag(np.fliplr(init_state.board), k=cont_diag_idx)
        cont_diag = np.diag(np.fliplr(state.board), k=cont_diag_idx)
        value -= method1_line_evaluate(init_cont_diag, color)
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