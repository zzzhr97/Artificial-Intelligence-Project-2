import numpy as np
from ai import State, INF_VALUE

BLACK = -1
EMPTY = 0
WHITE = 1

def evaluate_func(state, **kwargs):
    # TODO: implement evaluate function
    mode = kwargs['mode']
    drops = kwargs['drops']
    last_evaluate = kwargs['last_evaluate']
    if mode=='simple':
        return simple_evaluate(state, BLACK)-simple_evaluate(state, WHITE)
    elif mode=='method1':
        return (method1_evaluate(state, BLACK, drops, last_evaluate)
                -method1_evaluate(state, WHITE, drops, last_evaluate))


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


def method1_evaluate(state, color, drops, last_evaluate):
    """Evaluate the state using method 1"""
    size = len(state.board)
    this_value = 0  # value about this drop after taking the drops
    last_value = 0  # value about this drop before taking the drops
    last_state_board = np.array(state.board)

    # If this is the first invoke, calculate the whole board
    if last_evaluate is None:
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
    
    # Otherwise, only need to calculate the part that's affected by this drop
    # evaluate rows and columns
    for drop in drops:
        last_state_board[drop[0], drop[1]] = EMPTY
    for drop in drops:
        # evaluate rows and columns
        this_row = state.board[drop[0]]
        this_col = state.board[:, drop[1]]
        last_row = last_state_board[drop[0]]
        last_col = last_state_board[:, drop[1]]
        this_value += method1_line_evaluate(this_row, color)
        this_value += method1_line_evaluate(this_col, color)    
        last_value += method1_line_evaluate(last_row, color)
        last_value += method1_line_evaluate(last_col, color)
        # evaluate diagonals
        this_diag = get_diag(state.board, drop)
        this_cont_diag = get_cont_diag(state.board, drop)
        last_diag = get_diag(last_state_board, drop)
        last_cont_diag = get_cont_diag(last_state_board, drop)
        if (this_diag is None or 
            this_cont_diag is None or 
            last_diag is None or 
            last_cont_diag is None):
            continue
        this_value += method1_line_evaluate(this_diag, color)
        this_value += method1_line_evaluate(this_cont_diag, color)
        last_value += method1_line_evaluate(last_diag, color)
        last_value += method1_line_evaluate(last_cont_diag, color)
    return last_evaluate-last_value+this_value


def get_diag(board, pos):
    """Get the diagonal line passing the position in a board"""
    size = len(board)
    if pos[0] > size-5 and pos[1] < 4:
        return None
    if pos[0] < 4 and pos[1] > size-5:
        return None
    offset = pos[1] - pos[0]
    return np.diag(board, k=offset)


def get_cont_diag(board, pos):
    """Get the diagonal line with different direction passing the position in a board"""
    size = len(board)
    if pos[0] < 4 and pos[1] < 4:
        return None
    if pos[0] > size-5 and pos[1] > size-5:
        return None
    if pos[0] >= pos[1]:
        offset = -(size-1-(pos[0]-pos[1]))
    else:
        offset = size-1-(pos[1]-pos[0])
    return np.diag(np.fliplr(board), k=offset)


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

    # very likely to win
    if consecutive == 4 and block_count == 0:
        if empty_inside:
            return 3000
        return 100000
    
    # likely to win
    if consecutive == 3 and block_count == 0:
        if empty_inside:
            return 1000
        return 10000
    
    # Common cases
    consecutive_score = (2, 5, 2000, 10000)    # Corresponding scores for 1, 2, 3, 4 consecutive chess
    block_count_score = (1, 0.6, 0.01)   # Number of blocks and correspoinding influence factor
    empty_score = (1, 0.8, 0.8, 1)    # How much influence empty can make for 1, 2, 3, 4 consecutive chess cases

    idx = consecutive-1
    value = consecutive_score[idx]
    value *= block_count_score[block_count]
    if empty_inside:
        value *= empty_score[idx]
    return int(value)