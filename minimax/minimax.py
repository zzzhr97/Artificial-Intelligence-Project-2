import numpy as np
from tqdm import tqdm
from minimax.ai import ai, State, INF_VALUE
from minimax.eval import evaluate_func

class minimax(ai): 

    def __init__(self, args, chessboard, robot_color, last_drop):
        super(minimax, self).__init__()
        self.init_state = State(chessboard, robot_color, last_drop, args.depth, None, None, args.shuffle)
        self.args = args
        self.init_evaluate = evaluate_func(self.init_state, drops=None, last_evaluate=None, mode=self.args.mode)

    def get_best_drop(self):
        """Get the best drop in given init_state"""
        if np.sum(abs(self.init_state.board)) == 0:
            return self.get_first_drop()
        if np.sum(abs(self.init_state.board)) == 1:
            return self.get_second_drop(self.init_state.board)
        
        print("Get init drops...")
        drop_value = [-1, -1, INF_VALUE[self.init_state.color]]
        drops = self.get_init_top_drops()
        print("Get init drops: Done.")

        with tqdm(total=len(drops), desc="Thinking...") as pbar:
            # debug
            drop_values = []
            for drop in drops:
                if drop_value[0] == -1:
                    drop_value[:2] = drop[:2]

                new_state = self.init_state.next(drop)
                new_value = self._minimax(new_state, INF_VALUE[-1], INF_VALUE[1])
                # debug
                drop_values.append([drop[0], drop[1], new_value])
                if self.init_state.color == -1: # black
                    if new_value > drop_value[-1]:
                        drop_value = [drop[0], drop[1], new_value]
                else:   # white
                    if new_value < drop_value[-1]:
                        drop_value = [drop[0], drop[1], new_value]

                pbar.update(1)
            pbar.set_description('Done.')

        # debug
        print("Drops: ", drop_values)
        if self.init_state.color == -1:
            print("Best drop: ", drop_values[np.argmax(np.array(drop_values)[:, 2])])
        else:
            print("Best drop: ", drop_values[np.argmin(np.array(drop_values)[:, 2])])

        return drop_value[:2]
    
    def get_init_top_drops(self):
        """Get the top drops in given init_state through evaluation function"""
        top_drops = self.init_state.legal_drops()
        drop_value = []
        for drop in top_drops:
            value = evaluate_func(self.init_state.next(drop), 
                                drops=[drop],
                                last_evaluate=self.init_evaluate,
                                mode=self.args.mode)
            drop_value.append([drop[0], drop[1], value])
        sorted_drop_value = sorted(drop_value, key=lambda x: x[-1], reverse=self.init_state.color == -1)
        sorted_drops = [[x[0], x[1]] for x in sorted_drop_value[:self.args.init_n]]
        return sorted_drops
    
    def get_top_drops(self, state):
        """Get the top drops in given state through evaluation function"""
        top_drops = state.legal_drops()
        if not self.args.eval_legal_drops:
            return top_drops[:self.args.n]

        drop_value = []
        for drop in top_drops:
            value = evaluate_func(state.next(drop), 
                                drops=state.new_drops[1:]+[drop],
                                last_evaluate=self.init_evaluate,
                                mode=self.args.mode)
            drop_value.append([drop[0], drop[1], value])
        sorted_drop_value = sorted(drop_value, key=lambda x: x[-1], reverse=state.color == -1)
        sorted_drops = [[x[0], x[1]] for x in sorted_drop_value[:self.args.n]]
        return sorted_drops

    def _minimax(self, state, a, b):
        """Get the next best drop-value"""   
        if state.depth == 0 or self.is_win(state.board, state.new_drops[-1]):
            value = evaluate_func(state, 
                                drops=state.new_drops[1:],
                                last_evaluate=self.init_evaluate,
                                mode=self.args.mode)
            # debug
            #if self.is_win(state.board, state.new_drops[-1]):
                #print(f"{-state.color} win: {state.new_drops}, {value}")
            return value
        
        value = INF_VALUE[state.color]
        
        if state.color == -1:   # black
            for drop in self.get_top_drops(state):
                new_state = state.next(drop)
                value = max(value, self._minimax(new_state, a, b))
                if value >= b:
                    return value
                a = max(a, value)

        else:   # white
            for drop in self.get_top_drops(state):
                new_state = state.next(drop)
                value = min(value, self._minimax(new_state, a, b))
                if value <= a:
                    return value
                b = min(b, value)

        return value


def get_drop(args, chessboard, robot_color, last_drop):
    """Get a drop from ai-minimax"""
    robot = minimax(args, chessboard, robot_color, last_drop)
    return robot.get_best_drop()