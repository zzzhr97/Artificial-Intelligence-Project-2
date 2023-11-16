import numpy as np
from tqdm import tqdm
from ai import ai, State, INF_VALUE
from eval import evaluate_func

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
        
        print("Get init drops...")
        drop_value = [-1, -1, INF_VALUE[self.init_state.color]]
        drops = self.get_init_top_drops()
        print("Get init drops: Done.")

        with tqdm(total=len(drops), desc="Thinking...") as pbar:
            for drop in drops:
                if drop_value[0] == -1:
                    drop_value[:2] = drop[:2]

                new_state = self.init_state.next(drop)
                new_value = self._minimax(new_state, INF_VALUE[-1], INF_VALUE[1])
                if self.init_state.color == -1: # black
                    if new_value > drop_value[-1]:
                        drop_value = [drop[0], drop[1], new_value]
                else:   # white
                    if new_value < drop_value[-1]:
                        drop_value = [drop[0], drop[1], new_value]

                pbar.update(1)
            pbar.set_description('Done.')

        return drop_value[:2]
    
    def get_init_top_drops(self):
        """Get the top drops in given init_state through evaluation function"""
        top_drops = self.init_state.legal_drops()
        sorted_top_drops = sorted(top_drops, 
                                key=lambda x: evaluate_func(self.init_state.next(x), 
                                                            drops=[x],
                                                            last_evaluate=self.init_evaluate,
                                                            mode=self.args.mode), 
                                reverse=True)
        # debug
        print(top_drops)
        return sorted_top_drops[:self.args.init_n] 
    
    def get_top_drops(self, state):
        """Get the top drops in given state through evaluation function"""
        top_drops = state.legal_drops()
        sorted_top_drops = sorted(top_drops, 
                                key=lambda x: evaluate_func(state.next(x), 
                                                            drops=state.new_drops[1:]+[x],
                                                            last_evaluate=self.init_evaluate,
                                                            mode=self.args.mode), 
                                reverse=True)
        return sorted_top_drops[:self.args.n]

    def _minimax(self, state, a, b):
        """Get the next best drop-value"""
        # the last drop leads to winning, not this drop
        if self.is_win(state.board, state.new_drops[-1]):
            return INF_VALUE[state.color]
        
        if state.depth == 0:
            return evaluate_func(state, 
                                drops=state.new_drops[1:],
                                last_evaluate=self.init_evaluate,
                                mode=self.args.mode)
        
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