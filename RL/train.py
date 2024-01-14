import random
import torch
import numpy as np
import os
from collections import deque
from board import Board, Gomoku
from player import RandomPlayer, AlphaZeroPlayer
from PVNet import PolicyValueNet 
import time
import logging
import configparser

config_file = 'RL/config.ini'

def read_config():
    """Read [config.ini]"""
    config = configparser.ConfigParser()
    config.read(config_file)
    return config['parameters']

class RandomBuffer(object):
    """Random queue."""
    def __init__(self, max_size):
        self.max_size = max_size
        self.queue = deque(maxlen=self.max_size)

    def extend(self, data_list):
        """Extend data to the queue."""
        self.queue.extend(data_list)

    def random_sample(self, batch_size):
        """Sample data from the queue."""
        return random.sample(self.queue, batch_size)

class RL_training():
    def __init__(self):

        # params of the board and the game
        self.board = Board()
        self.game = Gomoku(self.board)

        # get args
        args = read_config()
        self.set_params(args)

        # initialize file and directory
        self.init_file()

        # data buffer
        self.buffer_size = 16000
        self.data_buffer = RandomBuffer(max_size=self.buffer_size)

        # adaptively adjust the learning rate based on KL
        self.lr_multiplier = 1.0  

        # win ratio for the best model
        self.best_win_ratio = 0.0

        # policy value network
        self.policy_value_net = PolicyValueNet(
            load_path=self.load_path, 
            device=self.device, 
            internal_model=self.model_name,
            k=self.k)
        
        # AlphaZero player
        self.alphazero_player = AlphaZeroPlayer(
            self.policy_value_net.policy_value_fn,
            c_puct=self.c_puct,
            n_play=self.n_play,
            dirichlet_alpha=self.dirichlet_alpha,
            is_selfplay=1)
        
    def set_params(self, args):
        """Set parameters."""
        self.start_batch = int(args['start_batch'])
        self.model_name = args['model_name']
        self.load_path = args['load_path']
        self.save_path = args['save_path']
        self.save_name = args['save_name']
        self.log_path = args['log_path']
        self.log_name = args['log_name']

        self.device = args['device']
        self.lr = float(args['lr'])
        self.batch_size = int(args['batch_size'])
        self.n_epoch = int(args['n_epoch'])
        self.n_batch = int(args['n_batch'])
        self.n_game = int(args['n_game'])
        self.random_n_play = int(args['random_n_play'])

        self.n_play = int(args['n_play'])
        self.c_puct = float(args['c_puct'])
        self.temperature = float(args['temperature'])
        self.n_game_per_batch = int(args['n_game_per_batch'])
        self.kl_targ = float(args['kl_targ'])
        self.dirichlet_alpha = float(args['dirichlet_alpha'])
        self.k = float(args['k'])

        self.eval_every = int(args['eval_every'])

    def init_file(self):
        """Initialize file and directory."""
        os.makedirs(self.save_path, exist_ok=True)
        os.makedirs(self.log_path, exist_ok=True)
        logging.basicConfig(filename=f'{self.log_path}/{self.model_name}_{self.log_name}', 
            level=logging.INFO, 
            format='%(asctime)s - %(levelname)s - %(message)s')
        logging.info("Model will be saved to "
            f"[ {self.save_path}/policy_{self.model_name}_{self.save_name}_(best_)epochxxxx.pth ]")
        logging.info("Log will be saved to "
            f"[ {self.log_path}/{self.model_name}_{self.log_name} ]")
        print("Model will be saved to "
            f"[ {self.save_path}/policy_{self.model_name}_{self.save_name}_(best_)epochxxxx.pth ]")
        print("Log will be saved to "
            f"[ {self.log_path}/{self.model_name}_{self.log_name} ]")
        
    def print_result(self, play_result):
        print(("[n_play: {}] ",
            "[win: {}] "
            "[lose: {}] "
            "[tie: {}]"
            ).format(
                self.random_n_play,
                play_result[-1], 
                play_result[1], 
                play_result[0]))
        logging.info(("[n_play: {}] "
            "[win: {}] "
            "[lose: {}] "
            "[tie: {}]"
            ).format(
                self.random_n_play,
                play_result[-1], 
                play_result[1], 
                play_result[0]))
        
    def print_info(self, kl, loss, entropy):
        print((
            "\t"
            "[kl:{:.5f}] "
            "[lr_multiplier:{:.3f}] "
            "[loss:{:.5f}] "
            "[entropy:{:.5f}] "
            ).format(kl,
                self.lr_multiplier,
                loss,
                entropy))
        logging.info((
            "\t"
            "[kl:{:.5f}] "
            "[lr_multiplier:{:.3f}] "
            "[loss:{:.5f}] "
            "[entropy:{:.5f}] "
            ).format(kl,
                self.lr_multiplier,
                loss,
                entropy))

    def get_augment_data(self, play_data):
        """
        Augment the data set by rotation and flipping
        Param:
        - play_data: [(state, MCTS_pr, winner_cur), ..., ...]
            - state: 4 * width * height.
            - MCTS_pr: move probabilities from MCTS (15 * 15 numpy array).
            - winner_cur: winners from the perspective of the current player.
        """
        extend_data = []
        for state, mcts_porb, winner in play_data:
            for i in [1, 2, 3]:

                # rotate counterclockwise
                equi_state = np.array([np.rot90(layer, i) for layer in state])
                equi_mcts_prob = np.rot90(mcts_porb)
                extend_data.append((equi_state, equi_mcts_prob, winner))
                
                # flip horizontally
                equi_state = np.array([np.fliplr(layer) for layer in equi_state])
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append((equi_state, equi_mcts_prob, winner))

                # flip vertically
                equi_state = np.array([np.flipud(layer) for layer in equi_state])
                equi_mcts_prob = np.flipud(equi_mcts_prob)
                extend_data.append((equi_state, equi_mcts_prob, winner))

        return extend_data

    def get_selfplay_data(self, n_game=1):
        """collect self-play data for training"""
        for i in range(n_game):
            winner, play_data = self.game.self_play(
                self.alphazero_player,
                temperature=self.temperature)
            
            play_data = list(play_data)[:]
            self.n_move = len(play_data)

            # augment the data
            play_data = self.get_augment_data(play_data)
            self.data_buffer.extend(play_data)

    def get_batch_data(self):
        """Get a batch of data."""
        batch_data = self.data_buffer.random_sample(self.batch_size)
        state_batch = np.array([data[0] for data in batch_data])
        mcts_probs_batch = np.array([data[1] for data in batch_data])
        winner_batch = [data[2] for data in batch_data]
        return state_batch, mcts_probs_batch, winner_batch

    def change_lr(self, kl):
        """Given kl, change the learning rate."""
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

    def policy_improve(self):
        """Improve policy."""
        state_batch, mcts_probs_batch, winner_batch = self.get_batch_data()

        # get old probs and values
        # probs: (batch_size, 15, 15), value: (batch_size, 1)
        old_probs, old_value = self.policy_value_net.policy_value_batch(state_batch)
        old_value = old_value.detach().flatten().cpu().numpy()

        for i in range(self.n_epoch):
            loss, entropy = self.policy_value_net.train_once(
                    state_batch,
                    mcts_probs_batch,
                    winner_batch,
                    self.lr * self.lr_multiplier)
            new_probs, new_value = self.policy_value_net.policy_value_batch(state_batch)
            new_value = new_value.detach().flatten().cpu().numpy()

            # calculate kl divergence
            kl = torch.nn.functional.kl_div(
                torch.log(new_probs + 1e-10), 
                old_probs, 
                reduction='batchmean').item()

            if kl > self.kl_targ * 4:  
                break

        self.change_lr(kl)
        self.print_info(kl, loss, entropy)
        return loss, entropy

    def policy_evaluate(self):
        """Let current AlphaZeroPlayer play against RandomPlayer to evaluate."""
        alphazero_player = AlphaZeroPlayer(
            self.policy_value_net.policy_value_fn,
            c_puct=self.c_puct,
            n_play=self.n_play,
            is_selfplay=0)
        
        random_player = RandomPlayer(
            c_puct=5,
            n_play=self.random_n_play)
        
        play_result = {1: 0, -1: 0, 0: 0}
        for i in range(self.n_game):

            # -1, 1, -1, 1, ...
            start_player = (i % 2) * 2 - 1
            winner = self.game.play_game(
                alphazero_player,
                random_player,
                start_player=start_player,
                is_shown=0)
            play_result[winner] += 1

        win_ratio = 1.0 * (play_result[-1] + 0.5*play_result[0]) / self.n_game
        self.print_result(play_result)
        return win_ratio

    def train(self):
        start_time = time.time()
        try:
            for i in range(self.start_batch, self.n_batch):
                print(f"\t[Batch {i+1:04d}]\r", end='')

                self.get_selfplay_data(self.n_game_per_batch)

                print(f"\t[Batch {i+1:04d}] [Move {self.n_move:04d}] [Time {time.time() - start_time:.2f}s]")
                logging.info(f"\t[Batch {i+1:04d}] [Move {self.n_move:04d}] [Time {time.time() - start_time:.2f}s]")
                
                if len(self.data_buffer) > 4 * self.batch_size:
                    _, _ = self.policy_improve()

                if (i+1) % self.eval_every == 0:
                    self.evaluate(i)
                            
        except KeyboardInterrupt:
            print('\n\rStop training.')

    def evaluate(self, i):
        """Evaluate by playing."""
        print(f"[Batch {i+1:04d}]")
        logging.info(f"[Batch {i+1:04d}]")

        win_ratio = self.policy_evaluate()
        save_name = f'{self.save_path}/policy_{self.model_name}_{self.save_name}_epoch{i+1:04d}.pth'
        self.policy_value_net.save_model(save_name)

        if win_ratio > self.best_win_ratio:
            print("[Best Policy]")
            logging.info("[Best Policy]")
            self.best_win_ratio = win_ratio
            save_name = f'{self.save_path}/policy_{self.model_name}_{self.save_name}_best_epoch{i+1:04d}.pth'
            self.policy_value_net.save_model(save_name)

            if (self.best_win_ratio == 1.0 and self.random_n_play < 8000):
                self.random_n_play += 1000
                self.best_win_ratio = 0.0


if __name__ == '__main__':
    rl_training = RL_training()
    rl_training.train()