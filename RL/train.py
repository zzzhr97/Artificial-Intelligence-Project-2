import random
import torch
import numpy as np
import os
from collections import deque
from board import Board, Game
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphaZero import MCTSPlayer
from PVNet import PolicyValueNet 
from tqdm import tqdm
import time
import logging

save_path = './RL/models'
log_path = './RL/logs'

class TrainPipeline():
    def __init__(self, load_path=None):

        # params of the board and the game
        self.board_width = 15
        self.board_height = 15
        self.board = Board(width=self.board_width, height=self.board_height)
        self.game = Game(self.board)

        # training params
        self.learn_rate = 1e-3

        # adaptively adjust the learning rate based on KL
        self.lr_multiplier = 1.0  

        # the temperature param
        self.temp = 1.0  

        # number of simulations for each move
        self.n_playout = 400 
        self.c_puct = 5
        self.buffer_size = 10000
        self.batch_size = 512  
        self.data_buffer = deque(maxlen=self.buffer_size)

        # number of self play games per batch
        self.play_batch_size = 1   
        self.epochs = 5  
        self.kl_targ = 0.02

        # number of steps between two evaluation + update
        self.check_freq = 50
        self.game_batch_num = 1500

        # win ratio for the best model
        self.best_win_ratio = 0.0

        # num of simulations used for the pure MCTS
        self.pure_mcts_playout_num = 1000

        # policy value network
        self.policy_value_net = PolicyValueNet(load_path=load_path)
            
        self.mcts_player = MCTSPlayer(
            self.policy_value_net.policy_value_fn,
            c_puct=self.c_puct,
            n_playout=self.n_playout,
            is_selfplay=1)

    # flip + rotate to get more data
    def get_equi_data(self, play_data):
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
            for i in [1, 2, 3, 4]:

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

    def collect_selfplay_data(self, n_games=1):
        """collect self-play data for training"""
        for i in range(n_games):
            winner, play_data = self.game.start_self_play(
                self.mcts_player,
                temp=self.temp)
            
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)

            # augment the data
            play_data = self.get_equi_data(play_data)
            self.data_buffer.extend(play_data)

    def policy_update(self):
        """update the policy-value net"""
        # sample a batch of data from the buffer
        mini_batch = random.sample(self.data_buffer, self.batch_size)

        state_batch = np.array([data[0] for data in mini_batch])
        mcts_probs_batch = np.array([data[1] for data in mini_batch])
        winner_batch = [data[2] for data in mini_batch]

        # probs: (batch_size, 15, 15), value: (batch_size, 1)
        old_probs, old_value = self.policy_value_net.policy_value(state_batch)
        old_value = old_value.detach().flatten().cpu().numpy()

        for i in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step(
                    state_batch,
                    mcts_probs_batch,
                    winner_batch,
                    self.learn_rate * self.lr_multiplier)
            new_probs, new_value = self.policy_value_net.policy_value(state_batch)
            new_value = new_value.detach().flatten().cpu().numpy()

            # calculate the KL divergence between the old probs and the new probs
            kl = torch.nn.functional.kl_div(
                torch.log(new_probs + 1e-10), 
                old_probs, 
                reduction='batchmean').item()
            

            # early stopping if D_KL diverges badly
            if kl > self.kl_targ * 4:  
                break

        # adaptively adjust the learning rate
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        # explained variance score
        explained_var_old = (1 -
                            np.var(np.array(winner_batch) - old_value) /
                            np.var(np.array(winner_batch)))
        explained_var_new = (1 -
                            np.var(np.array(winner_batch) - new_value) /
                            np.var(np.array(winner_batch)))

        print((
            "\t"
            "[kl:{:.5f}] "
            "[lr_multiplier:{:.3f}] "
            "[loss:{:.5f}] "
            "[entropy:{:.5f}]"
            "[explained_var_old:{:.3f}] "
            "[explained_var_new:{:.3f}] "
            ).format(kl,
                self.lr_multiplier,
                loss,
                entropy,
                explained_var_old,
                explained_var_new))
        logging.info((
            "\t"
            "[kl:{:.5f}] "
            "[lr_multiplier:{:.3f}] "
            "[loss:{:.5f}] "
            "[entropy:{:.5f}]"
            "[explained_var_old:{:.3f}] "
            "[explained_var_new:{:.3f}] "
            ).format(kl,
                self.lr_multiplier,
                loss,
                entropy,
                explained_var_old,
                explained_var_new))
        return loss, entropy

    def policy_evaluate(self, n_games=10):
        """
        Evaluate the trained policy by playing against the pure MCTS player.
        Only to monitor.
        """
        current_mcts_player = MCTSPlayer(
            self.policy_value_net.policy_value_fn,
            c_puct=self.c_puct,
            n_playout=self.n_playout,
            is_selfplay=0)
        
        pure_mcts_player = MCTS_Pure(
            c_puct=5,
            n_playout=self.pure_mcts_playout_num)
        
        # begin playing
        win_cnt = {1: 0, -1: 0, 0: 0}
        for i in range(n_games):

            # -1, 1, -1, 1, ...
            start_player = (i % 2) * 2 - 1
            winner = self.game.start_play(
                current_mcts_player,
                pure_mcts_player,
                start_player=start_player,
                is_shown=0)
            win_cnt[winner] += 1

        win_ratio = 1.0 * (win_cnt[-1] + 0.5*win_cnt[0]) / n_games

        print("[n_playouts: {}] [win: {}] [lose: {}] [tie:{}]".format(
                self.pure_mcts_playout_num,
                win_cnt[-1], 
                win_cnt[1], 
                win_cnt[0]))
        logging.info("[n_playouts: {}] [win: {}] [lose: {}] [tie:{}]".format(
                self.pure_mcts_playout_num,
                win_cnt[-1], 
                win_cnt[1], 
                win_cnt[0]))
        return win_ratio

    def run(self):
        """run the training pipeline"""
        start_time = time.time()
        try:
            for i in range(self.game_batch_num):

                # collect self play data
                self.collect_selfplay_data(self.play_batch_size)
                print(f"\t[Batch {i+1:04d}] [Len {self.episode_len:04d}] [Time {time.time() - start_time:.2f}s]")
                logging.info(f"\t[Batch {i+1:04d}] [Len {self.episode_len:04d}] [Time {time.time() - start_time:.2f}s]")
                
                if len(self.data_buffer) > self.batch_size:
                    loss, entropy = self.policy_update()

                # evaluate model and save params
                if (i+1) % self.check_freq == 0:
                    print(f"[Batch {i+1:04d}]")
                    logging.info(f"[Batch {i+1:04d}]")
                    win_ratio = self.policy_evaluate()
                    self.policy_value_net.save_model(f'{save_path}/policy_epoch{i+1:04d}.pth')

                    if win_ratio > self.best_win_ratio:
                        print("[Best Policy]")
                        logging.info("[Best Policy]")
                        self.best_win_ratio = win_ratio
                        self.policy_value_net.save_model(f'{save_path}/policy_best_epoch{i+1:04d}.pth')

                        if (self.best_win_ratio == 1.0 and
                                self.pure_mcts_playout_num < 5000):
                            self.pure_mcts_playout_num += 1000
                            self.best_win_ratio = 0.0
                            
        except KeyboardInterrupt:
            print('\n\rQuit')


if __name__ == '__main__':
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    logging.basicConfig(filename=f'{log_path}/log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    training_pipeline = TrainPipeline()
    training_pipeline.run()