import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from Net import Net as Net
from network import SimpleNet, SimpleResNet

class PolicyValueNet():
    """Policy Value Network"""
    def __init__(self, load_path=None, device='cuda', internal_model='Res', weight_decay=1e-4, k=0):
        self.device = device
        self.weight_decay = weight_decay
        self.k = k

        # Network
        if internal_model == 'Res':
            self._internal_model = SimpleResNet(4)
            print("Use ResNet.")
        elif internal_model == 'Simple':
            self._internal_model = SimpleNet(4)
            print("Use SimpleNet.")
        else:
            raise ValueError(f"Unknown internal model: {internal_model}")
        
        self.policy_value_net = Net(self._internal_model)

        # GPU/CPU
        if self.device == 'cuda':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")
        print(f'Device: {self.device}')

        # Optimizer
        self.optim = optim.Adam(self.policy_value_net.parameters(), weight_decay=self.weight_decay)

        # load model from file
        if load_path:
            try:
                print(f'Loading model from {load_path}...')
                params = torch.load(load_path)
                self.policy_value_net.load_state_dict(params)
            except Exception as e:
                print(f"Error: {e}")
                print("Loading model failed, use new model instead.")
            else:
                print("Loading model successfully.")
        else:
            print("Using new model.")
                
        # Move model to GPU/CPU
        self.policy_value_net.to(self.device)

    # 训练时，通过概率的对数进行损失函数的计算，有助于提高数值计算的稳定性
    # 且cross entropy loss本身就是需要取对数的
    # 预测时，需要还原为概率
    def policy_value_batch(self, state_batch):
        """
        Param: 
        - state_batch: a batch of states.

        Return: 
        - a batch of move probabilities and state values.
        """
        state_batch = torch.Tensor(state_batch).to(self.device, torch.float)
        log_move_probs, value = self.policy_value_net(state_batch)

        # get move probabilities
        move_probs = torch.exp(log_move_probs).detach()
        value = value.view(-1).detach()
        return move_probs, value
    
    def policy_value_fn(self, board):
        """
        Param: 
        - board: A board

        Return: 
        - A list of (move, probability) tuples for each available
            move and the score of the board state
        - A value score in [-1, 1]
        """
        legal_positions = np.array(board.availables)

        # (4, 15, 15) --> (1, 4, 15, 15)
        current_state = np.array(board.get_state().reshape(-1, 4, 15, 15))
        current_state = torch.Tensor(current_state).to(self.device, torch.float)
        log_move_probs, value = self.policy_value_net(current_state)
        move_probs = torch.exp(log_move_probs).view(15, 15).detach().cpu().numpy()
        assert move_probs.shape == (15, 15), f"move_probs.shape: {move_probs.shape}"

        if len(legal_positions) > 0:
            move_probs = zip(legal_positions, move_probs[legal_positions[:, 0], legal_positions[:, 1]])
        else: 
            move_probs = None

        # extra
        move_probs = self.mask(board, move_probs)

        value = value.item()
        return move_probs, value
    
    def mask(self, board, move_probs):
        mask_matrix = np.full((15, 15), 1)
        for drop in board.drops:
            self._set_loc(mask_matrix, drop[0]+1, drop[1])
            self._set_loc(mask_matrix, drop[0]-1, drop[1])
            self._set_loc(mask_matrix, drop[0], drop[1]+1)
            self._set_loc(mask_matrix, drop[0], drop[1]-1)
            self._set_loc(mask_matrix, drop[0]+1, drop[1]+1)
            self._set_loc(mask_matrix, drop[0]-1, drop[1]-1)
            self._set_loc(mask_matrix, drop[0]+1, drop[1]-1)
            self._set_loc(mask_matrix, drop[0]-1, drop[1]+1)
        
        new_move_probs = []
        for move_prob in move_probs:
            new_move_prob = (move_prob[0], move_prob[1] * mask_matrix[move_prob[0][0], move_prob[0][1]])
            new_move_probs.append(new_move_prob)
        return new_move_probs
    
    def _set_loc(self, mask_matrix, x, y):
        """Set the adjacent locations of the given location"""
        if 0 <= x < 15 and 0 <= y < 15:
            mask_matrix[x, y] = 1
    
    def set_lr(self, optim, lr):
        """Sets the learning rate to the given value"""
        for param_group in optim.param_groups:
            param_group['lr'] = lr

    def train_once(self, state_batch, mcts_probs, winner_batch, lr):
        """perform a training step"""
        state_batch = torch.Tensor(state_batch).to(self.device, torch.float)
        mcts_probs = torch.Tensor(mcts_probs).to(self.device, torch.float)
        winner_batch = torch.Tensor(winner_batch).to(self.device, torch.float)

        # zero the parameter gradients
        self.optim.zero_grad()

        # set learning rate
        self.set_lr(self.optim, lr)

        # forward
        log_move_probs, value = self.policy_value_net(state_batch)

        assert log_move_probs.shape == mcts_probs.shape and log_move_probs.shape[1] == 15, \
            f"log_move_probs.shape: {log_move_probs.shape}, mcts_probs.shape: {mcts_probs.shape}"

        # loss = (z - v)^2 - pi^T * log(p) + c||theta||^2
        value_loss = F.mse_loss(value.view(-1), winner_batch)
        policy_loss = - torch.sum(torch.mul(mcts_probs, log_move_probs)) / mcts_probs.shape[0]
        loss = value_loss + policy_loss

        # get entropy
        entropy = - torch.sum(torch.mul(torch.exp(log_move_probs), log_move_probs)) / log_move_probs.shape[0]

        # backward and update parameters
        loss.backward()
        self.optim.step()
        return loss.item(), entropy.item()

    def save_model(self, save_path):
        """Save model."""
        print(f"Model saved to [ {save_path} ]... ", end='')
        try:
            torch.save(self.policy_value_net.state_dict(), save_path)
            print("Done.")
        except:
            print("Failed.")
