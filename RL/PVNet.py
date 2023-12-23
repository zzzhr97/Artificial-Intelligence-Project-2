import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from Net import Net as Net
from network import SimpleNet, SimpleResNet

def set_lr(optim, lr):
    """Sets the learning rate to the given value"""
    for param_group in optim.param_groups:
        param_group['lr'] = lr

class PolicyValueNet():
    """Policy Value Network"""
    def __init__(self, load_path=None, use_gpu=False):
        self.use_gpu = use_gpu

        # L2 penalty
        self.weight_decay = 1e-4  

        # Network
        self._internal_model = SimpleResNet(4)
        self.policy_value_net = Net(self._internal_model)

        # GPU/CPU
        if self.use_gpu:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")

        # Move model to GPU/CPU
        self.policy_value_net.to(self.device)

        # Optimizer
        self.optim = optim.Adam(self.policy_value_net.parameters(),
                                    weight_decay=self.weight_decay)

        # load model from file
        if load_path:
            params = torch.load(load_path)
            self.policy_value_net.load_state_dict(params)

    def policy_value(self, state_batch):
        """
        Param: 
        - state_batch: a batch of states.

        Return: 
        - a batch of action probabilities and state values.
        """
        # 训练时，通过概率的对数进行损失函数的计算，有助于提高数值计算的稳定性
        # 且cross entropy loss本身就是需要取对数的
        # 预测时，需要还原为概率

        state_batch = torch.Tensor(state_batch).to(self.device, torch.float)
        log_act_probs, value = self.policy_value_net(state_batch)

        # get action probabilities
        act_probs = torch.exp(log_act_probs).detach()
        value = value.view(-1).detach()
        return act_probs, value
    
    def policy_value_fn(self, board):
        """
        Param: 
        - board: A board

        Return: 
        - a list of (action, probability) tuples for each available
            action and the score of the board state
        """
        legal_positions = np.array(board.availables)

        # 将当前棋盘状态转换为神经网络的输入形式
        # (4, 15, 15) --> (1, 4, 15, 15)
        # 创建内存连续的numpy数组，这在一些情况下可以提高计算效率
        
        current_state = np.ascontiguousarray(board.current_state().reshape(-1, 4, 15, 15))
        
        # 获取当前棋盘状态的动作概率和状态价值
        current_state = torch.Tensor(current_state).to(self.device, torch.float)
        log_act_probs, value = self.policy_value_net(current_state)
        act_probs = torch.exp(log_act_probs).view(15, 15).detach().cpu().numpy()
        assert act_probs.shape == (15, 15), f"act_probs.shape: {act_probs.shape}"

        if len(legal_positions) > 0:
            act_probs = zip(legal_positions, 
                act_probs[legal_positions[:, 0], legal_positions[:, 1]])
        else: 
            act_probs = None

        value = value.item()
        return act_probs, value

    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        """perform a training step"""
        state_batch = torch.Tensor(state_batch).to(self.device, torch.float)
        mcts_probs = torch.Tensor(mcts_probs).to(self.device, torch.float)
        winner_batch = torch.Tensor(winner_batch).to(self.device, torch.float)

        # zero the parameter gradients
        self.optim.zero_grad()

        # set learning rate
        set_lr(self.optim, lr)

        # forward
        log_act_probs, value = self.policy_value_net(state_batch)

        assert log_act_probs.shape == mcts_probs.shape and log_act_probs.shape[1] == 15, \
            f"log_act_probs.shape: {log_act_probs.shape}, mcts_probs.shape: {mcts_probs.shape}"

        # define the loss = (z - v)^2 - pi^T * log(p) + c||theta||^2
        value_loss = F.mse_loss(value.view(-1), winner_batch)
        policy_loss = - torch.sum(torch.mul(mcts_probs, log_act_probs)) / mcts_probs.shape[0]
        loss = value_loss + policy_loss

        # get entropy
        entropy = - torch.sum(torch.mul(torch.exp(log_act_probs), log_act_probs)) / log_act_probs.shape[0]

        # backward and update parameters
        loss.backward()
        self.optim.step()
        return loss.item(), entropy.item()
    
    def get_params(self):
        params = self.policy_value_net.state_dict()
        return params

    def save_model(self, load_path):
        """ save model params to file """
        params = self.get_params()  
        torch.save(params, load_path)
