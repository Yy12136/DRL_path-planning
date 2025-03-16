import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.layer1 = nn.Linear(state_dim, 400)
        self.layer2 = nn.Linear(400, 300)
        self.layer3 = nn.Linear(300, action_dim * 11)  # 每个动作维度11个离散值
        self.action_dim = action_dim
        
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)  # 输出大小为 (batch_size, action_dim * 11)
        # 重塑为 (batch_size, action_dim, 11)
        return x.reshape(x.size(0), self.action_dim, 11)

class DQN:
    def __init__(self, state_dim, action_dim, max_action, device, 
                 lr=3e-4, gamma=0.99, tau=0.005):
        self.device = device
        self.action_dim = action_dim
        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        
        # Q networks
        self.q_net = QNetwork(state_dim, action_dim).to(device)
        self.target_q_net = copy.deepcopy(self.q_net)
        self.q_optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        
        # 为每个动作维度创建离散化空间
        self.action_space = np.linspace(-max_action, max_action, 11)
        
    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
            q_values = self.q_net(state)  # Shape: (1, action_dim, 11)
            action_indices = q_values.argmax(dim=2).cpu().numpy()  # Shape: (1, action_dim)
            
            # 将离散索引转换为连续动作值
            action = np.array([self.action_space[idx] for idx in action_indices[0]])
            return action
    
    def train(self, replay_buffer, batch_size=256):
        # 从经验回放中采样
        state, action, next_state, reward, done = replay_buffer.sample(batch_size)
        
        with torch.no_grad():
            # 计算目标Q值
            next_q_values = self.target_q_net(next_state)  # Shape: (batch_size, action_dim, 11)
            next_q_values = next_q_values.max(dim=2)[0]  # Shape: (batch_size, action_dim)
            next_q_values = next_q_values.mean(dim=1, keepdim=True)  # Shape: (batch_size, 1)
            target_q = reward + (1 - done) * self.gamma * next_q_values
        
        # 计算当前Q值
        current_q = self.q_net(state)  # Shape: (batch_size, action_dim, 11)
        
        # 将连续动作转换为离散动作索引
        action_indices = []
        for a in action.cpu().numpy():
            indices = []
            for i, act in enumerate(a):
                idx = np.abs(self.action_space - act).argmin()
                indices.append(idx)
            action_indices.append(indices)
        action_indices = torch.LongTensor(action_indices).to(self.device)
        
        # 收集每个动作维度的Q值
        q_values = []
        for i in range(self.action_dim):
            q_value = current_q[:, i, :].gather(1, action_indices[:, i:i+1])
            q_values.append(q_value)
        current_q = torch.mean(torch.cat(q_values, dim=1), dim=1, keepdim=True)
        
        # 计算损失并更新
        q_loss = nn.MSELoss()(current_q, target_q)
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()
        
        # 软更新目标网络
        for param, target_param in zip(self.q_net.parameters(), self.target_q_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
    def save(self, filename):
        torch.save(self.q_net.state_dict(), filename)
        
    def load(self, filename):
        self.q_net.load_state_dict(torch.load(filename))
        self.target_q_net = copy.deepcopy(self.q_net) 