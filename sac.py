import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.layer1 = nn.Linear(state_dim, 400)
        self.layer2 = nn.Linear(400, 300)
        self.mean = nn.Linear(300, action_dim)
        self.log_std = nn.Linear(300, action_dim)
        self.max_action = max_action
        
    def forward(self, state):
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        mean = self.max_action * torch.tanh(self.mean(x))
        log_std = torch.clamp(self.log_std(x), -20, 2)
        return mean, log_std
    
    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x = normal.rsample()  # 使用重参数化技巧
        action = torch.tanh(x) * self.max_action
        log_prob = normal.log_prob(x)
        # 修正因为tanh变换导致的概率密度变化
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        return action, log_prob

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Q1
        self.layer1 = nn.Linear(state_dim + action_dim, 400)
        self.layer2 = nn.Linear(400, 300)
        self.layer3 = nn.Linear(300, 1)
        # Q2
        self.layer4 = nn.Linear(state_dim + action_dim, 400)
        self.layer5 = nn.Linear(400, 300)
        self.layer6 = nn.Linear(300, 1)
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        # Q1
        q1 = F.relu(self.layer1(x))
        q1 = F.relu(self.layer2(q1))
        q1 = self.layer3(q1)
        # Q2
        q2 = F.relu(self.layer4(x))
        q2 = F.relu(self.layer5(q2))
        q2 = self.layer6(q2)
        return q1, q2

class SAC:
    def __init__(self, state_dim, action_dim, max_action, device, 
                 lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2):
        self.device = device
        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
            action, _ = self.actor.sample(state)
            return action.cpu().numpy().flatten()
    
    def train(self, replay_buffer, batch_size=256):
        state, action, next_state, reward, done = replay_buffer.sample(batch_size)
        
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_state)
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            target_q = reward + (1 - done) * self.gamma * target_q
        
        # 更新Critic
        current_q1, current_q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # 更新Actor
        new_action, log_prob = self.actor.sample(state)
        q1, q2 = self.critic(state, new_action)
        q = torch.min(q1, q2)
        actor_loss = (self.alpha * log_prob - q).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # 软更新目标网络
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def save(self, filename):
        torch.save(self.actor.state_dict(), filename)
        
    def load(self, filename):
        self.actor.load_state_dict(torch.load(filename)) 