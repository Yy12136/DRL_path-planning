import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Normal

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.layer1 = nn.Linear(state_dim, 400)
        self.layer2 = nn.Linear(400, 300)
        self.mean = nn.Linear(300, action_dim)
        self.log_std = nn.Linear(300, action_dim)
        self.max_action = max_action
        
    def forward(self, state):
        x = torch.relu(self.layer1(state))
        x = torch.relu(self.layer2(x))
        mean = self.max_action * torch.tanh(self.mean(x))
        log_std = torch.clamp(self.log_std(x), -20, 2)
        return mean, log_std

class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.layer1 = nn.Linear(state_dim, 400)
        self.layer2 = nn.Linear(400, 300)
        self.layer3 = nn.Linear(300, 1)
        
    def forward(self, state):
        x = torch.relu(self.layer1(state))
        x = torch.relu(self.layer2(x))
        return self.layer3(x)

class PPO:
    def __init__(self, state_dim, action_dim, max_action, device,
                 lr=3e-4, gamma=0.99, epsilon=0.2, epochs=10):
        self.device = device
        self.max_action = max_action
        self.gamma = gamma
        self.epsilon = epsilon
        self.epochs = epochs
        
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.critic = Critic(state_dim).to(device)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []
        
    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
            mean, log_std = self.actor(state)
            std = log_std.exp()
            
            normal = Normal(mean, std)
            action = normal.sample()
            log_prob = normal.log_prob(action)
            
            action = action.cpu().numpy().flatten()
            log_prob = log_prob.cpu().numpy().flatten()
            
            self.states.append(state)
            self.actions.append(action)
            self.log_probs.append(log_prob)
            
            return np.clip(action, -self.max_action, self.max_action)
    
    def train(self):
        # 将数据转换为tensor
        states = torch.cat(self.states)
        actions = torch.FloatTensor(self.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        
        # 计算优势函数
        with torch.no_grad():
            values = self.critic(states)
            advantages = torch.zeros_like(values)
            returns = torch.zeros_like(values)
            
            running_return = 0
            running_advantage = 0
            
            for t in reversed(range(len(self.rewards))):
                running_return = self.rewards[t] + self.gamma * running_return * (1 - self.dones[t])
                running_advantage = running_return - values[t].item()
                
                returns[t] = running_return
                advantages[t] = running_advantage
        
        # PPO更新
        for _ in range(self.epochs):
            # 计算新的动作概率
            mean, log_std = self.actor(states)
            std = log_std.exp()
            normal = Normal(mean, std)
            new_log_probs = normal.log_prob(actions)
            
            # 计算比率
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # 计算surrogate损失
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1-self.epsilon, 1+self.epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # 更新actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # 更新critic
            value_loss = nn.MSELoss()(self.critic(states), returns)
            self.critic_optimizer.zero_grad()
            value_loss.backward()
            self.critic_optimizer.step()
        
        # 清空缓存
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []
    
    def save(self, filename):
        torch.save(self.actor.state_dict(), filename)
        
    def load(self, filename):
        self.actor.load_state_dict(torch.load(filename)) 