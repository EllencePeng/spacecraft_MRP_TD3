import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np

class PolicyNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super(PolicyNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

        self.action_bound = action_bound  # action_bound是环境可以接受的动作最大值

    def forward(self, x):
        return torch.tanh(self.net(x)) * self.action_bound

class QValueNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QValueNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # #使用正态分布初始化权重(实验表明效果很差)
        # for layer in self.net:
        #     if isinstance(layer, nn.Linear):
        #         layer.weight.data.normal_(0,0.01)
        #         layer.bias.data.fill_(0)

    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1)   # 拼接状态和动作
        return self.net(cat)

class TD3:
    ''' TD3算法 '''
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound, sigma, actor_lr, critic_lr, tau, gamma, device, delay):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim, action_bound).to(device)
        self.critic_1 = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic_2 = QValueNet(state_dim, hidden_dim, action_dim).to(device)

        self.target_actor = PolicyNet(state_dim, hidden_dim, action_dim, action_bound).to(device)
        self.target_critic_1 = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic_2 = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        
        # 初始化目标网络,设置和原网络相同的参数
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        # 将target网络的参数都设置为不需要计算梯度
        for param in self.target_actor.parameters():
            param.requires_grad = False
        for param in self.target_critic_1.parameters():
            param.requires_grad = False
        for param in self.target_critic_2.parameters():
            param.requires_grad = False
        # 定义优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.sigma = sigma  # 高斯噪声的标准差,均值直接设为0
        self.tau = tau  # 目标网络软更新参数
        self.action_dim = action_dim
        self.device = device
        self.delay = delay

    def take_action(self, state):
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        action = np.array(self.actor(state).tolist())                       #把输出的tensor转换为三维ndarray
        #action = self.actor(state).item() （原版代码，适用于输出是一维的情况）
        # 给动作添加噪声，增加探索
        action = np.clip(action + self.sigma * np.random.randn(self.action_dim), -1, 1)     #生成服从标准正态分布的随机数，这里是生成三维分别加到action上
         
        return action   #输出的是三维ndarray
    
    def take_action_test(self, state):
        #删掉噪声
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        action = np.clip(np.array(self.actor(state).tolist()), -1, 1)                       #把输出的tensor转换为三维ndarray
        return action   #输出的是三维ndarray

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)   # 软更新过程，tau为1的时候表示把参数完全复制到目标网络中

    def update(self, transition_dict, env_i):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.float).to(self.device)
        #actions = torch.tensor(transition_dict['actions'], dtype=torch.float).view(-1, 1).to(self.device) （原版代码，适用于输出是一维的情况）
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device) 
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        next_actions = self.target_actor(next_states)
        next_q_values_1 = self.target_critic_1(next_states, next_actions)
        next_q_values_2 = self.target_critic_2(next_states, next_actions)
        next_q_values = torch.min(next_q_values_1,next_q_values_2)

        q_targets = rewards + self.gamma * next_q_values

        critic_1_loss = torch.mean(F.mse_loss(self.critic_1(states, actions), q_targets)) #这里，前者是Critic网络的输出，后者是要把输出往上靠的目标值，两个target网络仅用于计算后者的值。
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()

        critic_2_loss = torch.mean(F.mse_loss(self.critic_2(states, actions), q_targets)) #这里，前者是Critic网络的输出，后者是要把输出往上靠的目标值，两个target网络仅用于计算后者的值。
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        if env_i % self.delay == 0:
            actor_loss = -torch.mean(self.critic_1(states, self.actor(states))) #目标是让动作在C网络的评价中得分更高
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.soft_update(self.actor, self.target_actor)  # 软更新策略网络
            self.soft_update(self.critic_1, self.target_critic_1)  # 软更新价值网络
            self.soft_update(self.critic_2, self.target_critic_2)  # 软更新价值网络

    def save_model(self, path):
        # 保存四个网络的权重
        torch.save(self.actor.state_dict(), path + '/actor.pth')
        torch.save(self.critic_1.state_dict(), path + '/critic_1.pth')
        torch.save(self.critic_2.state_dict(), path + '/critic_2.pth')
        torch.save(self.target_actor.state_dict(), path + '/target_actor.pth')
        torch.save(self.target_critic_1.state_dict(), path + '/target_critic_1.pth')
        torch.save(self.target_critic_2.state_dict(), path + '/target_critic_2.pth')

    def load_model(self, path):
        # 加载四个网络的权重
        self.actor.load_state_dict(torch.load(path + '/actor.pth'))
        self.critic_1.load_state_dict(torch.load(path + '/critic_1.pth'))
        self.critic_2.load_state_dict(torch.load(path + '/critic_2.pth'))
        self.target_actor.load_state_dict(torch.load(path + '/target_actor.pth'))
        self.target_critic_1.load_state_dict(torch.load(path + '/target_critic_1.pth'))
        self.target_critic_2.load_state_dict(torch.load(path + '/target_critic_2.pth'))