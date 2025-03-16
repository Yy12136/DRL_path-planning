import numpy as np
import torch
import gymnasium as gym
from underwater_robot_env import UnderwaterRobotEnv
from td3 import TD3
from ddpg import DDPG
from dqn import DQN
from ppo import PPO
from replay_buffer import ReplayBuffer
import matplotlib.pyplot as plt
import argparse
import os
import time
from sac import SAC

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", type=str, default="TD3", 
                      choices=["TD3", "DDPG", "DQN", "PPO", "SAC"],
                      help="选择训练算法: TD3, DDPG, DQN, PPO 或 SAC")
    parser.add_argument("--difficulty", type=str, default="medium",
                      choices=["easy", "medium", "hard"],
                      help="选择环境难度: easy, medium 或 hard")
    parser.add_argument("--max_episodes", type=int, default=1000,
                      help="最大训练回合数")
    parser.add_argument("--run_id", type=str, default="run1",
                      help="训练运行的ID，用于区分不同次训练")
    parser.add_argument("--save_dir", type=str, default=None,
                      help="模型保存路径，如果为None则自动生成")
    return parser.parse_args()

args = parse_args()

# 创建保存路径
if args.save_dir is None:
    args.save_dir = f"./result/runs/{args.run_id}/{args.algorithm}_{args.difficulty}"

# 创建必要的目录
os.makedirs(f"{args.save_dir}/models", exist_ok=True)
os.makedirs(f"{args.save_dir}/data", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = UnderwaterRobotEnv(difficulty=args.difficulty)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

# 根据选择的算法初始化策略
if args.algorithm == "TD3":
    policy = TD3(state_dim, action_dim, max_action, device)
elif args.algorithm == "DDPG":
    policy = DDPG(state_dim, action_dim, max_action, device)
elif args.algorithm == "DQN":
    policy = DQN(state_dim, action_dim, max_action, device)
elif args.algorithm == "PPO":
    policy = PPO(state_dim, action_dim, max_action, device)
elif args.algorithm == "SAC":
    policy = SAC(state_dim, action_dim, max_action, device)
else:
    raise ValueError(f"未知算法: {args.algorithm}")

print(f"使用 {args.algorithm} 算法进行训练...")
print("显示初始环境状态...")
env.render()
input("按回车键继续训练...")

# 初始化经验回放缓冲区（对于需要的算法）
if args.algorithm in ["TD3", "DDPG", "DQN"]:
    replay_buffer = ReplayBuffer(state_dim, action_dim, device=device)

# 训练参数
max_episodes = args.max_episodes
max_timesteps = 1000
episode_rewards = []
path_lengths = []
episode_times = []
start_time = time.time()

# 用于记录最后一轮的轨迹
final_trajectory = []

# 训练循环
for episode in range(max_episodes):
    episode_start_time = time.time()
    state, _ = env.reset()
    episode_reward = 0
    current_trajectory = [env.robot_pos.copy()]
    
    for t in range(max_timesteps):
        # 选择动作
        action = policy.select_action(np.array(state))
        if args.algorithm == "TD3":  # TD3使用动作噪声
            action = action + np.random.normal(0, max_action * 0.1, size=action_dim)
        action = np.clip(action, -max_action, max_action)
        
        # 执行动作
        next_state, reward, done, _, _ = env.step(action)
        episode_reward += reward
        
        # 记录轨迹点
        current_trajectory.append(env.robot_pos.copy())
        
        # 存储转换（对于使用经验回放的算法）
        if args.algorithm in ["TD3", "DDPG", "DQN"]:
            replay_buffer.add(state, action, next_state, reward, done)
        elif args.algorithm == "PPO":
            policy.rewards.append(reward)
            policy.next_states.append(next_state)
            policy.dones.append(done)
        
        # 训练智能体
        if args.algorithm in ["TD3", "DDPG", "DQN"]:
            if replay_buffer.size > 256:
                policy.train(replay_buffer)
        
        if done:
            break
            
        state = next_state
    
    # PPO在每个回合结束后更新
    if args.algorithm == "PPO":
        policy.train()
    
    # 计算路径长度
    trajectory = np.array(current_trajectory)
    path_length = np.sum(np.sqrt(np.sum(np.diff(trajectory, axis=0)**2, axis=1)))
    
    # 记录数据
    episode_rewards.append(episode_reward)
    path_lengths.append(path_length)
    episode_times.append(time.time() - episode_start_time)
    
    print(f"Episode {episode}: Reward = {episode_reward:.2f}, Path Length = {path_length:.2f}m, Time = {episode_times[-1]:.2f}s")
    
    # 保存最后一轮的轨迹
    if episode == max_episodes - 1:
        final_trajectory = current_trajectory

# 保存训练数据
np.save(f'{args.save_dir}/data/episode_rewards.npy', np.array(episode_rewards))
np.save(f'{args.save_dir}/data/path_lengths.npy', np.array(path_lengths))
np.save(f'{args.save_dir}/data/episode_times.npy', np.array(episode_times))
np.save(f'{args.save_dir}/data/final_trajectory.npy', trajectory)

# 保存环境配置
env_config = {
    'obstacles': env.obstacles,
    'goal_pos': env.goal_pos.tolist(),
    'bounds': [-2, 10, -2, 10, -2, 10],
    'difficulty': args.difficulty  # 添加难度信息
}
np.save(f'{args.save_dir}/data/env_config.npy', env_config)

# 保存模型
torch.save(policy.actor.state_dict(), f"{args.save_dir}/models/actor.pth")
if args.algorithm in ["TD3", "DDPG"]:
    torch.save(policy.critic.state_dict(), f"{args.save_dir}/models/critic.pth")
elif args.algorithm == "PPO":
    torch.save(policy.critic.state_dict(), f"{args.save_dir}/models/critic.pth")

# 保存训练配置和统计信息
training_stats = {
    'algorithm': args.algorithm,
    'difficulty': args.difficulty,
    'max_episodes': args.max_episodes,
    'run_id': args.run_id,
    'final_reward': episode_rewards[-1],
    'final_path_length': path_lengths[-1],
    'total_training_time': time.time() - start_time,
    'average_episode_time': np.mean(episode_times),
    'training_time': time.strftime("%Y-%m-%d %H:%M:%S")
}
np.save(f'{args.save_dir}/data/training_stats.npy', training_stats)

print(f"\n训练完成！")
print(f"模型保存在: {args.save_dir}/models/")
print(f"训练数据保存在: {args.save_dir}/data/")