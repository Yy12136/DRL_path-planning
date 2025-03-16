import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from underwater_robot_env import UnderwaterRobotEnv
from td3 import TD3
import torch
import argparse
import os

def plot_sphere(ax, center, radius, color='b', alpha=0.3):
    """绘制球体"""
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color=color, alpha=alpha)

def plot_cylinder(ax, center, radius, height, color='r', alpha=0.3):
    """绘制圆柱体"""
    u = np.linspace(0, 2 * np.pi, 100)
    h = np.linspace(0, height, 100)
    x = center[0] + radius * np.cos(u)[:, np.newaxis] * np.ones_like(h)
    y = center[1] + radius * np.sin(u)[:, np.newaxis] * np.ones_like(h)
    z = center[2] + np.ones_like(u)[:, np.newaxis] * h
    ax.plot_surface(x, y, z, color=color, alpha=alpha)

def plot_velocity_vector(ax, pos, vel, color='cyan'):
    """绘制速度向量"""
    if np.linalg.norm(vel) > 0:
        ax.quiver(pos[0], pos[1], pos[2],
                 vel[0], vel[1], vel[2],
                 color=color, alpha=0.6,
                 length=np.linalg.norm(vel))

def create_policy(algorithm, state_dim, action_dim, max_action, device):
    """根据算法名称创建对应的策略"""
    if algorithm == "TD3":
        from td3 import TD3
        return TD3(state_dim, action_dim, max_action, device)
    elif algorithm == "DDPG":
        from ddpg import DDPG
        return DDPG(state_dim, action_dim, max_action, device)
    elif algorithm == "DQN":
        from dqn import DQN
        return DQN(state_dim, action_dim, max_action, device)
    elif algorithm == "PPO":
        from ppo import PPO
        return PPO(state_dim, action_dim, max_action, device)
    elif algorithm == "SAC":
        from sac import SAC
        return SAC(state_dim, action_dim, max_action, device)
    else:
        raise ValueError(f"不支持的算法: {algorithm}")

def visualize_trajectory(run_id, algorithm, difficulty='medium'):
    """可视化轨迹"""
    print("开始可视化轨迹...")
    
    # 设置路径
    base_dir = f"./result/runs/{run_id}/{algorithm}_{difficulty}"
    model_path = f"{base_dir}/models/actor.pth"
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"错误：模型文件不存在: {model_path}")
        return
    
    print("初始化环境和策略...")
    try:
        # 初始化环境和策略
        env = UnderwaterRobotEnv(difficulty=difficulty)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 创建对应算法的策略
        print(f"加载 {algorithm} 模型...")
        policy = create_policy(algorithm, state_dim, action_dim, max_action, device)
        policy.actor.load_state_dict(torch.load(model_path))
        
        # 创建可视化输出目录
        vis_dir = f"{base_dir}/visualizations"
        os.makedirs(vis_dir, exist_ok=True)
        
        print("开始生成轨迹...")
        # 运行一个回合并记录轨迹
        state, _ = env.reset()
        trajectory = [env.robot_pos.copy()]
        done = False
        step_count = 0
        max_steps = 1000  # 设置最大步数，防止无限循环
        
        while not done and step_count < max_steps:
            action = policy.select_action(np.array(state))
            state, _, done, _, _ = env.step(action)
            trajectory.append(env.robot_pos.copy())
            step_count += 1
            if step_count % 100 == 0:
                print(f"已执行 {step_count} 步...")
        
        trajectory = np.array(trajectory)
        print(f"轨迹生成完成，共 {step_count} 步")
        
        print("生成动画...")
        # 创建动画
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
    
        def update(frame):
            ax.cla()
            
            # 设置水下环境的背景色
            ax.set_facecolor((0.5, 0.8, 0.9, 0.3))
            
            # 添加水平面
            x_surface = np.linspace(-2, 10, 50)
            y_surface = np.linspace(-2, 10, 50)
            X, Y = np.meshgrid(x_surface, y_surface)
            Z = np.full_like(X, 10)  # 水面高度
            ax.plot_surface(X, Y, Z, alpha=0.1, color='cyan')
            
            # 添加水下粒子效果
            n_particles = 50
            x_particles = np.random.uniform(-2, 10, n_particles)
            y_particles = np.random.uniform(-2, 10, n_particles)
            z_particles = np.random.uniform(-2, 10, n_particles)
            ax.scatter(x_particles, y_particles, z_particles, color='white', alpha=0.2, s=10)
            
            # 设置坐标轴范围
            ax.set_xlim([-2, 10])
            ax.set_ylim([-2, 10])
            ax.set_zlim([-2, 10])
            
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Depth (m)')
            
            # 绘制障碍物
            for sphere in env.obstacles['spheres']:
                plot_sphere(ax, sphere['center'], sphere['radius'], color='gray', alpha=0.4)
            
            # 绘制起点和终点
            ax.scatter([0], [0], [0], color='lime', s=100, label='Start')
            ax.scatter([env.goal_pos[0]], [env.goal_pos[1]], [env.goal_pos[2]], 
                      color='magenta', s=100, label='Goal')
            
            # 绘制轨迹
            ax.plot(trajectory[:frame+1, 0], trajectory[:frame+1, 1], 
                    trajectory[:frame+1, 2], color='yellow', label='AUV Trajectory')
            
            # 绘制当前位置
            if frame < len(trajectory):
                ax.scatter([trajectory[frame, 0]], [trajectory[frame, 1]], 
                          [trajectory[frame, 2]], color='yellow', s=100)
            
            # 添加深度刻度
            z_ticks = ax.get_zticks()
            ax.set_zticklabels([f'{-z:.1f}' for z in z_ticks])
            
            ax.set_title('Underwater AUV Obstacle Avoidance Trajectory')
            ax.legend()
        
        ani = animation.FuncAnimation(fig, update, frames=len(trajectory), 
                                    interval=50, repeat=False)
            
        print(f"保存动画到 {vis_dir}/trajectory.gif ...")
        ani.save(f'{vis_dir}/trajectory.gif', writer='pillow')
        plt.close()
        print("轨迹可视化完成！")
        
    except Exception as e:
        print(f"可视化过程出错: {str(e)}")
        import traceback
        traceback.print_exc()

def load_data(run_id, algorithm, difficulty='medium'):
    """加载训练数据和环境配置"""
    base_dir = f"./result/runs/{run_id}/{algorithm}_{difficulty}/data"
    
    try:
        trajectory = np.load(f'{base_dir}/final_trajectory.npy')
        rewards = np.load(f'{base_dir}/episode_rewards.npy')
        path_lengths = np.load(f'{base_dir}/path_lengths.npy')
        episode_times = np.load(f'{base_dir}/episode_times.npy')
        env_config = np.load(f'{base_dir}/env_config.npy', allow_pickle=True).item()
        training_stats = np.load(f'{base_dir}/training_stats.npy', allow_pickle=True).item()
        
        return trajectory, rewards, path_lengths, episode_times, env_config, training_stats
    except Exception as e:
        print(f"加载数据时出错: {str(e)}")
        return None, None, None, None, None, None

def plot_rewards(rewards, run_id, algorithm):
    """绘制奖励曲线"""
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, label=algorithm)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(f'Training Rewards - {algorithm}')
    plt.grid(True)
    plt.legend()
    
    # 使用正确的保存路径
    save_dir = f'./result/runs/{run_id}/{algorithm}/visualizations'
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f'{save_dir}/training_rewards.png')
    plt.close()

def plot_3d_trajectory(trajectory, env_config):
    """绘制3D轨迹图"""
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # 设置水下环境的背景色
    ax.set_facecolor((0.5, 0.8, 0.9, 0.3))  # 淡蓝色半透明
    fig.patch.set_facecolor((0.5, 0.8, 0.9, 0.2))
    
    # 添加水平面
    bounds = env_config['bounds']
    x_surface = np.linspace(bounds[0], bounds[1], 50)
    y_surface = np.linspace(bounds[2], bounds[3], 50)
    X, Y = np.meshgrid(x_surface, y_surface)
    Z = np.full_like(X, bounds[5])  # 水面高度
    ax.plot_surface(X, Y, Z, alpha=0.1, color='cyan')
    
    # 添加水下粒子效果
    n_particles = 100
    x_particles = np.random.uniform(bounds[0], bounds[1], n_particles)
    y_particles = np.random.uniform(bounds[2], bounds[3], n_particles)
    z_particles = np.random.uniform(bounds[4], bounds[5], n_particles)
    ax.scatter(x_particles, y_particles, z_particles, color='white', alpha=0.2, s=10)
    
    # 设置坐标轴范围
    ax.set_xlim([bounds[0], bounds[1]])
    ax.set_ylim([bounds[2], bounds[3]])
    ax.set_zlim([bounds[4], bounds[5]])
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Depth (m)')
    
    # 绘制轨迹
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 
            color='yellow', label='AUV Trajectory', linewidth=2)
    
    # 绘制起点和终点
    ax.scatter(*trajectory[0], color='lime', s=100, label='Start')
    ax.scatter(*trajectory[-1], color='red', s=100, label='End')
    ax.scatter(*env_config['goal_pos'], color='magenta', s=100, label='Goal')
    
    # 绘制球体障碍物
    for sphere in env_config['obstacles']['spheres']:
        center = sphere['center']
        r = sphere['radius']
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        x = center[0] + r * np.outer(np.cos(u), np.sin(v))
        y = center[1] + r * np.outer(np.sin(u), np.sin(v))
        z = center[2] + r * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, color='gray', alpha=0.4)
    
    # 修改深度刻度的设置方式
    z_ticks = np.linspace(bounds[4], bounds[5], 6)  # 创建均匀分布的刻度
    ax.set_zticks(z_ticks)
    ax.set_zticklabels([f'{-z:.1f}' for z in z_ticks])
    
    # 每隔几个点绘制一个速度向量
    step = 10  # 可以调整这个值来改变箭头的密度
    dt = 0.1   # 定义时间步长，与环境中的设置相同
    for i in range(0, len(trajectory)-1, step):
        vel = (trajectory[i+1] - trajectory[i]) / dt
        plot_velocity_vector(ax, trajectory[i], vel)
    
    ax.legend()
    plt.title('Underwater AUV Obstacle Avoidance Trajectory')
    
    return fig, ax

def plot_training_metrics(rewards, path_lengths, episode_times, run_id, algorithm):
    """绘制训练指标"""
    vis_dir = f"./result/runs/{run_id}/{algorithm}/visualizations"
    os.makedirs(vis_dir, exist_ok=True)
    
    # 绘制奖励曲线
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, label=algorithm)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(f'Training Rewards - {algorithm}')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'{vis_dir}/training_rewards.png')
    plt.close()
    
    # 绘制路径长度曲线
    plt.figure(figsize=(10, 6))
    plt.plot(path_lengths, label=algorithm, color='green')
    plt.xlabel('Episode')
    plt.ylabel('Path Length (m)')
    plt.title(f'Path Length Evolution - {algorithm}')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'{vis_dir}/path_lengths.png')
    plt.close()
    
    # 绘制每回合运行时间
    plt.figure(figsize=(10, 6))
    plt.plot(episode_times, label=algorithm, color='orange')
    plt.xlabel('Episode')
    plt.ylabel('Episode Time (s)')
    plt.title(f'Episode Running Time - {algorithm}')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'{vis_dir}/episode_times.png')
    plt.close()

def visualize_training_results(run_id, algorithm, difficulty='medium'):
    """可视化训练结果"""
    # 加载数据
    trajectory, rewards, path_lengths, episode_times, env_config, training_stats = load_data(run_id, algorithm, difficulty)
    
    if trajectory is None:
        print("无法加载训练数据，可视化终止")
        return
    
    # 绘制训练指标
    plot_training_metrics(rewards, path_lengths, episode_times, run_id, algorithm)
    
    # 绘制3D轨迹
    fig, ax = plot_3d_trajectory(trajectory, env_config)
    
    # 保存图像并显示
    vis_dir = f"./result/runs/{run_id}/{algorithm}_{difficulty}/visualizations"
    os.makedirs(vis_dir, exist_ok=True)
    fig.savefig(f'{vis_dir}/3d_trajectory.png')
    
    # 打印训练统计信息
    print("\n训练统计信息:")
    print(f"总训练时间: {training_stats['total_training_time']:.2f}秒")
    print(f"平均每回合时间: {training_stats['average_episode_time']:.2f}秒")
    print(f"最终奖励值: {training_stats['final_reward']:.2f}")
    print(f"最终路径长度: {training_stats['final_path_length']:.2f}米")
    
    # 显示交互式3D图像
    print("\n显示3D轨迹图，可以用鼠标旋转查看。关闭窗口继续...")
    plt.show()
    plt.close()

def compare_algorithms(run_id="run1", algorithms=None, difficulty='medium'):
    """比较不同算法的轨迹和奖励"""
    comparison_dir = f"./result/comparisons/{run_id}_{difficulty}"
    os.makedirs(comparison_dir, exist_ok=True)
    
    # 为每个算法加载数据
    trajectories = {}
    rewards = {}
    path_lengths = {}
    episode_times = {}
    colors = {
        'TD3': 'red', 
        'DDPG': 'blue', 
        'DQN': 'green', 
        'PPO': 'purple',
        'SAC': 'orange'
    }
    
    for algo in algorithms:
        try:
            trajectory, reward, path_length, episode_time, env_config, _ = load_data(run_id, algo, difficulty)
            trajectories[algo] = trajectory
            rewards[algo] = reward
            path_lengths[algo] = path_length
            episode_times[algo] = episode_time
            if 'env_config' not in locals():
                env_config = env_config
        except Exception as e:
            print(f"警告：未找到算法 {algo} 的数据文件: {str(e)}")
            continue
    
    # 绘制奖励对比图
    plt.figure(figsize=(12, 8))
    for algo in rewards:
        plt.plot(rewards[algo], color=colors[algo], label=algo)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Rewards Comparison')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'{comparison_dir}/rewards_comparison.png')
    plt.close()
    
    # 绘制路径长度对比图
    plt.figure(figsize=(12, 8))
    for algo in path_lengths:
        plt.plot(path_lengths[algo], color=colors[algo], label=algo)
    plt.xlabel('Episode')
    plt.ylabel('Path Length (m)')
    plt.title('Path Length Comparison')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'{comparison_dir}/path_lengths_comparison.png')
    plt.close()
    
    # 绘制运行时间对比图
    plt.figure(figsize=(12, 8))
    for algo in episode_times:
        plt.plot(episode_times[algo], color=colors[algo], label=algo)
    plt.xlabel('Episode')
    plt.ylabel('Episode Time (s)')
    plt.title('Episode Time Comparison')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'{comparison_dir}/episode_times_comparison.png')
    plt.close()
    
    # 绘制3D轨迹对比图
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # 设置水下环境的背景色和其他设置
    ax.set_facecolor((0.5, 0.8, 0.9, 0.3))
    fig.patch.set_facecolor((0.5, 0.8, 0.9, 0.2))
    
    # 添加水平面和障碍物
    plot_environment(ax, env_config)
    
    # 绘制每个算法的轨迹
    for algo in trajectories:
        ax.plot(trajectories[algo][:, 0], trajectories[algo][:, 1], trajectories[algo][:, 2], 
                color=colors[algo], label=algo, linewidth=2)
    
    # 绘制起点和终点
    ax.scatter([0], [0], [0], color='lime', s=100, label='Start')
    ax.scatter([env_config['goal_pos'][0]], [env_config['goal_pos'][1]], 
              [env_config['goal_pos'][2]], color='magenta', s=100, label='Goal')
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Depth (m)')
    ax.set_title('Comparison of Algorithm Trajectories')
    ax.legend()
    
    plt.savefig(f'{comparison_dir}/trajectory_comparison.png')
    plt.close()

def plot_environment(ax, env_config):
    """绘制环境（水平面和障碍物）"""
    bounds = env_config['bounds']
    # 添加水平面
    x_surface = np.linspace(bounds[0], bounds[1], 50)
    y_surface = np.linspace(bounds[2], bounds[3], 50)
    X, Y = np.meshgrid(x_surface, y_surface)
    Z = np.full_like(X, bounds[5])
    ax.plot_surface(X, Y, Z, alpha=0.1, color='cyan')
    
    # 添加球体障碍物
    for sphere in env_config['obstacles']['spheres']:
        plot_sphere(ax, sphere['center'], sphere['radius'], color='gray', alpha=0.4)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, required=True,
                      help="要可视化的训练运行ID")
    parser.add_argument("--algorithm", type=str, 
                      choices=["TD3", "DDPG", "SAC", "PPO", "A2C"],
                      help="要可视化的算法")
    parser.add_argument("--difficulty", type=str, default="medium",
                      choices=["easy", "medium", "hard"],
                      help="环境难度: easy, medium 或 hard")
    parser.add_argument("--compare", action="store_true",
                      help="是否生成算法比较图")
    parser.add_argument("--interactive", action="store_true",
                      help="是否显示交互式3D图像")
    return parser.parse_args()

def main():
    args = parse_args()

    # 如果不是比较模式，则必须指定算法
    if not args.compare and not args.algorithm:
        parser.error("当不使用--compare时，必须指定--algorithm参数")

    if args.compare:
        # 获取该run下的所有算法
        base_path = f"./result/runs/{args.run_id}"
        algorithms = []
        for d in os.listdir(base_path):
            if os.path.isdir(f"{base_path}/{d}"):
                algo = d.split('_')[0]  # 从文件夹名称中提取算法名称
                if algo not in algorithms:
                    algorithms.append(algo)
                    
        if len(algorithms) > 1:
            compare_algorithms(args.run_id, algorithms, args.difficulty)
            print(f"算法比较图已保存到: ./result/comparisons/{args.run_id}_{args.difficulty}/")
            
            if args.interactive:
                print("显示算法比较图，可以用鼠标旋转查看。关闭窗口继续...")
                plt.show()
    else:
        print(f"正在处理 run_id: {args.run_id}, 算法: {args.algorithm}, 难度: {args.difficulty}")
        
        # 生成轨迹动画
        visualize_trajectory(args.run_id, args.algorithm, args.difficulty)
        
        # 绘制训练结果（奖励曲线和3D轨迹）
        visualize_training_results(args.run_id, args.algorithm, args.difficulty)
        
        print(f"可视化结果已保存到: ./result/runs/{args.run_id}/{args.algorithm}_{args.difficulty}/visualizations/")

if __name__ == "__main__":
    main() 