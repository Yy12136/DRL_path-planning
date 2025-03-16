import gymnasium as gym
import numpy as np
from gymnasium import spaces
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 添加3D支持

class UnderwaterRobotEnv(gym.Env):
    def __init__(self, difficulty='medium'):
        super(UnderwaterRobotEnv, self).__init__()
        
        # 添加难度设置
        self.difficulty = difficulty
        if difficulty not in ['easy', 'medium', 'hard']:
            raise ValueError("难度必须是 'easy', 'medium' 或 'hard'")
        
        # 物理参数
        self.water_density = 1000.0  # 水密度 (kg/m³)
        self.drag_coefficient = 0.5   # 阻力系数
        self.robot_mass = 10.0       # 机器人质量 (kg)
        self.robot_volume = 0.01     # 机器人体积 (m³)
        self.robot_cross_section = 0.1  # 横截面积 (m²)
        self.gravity = 9.81          # 重力加速度 (m/s²)
        self.dt = 0.1                # 时间步长 (s)
        
        # 定义动作空间和观察空间
        self.action_space = spaces.Box(
            low=np.array([-1, -1, -1]),
            high=np.array([1, 1, 1]),
            dtype=np.float32
        )
        
        # 扩展观察空间以包含速度信息
        self.observation_space = spaces.Box(
            low=np.array([-10, -10, -10, -2, -2, -2, -10, -10, -10]),
            high=np.array([10, 10, 10, 2, 2, 2, 10, 10, 10]),
            dtype=np.float32
        )
        
        # 初始化状态
        self.robot_pos = np.zeros(3, dtype=np.float32)
        self.robot_vel = np.zeros(3, dtype=np.float32)
        self.goal_pos = np.array([8, 8, 8], dtype=np.float32)
        self.obstacles = self._generate_obstacles()
        self.fig = None
        self.ax = None
        
    def _generate_obstacles(self):
        """根据难度生成障碍物"""
        if self.difficulty == 'easy':
            obstacles = {
                'spheres': [
                    {'center': np.array([6, 8, 4], dtype=np.float32), 'radius': 1.0},
                    {'center': np.array([2.5, 2.5, 1], dtype=np.float32), 'radius': 1.0},
                    {'center': np.array([5, 4.5, 6], dtype=np.float32), 'radius': 1.0},
                ]
            }
        elif self.difficulty == 'medium':
            obstacles = {
                'spheres': [
                    {'center': np.array([5.5, 1.5, 4], dtype=np.float32), 'radius': 1.2},
                    {'center': np.array([6, 8, 5], dtype=np.float32), 'radius': 1.0},
                    {'center': np.array([0, 6, 3], dtype=np.float32), 'radius': 1.2},
                    {'center': np.array([2, 2, 1], dtype=np.float32), 'radius': 1.0},
                    {'center': np.array([3, 6, 3], dtype=np.float32), 'radius': 1.0},
                    {'center': np.array([5, 4.5, 6], dtype=np.float32), 'radius': 1.0},
                ]
            }
        else:  # hard
            obstacles = {
                'spheres': [
                    {'center': np.array([5.5, 1.5, 4], dtype=np.float32), 'radius': 1.2},
                    {'center': np.array([6, 8, 7], dtype=np.float32), 'radius': 1.0},
                    {'center': np.array([0, 6, 3], dtype=np.float32), 'radius': 1.2},
                    {'center': np.array([2, 2, 1], dtype=np.float32), 'radius': 1.0},
                    {'center': np.array([3, 6, 3], dtype=np.float32), 'radius': 1.0},
                    {'center': np.array([5, 4.5, 6], dtype=np.float32), 'radius': 1.0},
                    {'center': np.array([4, 6.2, 5], dtype=np.float32), 'radius': 1.0},
                    {'center': np.array([7, 3.8, 3], dtype=np.float32), 'radius': 0.9},
                    {'center': np.array([1.5, 7, 5], dtype=np.float32), 'radius': 0.8},
                    {'center': np.array([1, 3.5, 6.5], dtype=np.float32), 'radius': 1.0},
                ]
            }
        return obstacles
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        # 重置机器人位置和速度
        self.robot_pos = np.zeros(3, dtype=np.float32)
        self.robot_vel = np.zeros(3, dtype=np.float32)
        
        return self._get_obs(), {}
    
    def _apply_water_resistance(self):
        """应用水阻力"""
        velocity_magnitude = np.linalg.norm(self.robot_vel)
        if velocity_magnitude > 0:
            # 计算阻力
            drag_force = -0.5 * self.drag_coefficient * self.water_density * \
                        self.robot_cross_section * velocity_magnitude * self.robot_vel
            # 应用阻力
            self.robot_vel += drag_force * self.dt / self.robot_mass

    def _apply_buoyancy(self):
        """应用浮力"""
        # 计算净浮力
        buoyancy_force = self.water_density * self.robot_volume * self.gravity
        gravity_force = self.robot_mass * self.gravity
        net_force = buoyancy_force - gravity_force
        
        # 应用到垂直方向的速度
        self.robot_vel[2] += net_force * self.dt / self.robot_mass

    def _get_obs(self):
        """获取观察状态（包含速度信息）"""
        closest_obstacle = self._get_closest_obstacle()
        return np.concatenate([
            self.robot_pos,    # 位置 [3]
            self.robot_vel,    # 速度 [3]
            closest_obstacle - self.robot_pos  # 相对障碍物位置 [3]
        ])
    
    def _get_closest_obstacle(self):
        # 获取最近障碍物的位置
        min_dist = float('inf')
        closest_point = None
        
        # 只检查球体
        for sphere in self.obstacles['spheres']:
            dist = np.linalg.norm(sphere['center'] - self.robot_pos)
            if dist < min_dist:
                min_dist = dist
                closest_point = sphere['center']
        
        return closest_point
    
    def step(self, action):
        # 更新速度（动作代表推进器的力）
        thrust_force = action * 20.0  # 将标准化动作转换为实际推力
        self.robot_vel += thrust_force * self.dt / self.robot_mass
        
        # 应用水动力学效应
        self._apply_water_resistance()
        self._apply_buoyancy()
        
        # 更新位置
        self.robot_pos += self.robot_vel * self.dt
        
        # 限制位置和速度
        self.robot_pos = np.clip(self.robot_pos, [-10, -10, -10], [10, 10, 10])
        self.robot_vel = np.clip(self.robot_vel, [-2, -2, -2], [2, 2, 2])
        
        # 计算奖励和完成状态
        done = self._check_collision()
        reward = self._compute_reward()
        
        # 检查是否到达目标
        if np.linalg.norm(self.robot_pos - self.goal_pos) < 0.5:
            done = True
            reward += 100
            
        return self._get_obs(), reward, done, False, {}
    
    def _check_collision(self):
        # 只检查与球体的碰撞
        for sphere in self.obstacles['spheres']:
            if np.linalg.norm(self.robot_pos - sphere['center']) <= sphere['radius']:
                return True
        return False
    
    def _compute_reward(self):
        """计算奖励"""
        reward = 0
        
        # 距离目标的奖励
        dist_to_goal = np.linalg.norm(self.robot_pos - self.goal_pos)
        reward -= dist_to_goal * 0.1
        
        # 能量消耗惩罚
        energy_penalty = np.sum(np.square(self.robot_vel)) * 0.01
        reward -= energy_penalty
        
        # 碰撞惩罚
        if self._check_collision():
            reward -= 100
            
        return reward 

    def render(self):
        """3D可视化环境"""
        if self.fig is None:
            self.fig = plt.figure(figsize=(10, 10))
            self.ax = self.fig.add_subplot(111, projection='3d')
        
        self.ax.clear()
        
        # 设置坐标轴
        self.ax.set_xlim([-2, 10])
        self.ax.set_ylim([-2, 10])
        self.ax.set_zlim([-2, 10])
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        
        # 绘制机器人
        self.ax.scatter(*self.robot_pos, color='blue', s=100, label='Robot')
        
        # 绘制目标点
        self.ax.scatter(*self.goal_pos, color='green', s=100, label='Goal')
        
        # 绘制球体障碍物
        for sphere in self.obstacles['spheres']:
            center = sphere['center']
            r = sphere['radius']
            
            # 创建球体的网格点
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 20)
            x = center[0] + r * np.outer(np.cos(u), np.sin(v))
            y = center[1] + r * np.outer(np.sin(u), np.sin(v))
            z = center[2] + r * np.outer(np.ones(np.size(u)), np.cos(v))
            
            self.ax.plot_surface(x, y, z, color='red', alpha=0.3)
        
        self.ax.legend()
        plt.pause(0.1)
    
    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None 