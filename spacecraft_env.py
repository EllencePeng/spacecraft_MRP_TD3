import numpy as np

import gym
from gym import spaces
from sympy import false

from model import spacecraft_simplified
from scipy.integrate import solve_ivp

import matplotlib.pyplot as plt

class SpacecraftEnv(gym.Env):
    def __init__(self, time, dt):
        self.max_speed = 0.1
        self.max_torque = 1
        self.dt = dt                      #每隔dt时间   获取一次控制量 dt,  积分dt时间
        self.J = np.array([[120, 0, 0],
                           [0, 100, 0],
                           [0, 0, 120]])
        self.sat = spacecraft_simplified(self.J)
        self.state = np.array([[0],[0],[0], [0.3],[0.1],[-0.2]])[:,-1]  #中括号内的意思表示取所有行的最后一列，得到的是一个列向量（储存的是一维数组）

        self.time_prev = 0
        self.i = 0      #i表示采取了多少次计算，共i个dt时间
        self.n = time/self.dt
        #给初值
        self.X_sol = np.array([[0],[0],[0], [0.3],[0.1],[-0.2]])  # 前三个对应omega，后三个对应MRP
        self.t_sol = np.array([self.time_prev])
        self.Tr_sol = np.array([[0],[0],[0]])


        states_high = np.array([self.max_speed, self.max_speed, self.max_speed, 1, 1, 1], dtype=np.float32)
        # 前三个状态为角速度，后三个对应MRP
        action_high = np.array([self.max_torque, self.max_torque, self.max_torque], dtype=np.float32)
        #对应三个方向的力矩

        #   This will throw a warning in tests/envs/test_envs in utils/env_checker.py as the space is not symmetric
        #   or normalised as max_torque == 2 by default. Ignoring the issue here as the default settings are too old
        #   to update to follow the openai gym api
        self.action_space = spaces.Box(low=-action_high, high=action_high, dtype=np.float32)
        self.observation_space = spaces.Box(low=-states_high, high=states_high, dtype=np.float32)

    def step(self, Tr):  #输入的Tr是三维的ndarray
        self.i += 1 
        dt = self.dt
        Tr = np.clip(Tr, -self.max_torque, self.max_torque)

        time = self.i * dt
        tspan = np.linspace(self.time_prev,time,100) # 每个deltat时间划分为100个步长来积分
        tspanivp = (tspan[0], tspan[-1])
        
        sol = solve_ivp(fun=lambda t, X: self.sat.dynamics(t, X, Tr),
                        t_span=tspanivp,
                        y0=self.X_sol[:,-1],
                        t_eval=tspan,
                        method='LSODA')

        self.state = sol.y[:,-1]
        self.t_sol = np.append(self.t_sol, time)
        self.X_sol = np.column_stack((self.X_sol, sol.y[:,-1]))
        self.Tr_sol = np.column_stack((self.Tr_sol, Tr))
        self.time_prev = time

        X = self.state
        omega = X[:3]
        p = X[3:]

        r1 = - np.linalg.norm(X, ord=1) - np.sum(Tr**2)
        r2 = 1 / (np.linalg.norm(p, ord=1) + 0.01)*(np.min(np.abs(X)) <= 0.1)
        r3 = -100*(np.max(np.abs(p))>=4 or np.max(np.abs(omega))>=4)
        R = r1+r2+r3

        if self.i >= self.n:
            return self.state, R, True, False #next_state, reward, done, _
        return self.state, R, False, False
    
    def reset(self):
        self.state = np.array([[0],[0],[0], [0.3],[0.1],[-0.2]])[:,-1]
        self.time_prev = 0
        self.i = 0      #i表示采取了多少次计算，共i个dt时间
        self.X_sol = np.array([[0],[0],[0], [0.3],[0.1],[-0.2]])  # 前三个对应omega，后四个对应四元数
        self.t_sol = np.array([self.time_prev])
        self.Tr_sol = np.array([[0],[0],[0]])
        return self.state
    
def plot_omega(t_sol, X_sol):
    plt.xlabel('Time')
    plt.ylabel('anguler-omega')
    plt.plot(t_sol, X_sol[0], linestyle='--', label='omega_x')
    plt.plot(t_sol, X_sol[1], linestyle='--', label='omega_y')
    plt.plot(t_sol, X_sol[2], linestyle='--', label='omega_z')
    plt.legend(loc='upper right')
    plt.title('anguler-omega')
    plt.grid()
    plt.show()

def plot_p(t_sol, X_sol):
    plt.xlabel('Time')
    plt.ylabel('MRP')
    plt.plot(t_sol, X_sol[3],  linestyle='-', label='p1')
    plt.plot(t_sol, X_sol[4],  linestyle='-', label='p2')
    plt.plot(t_sol, X_sol[5],  linestyle='-', label='p3')
    plt.legend(loc='upper right')
    plt.title('MRP')
    plt.grid()
    plt.show()

def plot_Tr(t_sol, Tr_sol):
    plt.xlabel('Time')
    plt.ylabel('Tr')
    plt.plot(t_sol, Tr_sol[0],  linestyle='-', label='Tr1')
    plt.plot(t_sol, Tr_sol[1],  linestyle='-', label='Tr2')
    plt.plot(t_sol, Tr_sol[2],  linestyle='-', label='Tr3')
    plt.legend(loc='upper right')
    plt.title('Tr')
    plt.grid()
    plt.show()