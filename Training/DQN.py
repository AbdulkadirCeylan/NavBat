#!/usr/bin/env python
import custom_env_walking
import gym
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines import DQN, PPO2,DDPG, A2C
import numpy
import matplotlib.pyplot as plt
import numpy as np

env = gym.make('QuadcopterLiveShow-v1')
#model = DQN(MlpPolicy, env, verbose=1,tensorboard_log="./DQN_TR-v0_tensorboard/")
model = DQN("MlpPolicy", env,exploration_fraction=0.02,learning_rate= 0.0001, verbose=1,tensorboard_log="./DQN_TR-v0_tensorboard/")
model.learn(total_timesteps=5000, tb_log_name="walking_tests")
model.save("Training-last6")

#done = False
#cumulated_reward = 0
#obs = env.reset()
#for k in range (10):
    #action, _states = model.predict(obs,deterministic=True)
    #obs, rewards, done, info = env.step(action)



