#!/usr/bin/env python
import custom_env
import gym
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines import DQN
import numpy
import matplotlib.pyplot as plt



env = gym.make('QuadcopterLiveShow-v0')
model = DQN(MlpPolicy, env, verbose=1,tensorboard_log="./DQN_TR-v0_tensorboard/")
model.learn(total_timesteps=20000, tb_log_name="eysorun")
cum_rew = []
model.save("Training-last")

#done = False
#cumulated_reward = 0
#obs = env.reset()
#for k in range (10):
    #action, _states = model.predict(obs,deterministic=True)
    #obs, rewards, done, info = env.step(action)



