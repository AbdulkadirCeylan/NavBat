#!/usr/bin/env python
import custom_env_trained_model
import gym
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines import DQN
import numpy
import matplotlib.pyplot as plt

def main():
    env = gym.make('QuadcopterLiveShow-v2')
    model = DQN.load("Training-v0")
    obs = env.reset()
    for k in range(5):
        action, _states = model.predict(obs,deterministic=True)
        obs, _, _, info = env.step(action)

    done = True
    return done


if __name__ == '__main__':
    main()