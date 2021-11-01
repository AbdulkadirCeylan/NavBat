#!/usr/bin/env python

'''
    Training code made by Ricardo Tellez <rtellez@theconstructsim.com>
    Based on many other examples around Internet
    Visit our website at www.theconstruct.ai
'''
import gym
import time
import numpy
import random
import time
import qlearn
from gym import wrappers
import matplotlib.pyplot as plt
# ROS packages required
import rospy
import rospkg

# import our training environment
import env_target_try
#import env_target_try


if __name__ == '__main__':
    
    env = gym.make('Quad_target_trying-v0')
    rospy.loginfo ("Gym environment done")
        
    # Set the logging system
    rospack = rospkg.RosPack()
    #outdir = pkg_path + '/training_results'
    #env = wrappers.Monitor(env, outdir, force=True) 
    rospy.loginfo ( "Monitor Wrapper started")

    last_time_steps = numpy.ndarray(0)

    # Loads parameters from the ROS param server
    # Parameters are stored in a yaml file inside the config directory
    # They are loaded at runtime by the launch file
    Alpha = 0.2
    Epsilon = 0.90
    Gamma = 0.8
    epsilon_discount = 0.995
    nepisodes = 1000
    nsteps = 500

    # Initialises the algorithm that we are going to use for learning
    qlearn = qlearn.QLearn(actions=range(env.action_space.n),
                    alpha=Alpha, gamma=Gamma, epsilon=Epsilon)
    initial_epsilon = qlearn.epsilon

    highest_reward = 0
    total_steps = 0
    cum_rew = []
    epsilons = []
    episode_time = []
    # Starts the main training loop: the one about the episodes to do
    for x in range(nepisodes):
        #rospy.loginfo ("STARTING Episode #"+str(x))
        start_time = time.time()
        
        cumulated_reward = 0  
        done = False
        if qlearn.epsilon > 0.05:
            qlearn.epsilon *= epsilon_discount

        epsilons = numpy.append(epsilons,qlearn.epsilon)
        # Initialize the environment and get first state of the robot
        observation = env.reset()
        state = ''.join(map(str, observation))
        
        # for each episode, we test the robot for nsteps
        for i in range(nsteps):
            total_steps = total_steps+1
            # Pick an action based on the current state
            action = qlearn.chooseAction(state)

            # Execute the action in the environment and get feedback
            observation, reward, done, info = env.step(action)
            cumulated_reward += reward
            if highest_reward < cumulated_reward:
                highest_reward = cumulated_reward

            nextState = ''.join(map(str, observation))

            # Make the algorithm learn based on the results
            qlearn.learn(state, action, reward, nextState)

            if not(done):
                state = nextState
            else:
                rospy.loginfo ("DONE")
                last_time_steps = numpy.append(last_time_steps, [int(i + 1)])
                break 
        end_time= time.time()-start_time
        cum_rew = numpy.append(cum_rew,cumulated_reward)
        episode_time = numpy.append(episode_time,end_time)
        #m, s = divmod(int(time.time() - start_time), 60)
        #h, m = divmod(m, 60)
        print( ("EP: "+str(x+1)+" - [alpha: "+str(round(qlearn.alpha,2))+" - gamma: "+str(round(qlearn.gamma,2))+" - epsilon: "+str(round(qlearn.epsilon,2))+"] - Reward: "+str(cumulated_reward)))
        print("total_step: ",total_steps)

    x = numpy.arange(0,len(cum_rew))
    plt.plot(x,cum_rew)
    plt.xlabel("Episodes")
    plt.ylabel("Cumulated Reward")
    plt.show()

    plt.plot(x,episode_time)
    plt.xlabel("Episodes")
    plt.ylabel("Time for Each Episode")
    plt.show()

    plt.plot(x,epsilons)
    plt.xlabel("Episodes")
    plt.ylabel("Epsilon")
    plt.show()
    rospy.loginfo ( ("\n|"+str(nepisodes)+"|"+str(qlearn.alpha)+"|"+str(qlearn.gamma)+"|"+str(initial_epsilon)+"*"+str(epsilon_discount)+"|"+str(highest_reward)+"| PICTURE |"))

    l = last_time_steps.tolist()
    l.sort()

    #print("Parameters: a="+str)
    rospy.loginfo("Overall score: {:0.2f}".format(last_time_steps.mean()))
    rospy.loginfo("Best 100 score: {:0.2f}".format(reduce(lambda x, y: x + y, l[-100:]) / len(l[-100:])))

    env.close()
