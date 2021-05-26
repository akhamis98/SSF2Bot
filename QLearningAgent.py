from agents import Agent

from SSF2Discretizer import SSF2Discretizer

import numpy as np
import matplotlib.pyplot as plt
import random
import argparse

import pathlib

import time


import numpy as np
import retro
import gym

MAX_TIME_STEPS = 10000 #used to define our state/action/reward array sizes per episode
MAX_TOTAL_RUN_TIME = 10000000 #not used anymore
TIME_PENALTY = -0.1 #reward penalty per frame
DISTANCE_ROUNDING_FACTOR = 5 #rounds distances values for our state space
FRAME_SKIP=5 #DEFAULT 5



#FROM GYM RETRO (heavily modified)
class Frameskip(gym.Wrapper):
    def __init__(self, env, skip=FRAME_SKIP, realtime=False):
        super().__init__(env)
        self._skip = skip
        self.env = env
        self.realtime = realtime
        self.last_time = time.time()

    # def reset(self):
    #     return self.env.reset()

    def render(self):
        if self.realtime:
            while (time.time() - self.last_time) <= 1/80:
                time.sleep(1/10000)
            self.last_time = time.time()
        self.env.render()

    def step(self, act):
        total_rew = 0.0
        done = None
        for i in range(self._skip):
            obs, rew, done, info = self.env.step(act)
            self.render()
            total_rew += rew + TIME_PENALTY
            if done or info['time_left'] == 0 or info['hp_p1'] == 0 or info['hp_p1'] == 0:
                break
        #if stuck in animation, continue the animation until done
        while info['active_p1']:
            obs, rew, done, info = self.env.step(0)
            self.render()
            total_rew += rew + TIME_PENALTY
            if done or info['time_left'] == 0 or info['hp_p1'] == 0 or info['hp_p1'] == 0:
                break
        return obs, total_rew, done, info



class QLearningAgent(Agent):
    
    def __init__(self, env, observation_space=600, gamma=0.9, epsilon = 1, alpha=0.1):   
        self.env = env
        self.actions = env.action_space.n 
        self.states = observation_space
        
        # policy estimate for each state, update in self.q_learning
        self.policy = np.zeros((self.states), dtype=np.int)


        # Q-value estimate in each state, update in self.q_learning
        self.q_values  = np.zeros((self.states,self.actions), dtype=np.float)
        
        self.gamma = gamma

        self.epsilon = epsilon
        
        # need a value for this
        self.alpha = alpha

    #compute distance between players, round to the nearest DISTANCE_ROUNDING_FACTOR
    def compute_dist(self, _info):
        return self.states/2 + int(DISTANCE_ROUNDING_FACTOR * round(float(_info['x_p2'] - _info['x_p1'])/DISTANCE_ROUNDING_FACTOR))

    def getEpsilon(self):
        return self.epsilon

    def setEpsilon(self, epsilon):
        self.epsilon = epsilon

    #set our policy to be random
    def initRandomPolicy(self):
        for i in range(self.policy.shape[0]):
            self.policy[i] = self.env.action_space.sample()

    #load a q-table from the path
    def loadQTable(self, path):
        self.q_values = np.load(pathlib.Path(path))
        self.policy = np.argmax(self.q_values, axis=1)

    def q_learning(self, states, actions, rewards):
        """
        states: np.array of size (N+1) with state at each time (integer) 
        The last state is only used to calculate the terminal Q-value; 
        it is not associated with an update

        actions: np.array of size (N) with action at each time (integer)
        rewards: np.array of size (N) with single step rewards (float)
        
        returns nothing, should modify self.policy, self.values in place

        """
        for i in range(states.shape[0]-1):
            state = int(states[i])
            action = int(actions[i])
            next_state = int(states[i+1])
            next_best_action = np.argmax(self.q_values[next_state])
            self.q_values[state, action] = self.q_values[state, action] + \
            self.alpha * (rewards[i] + self.gamma * self.q_values[next_state, next_best_action] - self.q_values[state, action])
      
        self.policy = np.argmax(self.q_values, axis=1)


    #Base of this taken from gym retro, heavily changed though
    def rollout(self):
        """
        Perform a rollout using a preset collection of actions
        """
        total_rew = 0
        self.env.reset()
        steps = 0
        states = np.zeros(MAX_TIME_STEPS+1, dtype=np.int)
        actions = np.zeros((MAX_TIME_STEPS), dtype=np.int)
        rewards = np.zeros((MAX_TIME_STEPS,), dtype=np.float)
        i = 0
        old_dist = abs(307-205)
        p2_hp = 100
        p1_hp = 100
        time_left = 100
        for j in range(MAX_TIME_STEPS):

            if (i > 0):
                states[i] = int(self.compute_dist(_info))

            # epsilon greedy
            if np.random.random() < self.epsilon:
                actions[i] = self.policy[states[i]]
            else:
                actions[i] = self.env.action_space.sample()
            
            #take a step (expected run Frameskip.step())
            _obs, rew, done, _info = self.env.step(actions[i])

            #small reward for distance reduction
            if (actions[i] != 0 and old_dist > abs(_info['x_p1'] - _info['x_p2'])):
                rew += 3
            
            total_rew += rew
            rewards[i] = rew
            old_dist = abs(_info['x_p1'] - _info['x_p2'])
            steps += 1
            i += 1
            if done == True or _info['time_left'] <= 0 or _info['hp_p1'] <= 0 or _info['hp_p1'] <= 0:
                p2_hp = _info['hp_p2']
                p1_hp = _info['hp_p1']
                time_left = _info['time_left']
                break
        states[-1] = int(self.compute_dist(_info))
       
        return steps, total_rew, states, actions, rewards, p1_hp, p2_hp, time_left

#base of this adopted from brute.py in retro/examples
def run_game(
    game,
    state=retro.State.DEFAULT,
    scenario=None,
    attempts=1,
    training_episodes=100,
    benchmarking_episodes=20
):
    

    #for benchmarking
    reward_log = np.zeros((attempts, benchmarking_episodes))
    p1_hp_log = np.zeros((attempts, benchmarking_episodes))
    p2_hp_log = np.zeros((attempts, benchmarking_episodes))
    time_left_log = np.zeros((attempts, benchmarking_episodes))
    victory_log = np.zeros((attempts, benchmarking_episodes))
    env = retro.make(game, state= state)
    env = SSF2Discretizer(env)
    print('SSF2Discretizer action_space', env.action_space)
    
    env = Frameskip(env)

    for attempt in range(attempts):
        
        agent = QLearningAgent(env, observation_space=1000, epsilon=0.5, gamma=0.9, alpha=0.01)
        agent.initRandomPolicy()
        desiredEpsilon = 0.8
        epsilonIncrementFactor = 0.05
        timesteps = 0
        best_rew = float('-inf')

        #training
        for i in range(training_episodes):
            steps, rew, states, actions, rewards, p1_hp, p2_hp, time_left = agent.rollout()
            acts = actions[:steps]
            agent.q_learning(states[:steps+1], acts, rewards[:steps])
            #acts, rew = agent.run()
            timesteps += len(acts)

            #increment epsilon slowly to our desired value
            if agent.getEpsilon() < desiredEpsilon:
                agent.setEpsilon(agent.getEpsilon() + epsilonIncrementFactor)


        #final benchmarking 
        agent.setEpsilon = 1
        for i in range(benchmarking_episodes):
            steps, rew, states, actions, rewards, p1_hp, p2_hp, time_left = agent.rollout()
            acts = actions[:steps]
            #acts, rew = agent.run()
            timesteps += len(acts)

            #record data
            reward_log[attempt, i] = rew
            p1_hp_log[attempt, i] = p1_hp/176
            p2_hp_log[attempt, i] = p2_hp/176
            time_left_log[attempt, i] = time_left
            victory_log[attempt, i] = int(p1_hp > p2_hp)

        #print results per attempt
        print("Benchmark results")
        print(f"Average reward: {np.mean(reward_log[attempt])}")
        print(f"Average P1 hp: {np.mean(p1_hp_log[attempt])}")
        print(f"Average P2 hp: {np.mean(p2_hp_log[attempt])}")
        print(f"Average time left: {np.mean(time_left_log[attempt])}")
        print(f"Average victories: {np.mean(p1_hp_log[attempt])}")


        plt.figure()
        plt.plot(reward_log[attempt])
        plt.title("Rewards")
        plt.ylabel("Reward")
        plt.savefig("plots/attempt_" + str(attempt) + "_rewards.png")


        plt.figure()
        plt.plot(p1_hp_log[attempt], color="b")
        plt.plot(p2_hp_log[attempt], color="r")
        plt.title("Player HP (blue=P1, red=P2)")
        plt.ylabel("HP")
        plt.savefig("plots/attempt_" + str(attempt) + "_hp.png")

        plt.figure()
        plt.plot(time_left_log[attempt])
        plt.title("Time left")
        plt.ylabel("Time")
        plt.savefig("plots/attempt_" + str(attempt) + "_time_left.png")

        plt.figure()
        plt.plot(victory_log[attempt])
        plt.title("Victories for P1")
        plt.ylabel("Victory")
        plt.savefig("plots/attempt_" + str(attempt) + "_victories.png")

        #save q-table
        table_path = './q-tables/' + str(state) + "/skip_" + str(FRAME_SKIP) + "/dist_rnd_" + str(DISTANCE_ROUNDING_FACTOR)
        pathlib.Path(table_path).mkdir(parents=True, exist_ok=True)
        np.save(table_path + "/" + str(int(time.time())), agent.q_values)

    #save plot of averages per attempt 
    plot_path = './plots/avg/' + str(state) + "/skip_" + str(FRAME_SKIP) + "/dist_rnd_" + str(DISTANCE_ROUNDING_FACTOR)
    pathlib.Path(plot_path).mkdir(parents=True, exist_ok=True)
    plot_path = plot_path[2:]
    #generate average plots per attempt
    plt.figure()
    plt.scatter(np.arange(attempts), np.mean(p1_hp_log, axis=1), color="b")
    plt.scatter(np.arange(attempts), np.mean(p2_hp_log, axis=1), color="r")
    plt.title("Avg Player HP per Attempt (blue=P1, red=P2)")
    plt.ylabel("HP")
    plt.xlabel("Attempt")
    plt.savefig(plot_path + "/hp.png")

    plt.figure()
    plt.scatter(np.arange(attempts), np.mean(reward_log, axis=1))
    plt.title("Avg Reward per attempt")
    plt.ylabel("Reward")
    plt.xlabel("Attempt")
    plt.savefig(plot_path + "/rewards.png")
    
    plt.figure()
    plt.scatter(np.arange(attempts), np.mean(victory_log, axis=1))
    plt.title("Avg Victories per attempt")
    plt.ylabel("Victories")
    plt.xlabel("Attempt")
    plt.savefig(plot_path + "/victories.png")

    plt.figure()
    plt.scatter(np.arange(attempts), np.mean(time_left_log, axis=1))
    plt.title("Avg time left per attempt")
    plt.ylabel("Time left")
    plt.xlabel("Attempt")
    plt.savefig(plot_path + "/time_left.png")
    


#uncomment this to run, 
run_game("SuperStreetFighter2-Snes-main", state="ryu_vs_ken_highest_difficulty", attempts=1, training_episodes=1000, benchmarking_episodes=1)

#state for cammy bot
#state="ryu_vs_cammy_highest_difficulty"