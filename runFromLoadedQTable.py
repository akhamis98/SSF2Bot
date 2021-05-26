from QLearningAgent import QLearningAgent
from QLearningAgent import Frameskip
import retro
import numpy as np
import time
import pathlib
import matplotlib.pyplot as plt
from SSF2Discretizer import SSF2Discretizer

MAX_TIME_STEPS = 6000

#Run the game a saved policy
def run_q_table(
    game,
    state=retro.State.DEFAULT,
    scenario=None,
    repeat=100,
    qTablePath=""
):
    
    #for benchmarking
    reward_log = np.zeros((repeat))
    p1_hp_log = np.zeros((repeat))
    p2_hp_log = np.zeros((repeat))
    time_left_log = np.zeros((repeat))
    victory_log = np.zeros((repeat))
    env = retro.make(game, state= state)
    env = SSF2Discretizer(env)
    print('SSF2Discretizer action_space', env.action_space)
    
    env = Frameskip(env, skip=2, realtime=True)
    agent = QLearningAgent(env, observation_space=1000, epsilon=1)
    agent.loadQTable(qTablePath)

    for i in range(repeat):
        
        

        steps, rew, states, actions, rewards, p1_hp, p2_hp, time_left = agent.rollout()
        acts = actions[:steps]
        #acts, rew = agent.run()

        #record data
        reward_log[i] = rew
        p1_hp_log[i] = p1_hp/176
        p2_hp_log[i] = p2_hp/176
        time_left_log[i] = time_left
        victory_log[i] = int(p1_hp > p2_hp)

    print("Benchmark results")
    print(f"Average reward: {np.mean(reward_log)}")
    print(f"Average P1 hp: {np.mean(p1_hp_log)}")
    print(f"Average P2 hp: {np.mean(p2_hp_log)}")
    print(f"Average time left: {np.mean(time_left_log)}")
    print(f"Average victories: {np.mean(victory_log)}")

    plot_path = './plots/tests/' + str(time.time())
    pathlib.Path(plot_path).mkdir(parents=True, exist_ok=True)
    plot_path = plot_path[2:]
    #generate average plots per attempt
    plt.figure()
    plt.scatter(np.arange(repeat), p1_hp_log, color="b")
    plt.scatter(np.arange(repeat), p2_hp_log, color="r")
    plt.title("Player HP per test (blue=P1, red=(Cammy))")
    plt.ylabel("HP")
    plt.xlabel("Attempt")
    plt.savefig(plot_path + "/hp.png")

    plt.figure()
    plt.scatter(np.arange(repeat), reward_log)
    plt.title("Reward per test (Cammy)")
    plt.ylabel("Reward")
    plt.xlabel("Attempt")
    plt.savefig(plot_path + "/rewards.png")
    
    plt.figure()
    plt.scatter(np.arange(repeat), victory_log)
    plt.title("Victories per test (Cammy)")
    plt.ylabel("Victories")
    plt.xlabel("Attempt")
    plt.savefig(plot_path + "/victories.png")

    plt.figure()
    plt.scatter(np.arange(repeat), time_left_log)
    plt.title("Time left per test (Cammy)")
    plt.ylabel("Time left")
    plt.xlabel("Attempt")
    plt.savefig(plot_path + "/time_left.png")


    


#run_game must be commented in QLearningAgent.py
run_q_table("SuperStreetFighter2-Snes-main", state="ryu_vs_ken_highest_difficulty", repeat=2, qTablePath="./q-tables/State.DEFAULT/skip_5/dist_rnd_5/best.npy")
