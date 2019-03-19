import gym
import numpy as np
import random
from time import sleep

env = gym.make("FrozenLake-v0")
env.reset()

def create_random_policy(env):
    policy ={}
    for key in range(0, env.observation_space.n):
        p = {}
        for action in range(0, env.action_space.n):
            p[action] = 1/env.action_space.n
        policy[key] = p
    return policy

def create_state_action_dictionary(env, policy):
    Q = {}
    for key in policy.keys():
        Q[key] = {a: 0.0 for a in range(0, env.action_space.n)}
    return Q

def run_game(env, policy, display = True):
    env.reset()
    episode = []
    finished = False

    while not finished:
        s = env.env.s
        if display:
            env.render()
            sleep(1)

        timestep = []
        timestep.append(s)
        n = random.uniform(0, sum(policy[s].values()))
        top_range = 0
        for prob in policy[s].items():
            top_range = top_range + prob[1]
            if n <= top_range:
                action = prob[0]
                break
        state, reward, finished, info = env.step(action)
        timestep.append(action)
        timestep.append(reward)
        episode.append(timestep)

    if display:
        env.render()
        sleep(1)
    return episode

def test_policy(policy, env):
    wins = 0
    r = 100
    for i in range(r):
        w = run_game(env, policy, display = False)[-1][-1]
        if w == 1:
            wins = wins + 1
    return wins/r

def monte_carlo_e_soft(env, episodes=10, policy=None, epsilon = 0.01):
    if not policy:
        policy = create_random_policy(env)
    #print(policy)
    Q = create_state_action_dictionary(env, policy)
    #print(Q)
    returns = {}
    for _ in range(episodes):
        G = 0
        episode = run_game(env, policy, display = False)
        print(episode)
        print(len(episode))
        k = 0
        for i in reversed(range(0, len(episode))):
            s_t, a_t, r_t = episode[i]
            state_action = s_t, a_t
            print(state_action)
            print("break")
            G += r_t

            if not state_action in [(x[0], x[1]) for x in episode[0:i]]:
                k = k + 1
                if returns.get(state_action):
                    returns[state_action].append(G)
                else:
                    returns[state_action] = [G]
                print(returns)

        print("total k",k)

#policy = create_random_policy(env)
#create_state_action_dictionary(env, policy)
#wins = test_policy(policy, env)
#print(wins)

monte_carlo_e_soft(env)