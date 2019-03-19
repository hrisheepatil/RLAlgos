import numpy as np
import gym

env = gym.make('FrozenLake-v0')
env.reset()
env.render()

Q = np.zeros((env.observation_space.n, env.action_space.n))
n_episodes = 100000
max_trials = 100
learning_rate = 0.8
gamma = 0.9
for n in range(n_episodes):
    current_state = env.reset()
    if np.random.uniform(0, 1) < 0.8:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q[current_state, :])
    t = 0
    while t < max_trials:
        next_state, reward, done, info = env.step(action)
        if np.random.uniform(0, 1) < 0.8:
            new_action = env.action_space.sample()
        else:
            new_action = np.argmax(Q[next_state, :])
        Q[current_state, action] = Q[current_state, action] + learning_rate*(reward + gamma * Q[next_state, new_action] - Q[current_state, action])
        current_state = next_state
        action = new_action
        t += 1
        if done:
            break
env.render()