import numpy as np
import gym

env = gym.make('FrozenLake-v0')
env.reset()
env.render()

Q = np.zeros((env.observation_space.n, env.action_space.n))
n_episodes = 10000
max_trials = 100
learning_rate = 0.8
gamma = 0.9
for n in range(n_episodes):
    current_state = env.reset()
    t = 0
    while t < max_trials:
        if np.random.uniform(0, 1) < 0.8:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[current_state, :])
        next_state, reward, done, info = env.step(action)
        Q[current_state, action] = Q[current_state, action] + learning_rate*(reward + gamma * np.max(Q[next_state, :]) - Q[current_state, action])
        current_state = next_state
        t += 1
        if done:
            break
env.render()
