from agent import Agent
from monitor import interact
import  gym #gymnasium as
import numpy as np

import faulthandler

faulthandler.enable()

env = gym.make('Taxi-v3', render_mode="human")

agent = Agent()

seed = 10
np.random.seed(seed)
#env.seed(seed)
env.action_space.seed(seed)

cumulative_best_reward = 0 
n_runs = 35
for i in range(n_runs):
    try:
        print('Test {}:'.format(i))
        avg_rewards, best_avg_reward = interact(env, agent)
        cumulative_best_reward += best_avg_reward
    except:
        i = i-1

avg_rewards = cumulative_best_reward / n_runs

print('Overall average rewards over {} test: {}'.format(n_runs, avg_rewards))