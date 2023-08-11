import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6, gamma = .8, epsilon=1., alpha=1., decay = .99999, alpha_min =.5, epsilon_min =.005):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.alpha = alpha
        self.alpha_min = alpha_min
        self.decay = decay

        self.trial_count = 1

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        self.epsilon = self.epsilon / self.trial_count
        val = np.random.uniform()
        greedy_action = np.argmax(self.Q[state])

        p = np.ones(self.nA)/(self.nA-1)
        p[greedy_action] = 0

        if val < (self.epsilon - self.epsilon/self.nA):
            # choose action randomly avoiding greedy action
            action = np.random.choice(np.arange(self.nA), p=p)
        else:
            # choose greedy action
            action = greedy_action

        return action
    

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """

        if done:
            return

        greedy_action = np.argmax(self.Q[next_state])

        self.Q[state][action] += self.alpha*(reward + self.gamma*self.Q[next_state][greedy_action] - self.Q[state][action])

        self.alpha = max(self.alpha*self.decay, self.alpha_min)
        self.trial_count += 1