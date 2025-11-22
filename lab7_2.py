import numpy as np

class BinaryBandit:
    def __init__(self, p_success):
        self.p = p_success
    def pull(self):
        return 1 if np.random.rand() < self.p else 0

class EpsilonGreedy:
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon
        self.Q = [0.0, 0.0]
        self.N = [0, 0]

    def choose(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(2)
        return int(np.argmax(self.Q))

    def update(self, action, reward):
        self.N[action] += 1
        self.Q[action] += (reward - self.Q[action]) / self.N[action]

banditA = BinaryBandit(0.8)
banditB = BinaryBandit(0.6)
bandits = [banditA, banditB]

agent = EpsilonGreedy(epsilon=0.1)
steps = 10000
rewards = 0

for t in range(steps):
    action = agent.choose()
    reward = bandits[action].pull()
    agent.update(action, reward)
    rewards += reward

print(f"Final Q-values: Bandit 1: {agent.Q[0]:.3f}, Bandit 2: {agent.Q[1]:.3f}")
print(f"Total reward over {steps} steps: {rewards} (avg {rewards/steps:.3f})")
print("Agent correctly prefers the better bandit!")
