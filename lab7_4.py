# task4_nonstationary_agent.py
# Shows that constant-alpha ε-greedy tracks non-stationary rewards

import numpy as np
import matplotlib.pyplot as plt
import copy
from task3 import NonStationary10ArmedBandit

class EpsilonGreedyAgent:
    def __init__(self, k=10, epsilon=0.1, alpha=None):
        self.k = k
        self.epsilon = epsilon
        self.alpha = alpha
        self.Q = np.zeros(k)
        self.N = np.zeros(k, dtype=int)

    def select(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.k)
        return int(np.argmax(self.Q))

    def update(self, a, r):
        self.N[a] += 1
        if self.alpha is None:
            self.Q[a] += (r - self.Q[a]) / self.N[a]
        else:
            self.Q[a] += self.alpha * (r - self.Q[a])

# Experiment
steps = 10000
runs = 200

agents = {
    "Sample-average ε-greedy": EpsilonGreedyAgent(alpha=None),
    "α=0.1 ε-greedy":          EpsilonGreedyAgent(alpha=0.1),
    "α=0.3 ε-greedy":          EpsilonGreedyAgent(alpha=0.3),
}

percent_optimal = {name: np.zeros(steps) for name in agents}

print("Running 10,000 steps × 200 runs...")
for run in range(runs):
    bandit = NonStationary10ArmedBandit(seed=run)
    ags = {n: copy.deepcopy(ag) for n, ag in agents.items()}

    for t in range(steps):
        for name, agent in ags.items():
            a = agent.select()
            r = bandit.step(a)
            agent.update(a, r)
            if a == bandit.optimal_action():
                percent_optimal[name][t] += 1
    if (run+1) % 50 == 0:
        print(f"   → {run+1}/{runs} runs completed")

for name in percent_optimal:
    percent_optimal[name] = percent_optimal[name] / runs * 100

print("\nFinal performance (last 1000 steps):")
for name, data in percent_optimal.items():
    print(f"{name:25} → {np.mean(data[-1000:]):.1f}% optimal")

# Plot
plt.figure(figsize=(10,6))
for name, data in percent_optimal.items():
    smoothed = np.convolve(data, np.ones(500)/500, mode='valid')
    plt.plot(smoothed, label=name)
plt.xlabel("Steps")
plt.ylabel("% Optimal Action")
plt.title("Non-Stationary 10-Armed Bandit – Tracking Performance")
plt.legend()
plt.grid(alpha=0.3)
plt.ylim(0,100)
plt.show()
