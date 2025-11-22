import numpy as np

class NonStationary10ArmedBandit:
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.true_means = np.zeros(10)   # q*(a)

    def step(self, action):
        
        self.true_means += self.rng.normal(0, 0.01, 10)
        reward = self.rng.normal(self.true_means[action], 1.0)
        return reward

    def optimal_action(self):
        return int(np.argmax(self.true_means))

# Demo
if __name__ == "__main__":
    bandit = NonStationary10ArmedBandit()
    print("First 10 steps – optimal arm changes over time:")
    for t in range(10):
        bandit.step(0)  # dummy action just to advance drift
        print(f"Step {t+1:2d} → optimal arm = {bandit.optimal_action()}")
    print("\nEnvironment ready – use in task 4!")
