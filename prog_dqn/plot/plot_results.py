import numpy as np
import matplotlib.pyplot as plt

iterations = np.load("run_x_iteration_checkpoints.npy")
best_means = np.load("run_x_best_mean_rewards.npy")
mean_100_rewards = np.load("run_x_mean_100_rewards.npy")

iterations = iterations / 1000.0
# plt.plot(iterations, best_means)
plt.plot(iterations, mean_100_rewards)
# plt.plot(iterations, mean_100_rewards)
# plt.ylabel('Best Mean Reward')
plt.ylabel('Mean 100 Reward')
plt.xlabel('Thousand Time Steps')
plt.title("Best Mean Reward vs. Time for Maze")
plt.show()
