import numpy as np
import helicopter as helicopter
import time


params = {
        'length' : 60,
        'height' : 10,
        'visible_width' : 10,
        'num_column_obstacles' : 5,
        'flatten_output' : True,
        'padding' : 0,
        'num_positive_pellets' : 100,
        'num_negative_pellets' : 100,
        'num_random_obstacles' : 0,
}

helicopter_test = helicopter.HelicopterEnv(params)
helicopter_test.reset()
helicopter_test.display_grid()


tc = 0
rew_total = 0
num_iterations = 10
for i in range(num_iterations):
    helicopter_test.reset()
    done = False
    helicopter_test.display_grid()

    count = 0
    while not done:
        time.sleep(0.5)
        helicopter_test.display_grid()

        count += 1
        obs, rew, done, _ = helicopter_test.step(np.random.randint(0,2))
        rew_total += rew
    tc += count


print(tc / num_iterations)
print(rew_total / num_iterations)

# maze_test.step(3)
# maze_test.display_grid()
