import numpy as np
import helecopter as helecopter
import time


params = {
        'length' : 60,
        'height' : 10,
        'visible_width' : 10,
        'num_column_obstacles' : 5,
        'flatten_output' : True,
        'padding' : 0,
}

helecopter_test = helecopter.HelecopterEnv(params)
helecopter_test.reset()
helecopter_test.display_grid()


tc = 0
rew_total = 0
num_iterations = 10
for i in range(num_iterations):
    helecopter_test.reset()
    done = False
    helecopter_test.display_grid()

    count = 0
    while not done:
        time.sleep(0.5)
        helecopter_test.display_grid()

        count += 1
        obs, rew, done, _ = helecopter_test.step(np.random.randint(0,2))
        rew_total += rew
    tc += count


print(tc / num_iterations)
print(rew_total / num_iterations)

# maze_test.step(3)
# maze_test.display_grid()
