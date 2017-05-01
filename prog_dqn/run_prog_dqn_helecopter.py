import argparse
import random
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

# Hack to import dqn from parallel directory
import os
import sys
from os import path
sys.path.append("../")

import prog_dqn
from dqn_utils import *
from helecopter import helecopter
import prog_nn

Q_FUNC_PARAM_COL_NUM = 'column_number'
CHECKPOINT_BASE_PATH = 'checkpoint_path'
CHECKPOINTS_LIST = 'checkpoints_list'

def helecopter_model(input_shape, num_actions, session, prog_q_func_params, scope, reuse=False):
    column_number = prog_q_func_params[Q_FUNC_PARAM_COL_NUM]
    checkpoint_base_path = prog_q_func_params[CHECKPOINT_BASE_PATH]
    checkpoint_list = prog_q_func_params[CHECKPOINTS_LIST]

    if column_number < 1:
        # Assuming input_shape is length 1 (haven't implemented convolution yet)
        topology1 = [input_shape[0], 128, 128, 64, num_actions]
        activations = [tf.nn.relu, tf.nn.relu, tf.nn.relu, None]

        column = prog_nn.InitialColumnProgNN(topology1, activations, session, checkpoint_base_path, dtype=tf.float32)
    else:
        # TODO: SEPERATE, STORE, AND LOAD FOR EACH INDIVIDUAL COLUMN BEING RESTORED!!!
        topology1 = [input_shape[0], 128, 128, 64, num_actions]
        activations = [tf.nn.relu, tf.nn.relu, tf.nn.relu, None]

        # Restore previous columns here
        prev_columns = []
        for i in range(column_number):
            print("reconstructing column i: ", i)
            if i == 0:
                col_i = prog_nn.InitialColumnProgNN(topology1, activations, session, checkpoint_base_path, dtype=tf.float32)
            else:
                col_i = prog_nn.ExtensibleColumnProgNN(topology1, activations, session, checkpoint_base_path, prev_columns, dtype=tf.float32)

            col_i.restore_weights(checkpoint_list[i])
            prev_columns.append(col_i)
            print("column successfully restored")

        print("previous columns are: ", prev_columns)

        column = prog_nn.ExtensibleColumnProgNN(topology1, activations, session, checkpoint_base_path, prev_columns, dtype=tf.float32)

    # TODO: parametrize topology in prog_q_func_params
    return column

def helecopter_learn(env, session, num_timesteps):
    # This is just a rough estimate
    num_iterations = float(num_timesteps) / 4.0

    # lr_multiplier = 0.1
    lr_multiplier = 1.0
    lr_schedule = PiecewiseSchedule([
                                         (0,                   1e-4 * lr_multiplier),
                                         (num_iterations / 10, 1e-4 * lr_multiplier),
                                         (num_iterations / 2,  5e-5 * lr_multiplier),
                                    ],
                                    outside_value=5e-5 * lr_multiplier)
    optimizer = prog_dqn.OptimizerSpec(
        constructor=tf.train.AdamOptimizer,
        kwargs=dict(epsilon=1e-4),
        lr_schedule=lr_schedule
    )

    def stopping_criterion(env, t):
        return t > 100000000

    exploration_schedule = PiecewiseSchedule(
        [
            (0, 0.2),
            # (0, 0.1),
            ((1e6)/2, 0.01),
            # (1e6, 0.1),
            (num_iterations / 10, 0.01),
            # (num_iterations / 2, 0.01),
        ], outside_value=0.01
    )

    # q_func_params = {
    #     Q_FUNC_PARAM_COL_NUM : 0,
    #     CHECKPOINT_BASE_PATH : 'helecopter_test',
    #     CHECKPOINTS_LIST : [],
    # }
    #
    # # For second column
    # q_func_params = {
    #     Q_FUNC_PARAM_COL_NUM : 1,
    #     CHECKPOINT_BASE_PATH : 'helecopter_test',
    #     CHECKPOINTS_LIST : [65], # Change to latest checkpoint
    # }
    #
    # For third column
    q_func_params = {
        Q_FUNC_PARAM_COL_NUM : 2,
        CHECKPOINT_BASE_PATH : 'helecopter_test',
        CHECKPOINTS_LIST : [65, 153], # Change to latest checkpoints
    }


    prog_dqn.learn(
        env,
        prog_q_func=helecopter_model,
        prog_q_func_params=q_func_params,
        optimizer_spec=optimizer,
        session=session,
        exploration=exploration_schedule,
        stopping_criterion=stopping_criterion,
        replay_buffer_size=1000000,
        batch_size=32,
        gamma=0.99,
        learning_starts=50,
        learning_freq=4,
        frame_history_len=4,
        target_update_freq=10000,
        # target_update_freq=10000,
        grad_norm_clipping=10
    )
    env.close()

def set_global_seeds(i):
    try:
        import tensorflow as tf
    except ImportError:
        pass
    else:
        tf.set_random_seed(i)
    np.random.seed(i)
    random.seed(i)

def get_session():
    tf.reset_default_graph()
    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1)
    session = tf.Session(config=tf_config)
    return session

def get_env(seed):
    # params = { # Col 0 params
    #         'length' : 10,
    #         'height' : 10,
    #         'visible_width' : 10,
    #         'num_column_obstacles' : 0,
    #         'flatten_output' : True,
    #         'padding' : 0,
    # }

    # params = {
    #         'length' : 30,
    #         'height' : 10,
    #         'visible_width' : 10,
    #         'num_column_obstacles' : 1,
    #         'flatten_output' : True,
    #         'padding' : 0,
    # }

    params = {
            'length' : 100,
            'height' : 10,
            'visible_width' : 10,
            'num_column_obstacles' : 15,
            'flatten_output' : True,
            'padding' : 0,
    }
    env = helecopter.HelecopterEnv(params)

    set_global_seeds(seed)
    # env.seed(seed)
    return env

def main():
    # Run training
    seed = 0 # Use a seed of zero (you may want to randomize the seed!)
    env = get_env(seed)
    session = get_session()
    helecopter_learn(env, session, num_timesteps=int(4e7))

if __name__ == "__main__":
    main()
