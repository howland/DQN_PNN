import sys
import gym.spaces
import itertools
import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.layers as layers
from collections import namedtuple
from dqn_utils import *
import time

OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs", "lr_schedule"])

def learn(env,
          prog_q_func,
          prog_q_func_params,
          optimizer_spec,
          session,
          exploration=LinearSchedule(1000000, 0.1),
          stopping_criterion=None,
          replay_buffer_size=1000000,
          batch_size=32,
          gamma=0.99,
          learning_starts=50000,
          learning_freq=4,
          frame_history_len=4,
          target_update_freq=10000,
          grad_norm_clipping=10):
    """Run Deep Q-learning algorithm.

    You can specify your own convnet using q_func.

    All schedules are w.r.t. total number of steps taken in the environment.

    Parameters
    ----------
    env: gym.Env
        gym environment to train on.
    q_func: function
        Model to use for computing the q function. It should accept the
        following named arguments:
            img_in: tf.Tensor
                tensorflow tensor representing the input image
            num_actions: int
                number of actions
            scope: str
                scope in which all the model related variables
                should be created
            reuse: bool
                whether previously created variables should be reused.
    optimizer_spec: OptimizerSpec
        Specifying the constructor and kwargs, as well as learning rate schedule
        for the optimizer
    session: tf.Session
        tensorflow session to use.
    exploration: rl_algs.deepq.utils.schedules.Schedule
        schedule for probability of chosing random action.
    stopping_criterion: (env, t) -> bool
        should return true when it's ok for the RL algorithm to stop.
        takes in env and the number of steps executed so far.
    replay_buffer_size: int
        How many memories to store in the replay buffer.
    batch_size: int
        How many transitions to sample each time experience is replayed.
    gamma: float
        Discount Factor
    learning_starts: int
        After how many environment steps to start replaying experiences
    learning_freq: int
        How many steps of environment to take between every experience replay
    frame_history_len: int
        How many past frames to include as input to the model.
    target_update_freq: int
        How many experience replay rounds (not steps!) to perform between
        each update to the target Q network
    grad_norm_clipping: float or None
        If not None gradients' norms are clipped to this value.
    """

    input_shape = env.observation_space_shape
    num_actions = env.action_space.shape[0]

    # set up placeholders
    # placeholder for current observation (or state)
    obs_t_ph = tf.placeholder(tf.uint8, [None] + list(input_shape), name="obs_t_ph")
    # placeholder for current action
    act_t_ph = tf.placeholder(tf.int32,   [None], name="act_t_ph")
    # placeholder for current reward
    rew_t_ph = tf.placeholder(tf.float32, [None], name="rew_t_ph")
    # placeholder for next observation (or state)
    obs_tp1_ph = tf.placeholder(tf.uint8, [None] + list(input_shape), name="obs_tp1_ph")
    # placeholder for end of episode mask
    done_mask_ph = tf.placeholder(tf.float32, [None], name="done_mask_ph")

    current_q_out = prog_q_func(input_shape, num_actions, session, prog_q_func_params, scope='current_q_out')
    target_q_out = prog_q_func(input_shape, num_actions, session, prog_q_func_params, scope='target_q_out')

    batch_actions = tf.one_hot(act_t_ph, num_actions)
    batch_qs = tf.reduce_sum(current_q_out.h[-1] * batch_actions, reduction_indices=1)
    batch_target_qs = rew_t_ph + ( gamma * tf.reduce_max(target_q_out.h[-1], reduction_indices=1) *  (1.0 - done_mask_ph))
    total_error = tf.reduce_mean(tf.square(batch_qs - batch_target_qs))

    # q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='current_q_out')
    # target_q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_q_out')
    q_func_vars = current_q_out.pc.params
    target_q_func_vars = target_q_out.pc.params

    # construct optimization op (with gradient clipping)
    learning_rate = tf.placeholder(tf.float32, (), name="learning_rate")
    optimizer = optimizer_spec.constructor(learning_rate=learning_rate, **optimizer_spec.kwargs)
    train_fn = minimize_and_clip(optimizer, total_error,
                 var_list=q_func_vars, clip_val=grad_norm_clipping)

    # update_target_fn will be called periodically to copy Q network to target Q network
    update_target_fn = []

    # TODO: Resort out this issue... not solved (bad names given in network, cant get consistency between old & new qfunc ordering)
    # for var, var_target in zip(sorted(q_func_vars,        key=lambda v: v.name),
                            #    sorted(target_q_func_vars, key=lambda v: v.name)):
    for var, var_target in zip(q_func_vars,target_q_func_vars):
        update_target_fn.append(var_target.assign(var))
    update_target_fn = tf.group(*update_target_fn)

    # construct the replay buffer
    replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len)

    iteration_checkpoints = []
    episode_rewards = []
    mean_100_rewards = []
    best_mean_rewards = []

    model_initialized = False
    num_param_updates = 0
    mean_episode_reward      = -float('nan')
    best_mean_episode_reward = -float('inf')
    last_obs = env.reset()
    total_episode_reward = 0
    LOG_EVERY_N_EPISODES = 50
    SAVE_EVERY_N_EPISODES = 1000
    # DISPLAY_EVERY_N_TRIALS = 1
    DISPLAY_EVERY_N_TRIALS = 2000
    TRIAL_COUNTER = 1
    checkpoint_count = 0

    for t in itertools.count():
        ### 1. Check stopping criterion
        if stopping_criterion is not None and stopping_criterion(env, t):
            break

        last_obs_index = replay_buffer.store_frame(last_obs)
        # Add straight path exploration here. If small rv, then pick random length,
        # then repeat same action for that entire length. After, return to normal sampling.\
        # Hopefully will lead to deeper exploration, rather than the typical random walk behavior
        # that doesn't stray far from the current policy.
        if np.random.random() < exploration.value(t) or not model_initialized:
            action = np.random.choice(env.action_space)
        else:
            replay_buffer_observations = np.expand_dims(replay_buffer.encode_recent_observation(), axis=0)
            query_feed_dict = current_q_out.add_input_to_feed_dict({}, replay_buffer_observations)
            action = np.argmax(current_q_out.session.run([current_q_out.h[-1]], feed_dict=query_feed_dict))

        # last_obs, reward, done, info = env.step(action)
        last_obs, reward, done, _ = env.step(action)
        if TRIAL_COUNTER % DISPLAY_EVERY_N_TRIALS == 0:
            print("Displaying new iteration: ", t)
            print("Observed reward: ", reward, " after action: ", action, " done: ", done)
            env.display_grid()
            time.sleep(0.2)

        total_episode_reward += reward
        replay_buffer.store_effect(last_obs_index, action, reward, done)
        if done:
            episode_rewards.append(total_episode_reward)
            total_episode_reward = 0
            last_obs = env.reset()
            TRIAL_COUNTER += 1

        ### 3. Perform experience replay and train the network.
        # note that this is only done if the replay buffer contains enough samples
        # for us to learn something useful -- until then, the model will not be
        # initialized and random actions should be taken
        if (t > learning_starts and
                t % learning_freq == 0 and
                replay_buffer.can_sample(batch_size)):
            # Use replay buffer to sample batch of transitions
            obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = replay_buffer.sample(batch_size)

            # Preprocess obs_batch
            obs_batch = obs_batch / 3.0
            next_obs_batch = next_obs_batch / 3.0

            if not model_initialized:
                initialize_interdependent_variables(session, tf.global_variables(), {current_q_out.o_n: obs_batch, target_q_out.o_n: next_obs_batch,})
                model_initialized = True

            # Train the model
            train_feed_dict = {
                                act_t_ph : act_batch,
                                rew_t_ph : rew_batch,
                                done_mask_ph : done_mask,
                                learning_rate : optimizer_spec.lr_schedule.value(t)
                                }

            # Add input batch to current q value function (it may need to be passed into multiple columns)
            train_feed_dict = current_q_out.add_input_to_feed_dict(train_feed_dict, obs_batch)
            # Add input batch to target q value function (it may need to be passed into multiple columns)
            train_feed_dict = target_q_out.add_input_to_feed_dict(train_feed_dict, next_obs_batch)

            session.run(train_fn, feed_dict=train_feed_dict)

            # Periodically update target network
            if t % target_update_freq == 0:
                session.run(update_target_fn)
                num_param_updates += 1

        ### 4. Log progress
        # episode_rewards = get_wrapper_by_name(env, "Monitor").get_episode_rewards()
        if len(episode_rewards) > 0:
            mean_episode_reward = np.mean(episode_rewards[-50:])
            # mean_episode_reward = np.mean(episode_rewards[-100:])
        if len(episode_rewards) > 100:
            # print("trying for new best")
            best_mean_episode_reward = max(best_mean_episode_reward, mean_episode_reward)
        if TRIAL_COUNTER % LOG_EVERY_N_EPISODES == 0 and model_initialized and done:
            print("Timestep %d" % (t,))
            print("mean reward (50 episodes) %f" % mean_episode_reward)
            print("best mean reward %f" % best_mean_episode_reward)
            print("episodes %d" % len(episode_rewards))
            print("exploration %f" % exploration.value(t))
            print("learning_rate %f" % optimizer_spec.lr_schedule.value(t))

            iteration_checkpoints.append(t)
            mean_100_rewards.append(mean_episode_reward)
            best_mean_rewards.append(best_mean_episode_reward)

            np.save("run_x_iteration_checkpoints", np.array(iteration_checkpoints))
            np.save("run_x_mean_100_rewards", np.array(mean_100_rewards))
            np.save("run_x_best_mean_rewards", np.array(best_mean_rewards))

            sys.stdout.flush()
        if TRIAL_COUNTER % SAVE_EVERY_N_EPISODES == 0 and model_initialized and done:
            print("Saving model at timestep %d" % (t,))
            current_q_out.save(checkpoint_count)
            checkpoint_count += 1
