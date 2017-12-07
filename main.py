import numpy as np
import tensorflow as tf
import gym
from utils import *
import os
import time
from Agents.TRPOAgentDiscrete import TRPO as TRPOD
from Agents.TRPOAgent  import TRPO
import argparse
import logging
import json
from mpi4py import MPI
import sys

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
comm_size = comm.Get_size()
# cpu = MPI.Get_processor_name()
# print("Hello world from processor {}, process {} out of {}".format(cpu,rank,comm_size))
# sys.stdout.flush()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
RESULTS_DIR = os.path.join(os.getcwd(), 'Results')
if not os.path.exists(RESULTS_DIR):
    os.mkdir(RESULTS_DIR)

parser = argparse.ArgumentParser(description='TRPO.')
parser.add_argument("--task", type=str, default='SpaceInvaders-ram-v0')
parser.add_argument("--timesteps_per_batch", type=int, default=40000)
parser.add_argument("--n_steps", type=int, default=1000000000)
parser.add_argument("--n_iter", type=int, default=100)
parser.add_argument("--gamma", type=float, default=.995)
parser.add_argument("--max_kl", type=float, default=.01)
parser.add_argument("--cg_damping", type=float, default=0.1)
parser.add_argument("--monitor", type=bool, default=False)
parser.add_argument("--parallel_balancing", type=str, default="timesteps") # timesteps, episodes
parser.add_argument("--discrete", type=bool, default=True)

# change these parameters for hyperparameter adaptation (kvfrans)
parser.add_argument("--decay_method", type=str, default="none") # adaptive, none
parser.add_argument("--timestep_adapt", type=int, default=0)
parser.add_argument("--kl_adapt", type=float, default=0)

args = parser.parse_args()
args.max_pathlength = gym.spec(args.task).timestep_limit
if rank == 0:
    print(args)
    sys.stdout.flush()

# initialize TRPO learner on all processes, distribute the starting weights
learner_env = gym.make(args.task)

if args.discrete:
    learner = TRPOD(args, learner_env)
else:
    learner = TRPO(args, learner_env)
if rank == 0:
    # statbar = tf.contrib.keras.utils.Progbar(args.n_iter )
    new_policy_weights = learner.get_starting_weights()
else:
    new_policy_weights = None

start_time = time.time()
history = {}
history["rollout_time"] = []
history["learn_time"] = []
history["bcast_time"] = []
history["gather_time"] = []
history["iteration_time"] = []
history["mean_reward"] = []
history["timesteps"] = []
history["maxkl"] = []
history["episodes"] = []

# start it off with a big negative number
last_reward = -1000000
recent_total_reward = 0

totalsteps = 0

starting_timesteps = args.timesteps_per_batch
starting_kl = args.max_kl

iteration = 0
isDone = 0

logging.getLogger().setLevel(logging.WARNING)

while isDone == 0:
    iteration += 1

    # synchronize model and update actor weights locally
    bcast_start = time.time()
    new_policy_weights = comm.bcast(new_policy_weights, root=0)
    learner.set_policy_weights(new_policy_weights)
    bcast_time = (time.time() - bcast_start)

    # start worker processes collect experience for a minimum args.timesteps_per_batch timesteps
    rollout_start = time.time()
    data_paths, data_rewards = learner.rollout(args.timesteps_per_batch / comm_size)
    rollout_time = (time.time() - rollout_start)

    # gathering of experience on root process
    gather_start = time.time()
    paths, episodes_rewards = gather_paths(data_paths, data_rewards, comm, rank, args.parallel_balancing)
    gather_time = (time.time() - gather_start)

    # only master process does learning on TF graph
    if rank == 0:
        learn_start = time.time()
        if args.decay_method != "none":
            learner.adjust_kl(args.max_kl)
        new_policy_weights, stats = learner.learn(paths, episodes_rewards)
        learn_time = (time.time() - learn_start)
        iteration_time = rollout_time + learn_time + gather_time + bcast_time

        print(("\n-------- Iteration %d ----------" % iteration))
        print(("Reward Statistics:"))
        for k, v in stats.items():
            print("\t{} = {:.3f}".format(k,v))
        print(("Timing Statistics:"))
        print(("\tBroadcast time = %.3f s" % bcast_time))
        print(("\tRollout time = %.3f s" % rollout_time))
        print(("\tGather time = %.3f s" % gather_time))
        print(("\tLearn time = %.3f s" % learn_time))
        print(("\tTotal iteration time = %.3f s" % (rollout_time + learn_time + gather_time + bcast_time)))

        history["rollout_time"].append(rollout_time)
        history["learn_time"].append(learn_time)
        history["bcast_time"].append(bcast_time)
        history["gather_time"].append(gather_time)
        history["iteration_time"].append(rollout_time + learn_time + gather_time + bcast_time)
        history["mean_reward"].append(stats["Avg_Reward"])
        history["timesteps"].append(args.timesteps_per_batch)
        history["maxkl"].append(args.max_kl)
        history["episodes"].append(stats['Episodes'])

        # compute 100 episode average reward
        ep = 0
        it = iteration-1
        rew = 0
        while ep < 100 and it >= 0:
            ep += history['episodes'][it]
            rew += history['mean_reward'][it]*history['episodes'][it]
            it -= 1
        print(("Cumulative Reward Statistics:"))
        print(("\tMaximum Avg_reward = %.3f from iteration %d" % (np.max(history["mean_reward"]), 1+np.argmax(history["mean_reward"]))))
        print(("\tLast %d Episode Avg_reward = %.3f" % (ep, (rew / ep))))

        print(("Cumulative Mean Timing Statistics:"))
        print(("\tBroadcast time = %.3f s" % np.mean(history["bcast_time"])))
        print(("\tRollout time = %.3f s" % np.mean(history["rollout_time"])))
        print(("\tGather time = %.3f s" % np.mean(history["gather_time"])))
        print(("\tLearn time = %.3f s" % np.mean(history["learn_time"])))
        print(("\tTotal iteration time = %.3f s" % np.mean(history["iteration_time"])))

        # hyperparameter adaptation (kvfrans)
        recent_total_reward += stats["Avg_Reward"]
        if args.decay_method == "adaptive":
            if iteration % 10 == 0:
                if recent_total_reward < last_reward:
                    print("Policy is not improving. Decrease KL and increase steps.")
                    if args.timesteps_per_batch < 20000:
                        args.timesteps_per_batch += args.timestep_adapt
                    if args.max_kl > 0.001:
                        args.max_kl -= args.kl_adapt
                else:
                    print("Policy is improving. Increase KL and decrease steps.")
                    if args.timesteps_per_batch > 1200:
                        args.timesteps_per_batch -= args.timestep_adapt
                    if args.max_kl < 0.01:
                        args.max_kl += args.kl_adapt
                last_reward = recent_total_reward
                recent_total_reward = 0
        if args.decay_method == "adaptive-margin":
            if iteration % 10 == 0:
                scaled_last = last_reward + abs(last_reward * 0.05)
                print(("Last reward: %f Scaled: %f Recent: %f" % (last_reward, scaled_last, recent_total_reward)))
                if recent_total_reward < scaled_last:
                    print("Policy is not improving. Decrease KL and increase steps.")
                    if args.timesteps_per_batch < 10000:
                        args.timesteps_per_batch += args.timestep_adapt
                    if args.max_kl > 0.001:
                        args.max_kl -= args.kl_adapt
                else:
                    print("Policy is improving. Increase KL and decrease steps.")
                    if args.timesteps_per_batch > 1200:
                        args.timesteps_per_batch -= args.timestep_adapt
                    if args.max_kl < 0.01:
                        args.max_kl += args.kl_adapt
                last_reward = recent_total_reward
                recent_total_reward = 0
        # print(("Current step number is " + str(args.timesteps_per_batch) + " and KL is " + str(args.max_kl)))

        if iteration % 10 == 0:
            with open("Results/%s-%d-%f-%d" % (args.task, starting_timesteps, starting_kl, comm_size), "w") as outfile:
                json.dump(history,outfile)
            learner.save_weights("{}-{}-{}-{}_{}.ckpt".format(args.task, starting_timesteps, starting_kl, comm_size, iteration))

        # statbar.add(1, [('Iteration Time',iteration_time ), ("Brodcast Time", bcast_start),
        #                  ("Rollout time", rollout_time), ("Gather Time", gather_time),
        #                  ("Learn time", learn_time)] + list(stats.items()))

        totalsteps += stats["Timesteps"]
        print(("%d total steps have happened (Elapsed time = %.3f s)" % (totalsteps,time.time() - start_time)))
        sys.stdout.flush()
        if iteration >= args.n_iter or totalsteps >= args.n_steps:
            isDone = 1
    else:
        new_policy_weights = None

    isDone = comm.bcast(isDone, root=0)

if rank == 0:
    print(("\n----- Evaluation complete! -----"))
