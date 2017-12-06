import numpy as np
import tensorflow as tf
import gym
from utils import *
from Agents.TRPOAgentNew import *
# from Agents.TRPOAgent import *
import argparse
import json
from mpi4py import MPI
import sys

comm = MPI.COMM_WORLD
cpu = MPI.Get_processor_name()
rank = comm.Get_rank()
size = comm.Get_size()
# print("Hello world from processor {}, process {} out of {}".format(cpu,rank,size))
# sys.stdout.flush()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

parser = argparse.ArgumentParser(description='TRPO.')

# these parameters should stay the same
parser.add_argument("--task", type=str, default='SpaceInvaders-ram-v0')
parser.add_argument("--timesteps_per_batch", type=int, default=400)
parser.add_argument("--n_steps", type=int, default=10000000)
parser.add_argument("--n_iter", type=int, default=100)
parser.add_argument("--gamma", type=float, default=.995)
parser.add_argument("--max_kl", type=float, default=.01)
parser.add_argument("--cg_damping", type=float, default=0.1)
parser.add_argument("--num_threads", type=int, default=1)
parser.add_argument("--monitor", type=bool, default=False)

# change these parameters for testing
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
history["mean_reward"] = []
history["timesteps"] = []
history["maxkl"] = []

# start it off with a big negative number
last_reward = -1000000
recent_total_reward = 0

totalsteps = 0

starting_timesteps = args.timesteps_per_batch
starting_kl = args.max_kl

iteration = 0
isDone = 0

logging.getLogger().setLevel(logging.INFO)

while isDone == 0:
    iteration += 1

    # synchronize model and update actor weights locally
    bcast_start = time.time()
    new_policy_weights = comm.bcast(new_policy_weights, root=0)
    learner.set_policy_weights(new_policy_weights)
    bcast_time = (time.time() - bcast_start)

    # start worker processes collect experience for a minimum args.timesteps_per_batch timesteps
    rollout_start = time.time()
    data_paths, data_steps = learner.rollout(args.timesteps_per_batch / size)
    rollout_time = (time.time() - rollout_start)

    # gathering of experience on root process
    gather_start = time.time()
    sendcounts = np.array(comm.gather(data_steps[0], 0))
    if rank == 0:
        paths = np.empty([np.sum(sendcounts), data_paths.shape[1]], dtype=float)
        episodes_rewards = np.empty(2, dtype = np.int)
        sendcounts *= data_paths.shape[1]
    else:
        paths = None
        episodes_rewards = None
    comm.Gatherv(data_paths, [paths, sendcounts], root = 0)
    comm.Reduce(data_steps[1:3], episodes_rewards, op=MPI.SUM, root = 0)
    gather_time = (time.time() - gather_start)

    # only master process does learning on TF graph
    if rank == 0:
        learn_start = time.time()
        learner.adjust_kl(args.max_kl)
        new_policy_weights, stats = learner.learn(paths, episodes_rewards)
        learn_time = (time.time() - learn_start)
        iteration_time = rollout_time + learn_time + gather_time + bcast_time

        print(("\n-------- Iteration %d ----------" % iteration))
        print(("Reward Statistics:"))
        for k, v in stats.items():
            print("\t{} = {:.3f}".format(k,v))
        print(("\nTiming Statistics:"))
        print(("\tBroadcast time = %.3f s" % bcast_time))
        print(("\tRollout time = %.3f s" % rollout_time))
        print(("\tGather time = %.3f s" % gather_time))
        print(("\tLearn time = %.3f s" % learn_time))
        print(("\tTotal iteration time = %.3f s" % (rollout_time + learn_time + gather_time + bcast_time)))

        history["rollout_time"].append(rollout_time)
        history["learn_time"].append(learn_time)
        history["bcast_time"].append(bcast_time)
        history["gather_time"].append(gather_time)
        history["mean_reward"].append(stats["Avg_Reward"])
        history["timesteps"].append(args.timesteps_per_batch)
        history["maxkl"].append(args.max_kl)

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

        if iteration % 100 == 0:
            with open("Results/%s-%s-%f-%f-%f-%f" % (args.task, args.decay_method, starting_timesteps, starting_kl, args.timestep_adapt, args.kl_adapt), "w") as outfile:
                json.dump(history,outfile)

            learner.save_weights("iter_{}.ckpt".format(iteration))

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

    # Check for complete on all processes
    isDone = comm.bcast(isDone, root=0)

if rank == 0:
    print("Evaluation complete!")
