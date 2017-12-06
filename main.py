import numpy as np
import tensorflow as tf
import gym
from utils import *
from Agents.TRPOAgentNew import *
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
observation_size = int(learner_env.observation_space.shape[0])
learner = TRPO(args, learner_env)
if rank == 0:
    # starting_weights = learner.get_starting_weights()
    new_policy_weights = learner.get_starting_weights()
else:
    # starting_weights = None
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
while True:
    iteration += 1

    # synchronize model and update actor weights locally
    bcast_start = time.time()
    new_policy_weights = comm.bcast(new_policy_weights, root=0)
    learner.set_policy_weights(new_policy_weights)

    bcast_time = (time.time() - bcast_start)

    # start worker processes collect experience for a minimum args.timesteps_per_batch timesteps
    rollout_start = time.time()

    data = learner.rollout(args.timesteps_per_batch / size)
    rollout_time = (time.time() - rollout_start)

    collect_time_start = time.time()
    num_steps = learner.find_timesteps()
    total_steps =comm.reduce(num_steps, root=0)
    collect_time_end = time.time() - collect_time_start


    # synchronization of experience
    gather_start = time.time()
    agg_data = None
    paths = []
    if rank == 0:
        agg_data = np.zeros((int(total_steps), observation_size + 6))
    comm.Gather(data, agg_data, root=0)

    gather_time = (time.time() - gather_start)

    print(data.shape)
    # only master process does learning on TF graph
    if rank == 0:
        learn_start = time.time()
        learner.adjust_kl(args.max_kl)
        new_policy_weights, stats = learner.update(paths)
        learn_time = (time.time() - learn_start)

        print(("\n-------- Iteration %d ----------" % iteration))
        for k, v in stats.items():
            print("{} = {:.3e}".format(k,v))
        print(("Iteration time = %.3f s" % (rollout_time + learn_time + gather_time + bcast_time)))
        print(("    Broadcast time = %.3f s" % bcast_time))
        print(("    Rollout time = %.3f s" % rollout_time))
        print(("    Gather time = %.3f s" % gather_time))
        print(("    Learn time = %.3f s" % learn_time))

        history["rollout_time"].append(rollout_time)
        history["learn_time"].append(learn_time)
        history["bcast_time"].append(bcast_time)
        history["gather_time"].append(gather_time)
        history["mean_reward"].append(stats["Average sum of rewards per episode"])
        history["timesteps"].append(args.timesteps_per_batch)
        history["maxkl"].append(args.max_kl)

        recent_total_reward += stats["Average sum of rewards per episode"]

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
            with open("%s-%s-%f-%f-%f-%f" % (args.task, args.decay_method, starting_timesteps, starting_kl, args.timestep_adapt, args.kl_adapt), "w") as outfile:
                json.dump(history,outfile)

        totalsteps += stats["Timesteps"]
        print(("%d total steps have happened (Elapsed time = %.3f s)" % (totalsteps,time.time() - start_time)))
        sys.stdout.flush()

        if iteration >= args.n_iter or totalsteps >= args.n_steps:
            break
    else:
        new_policy_weights = None
        totalsteps += args.timesteps_per_batch
        if iteration >= args.n_iter or totalsteps >= args.n_steps:
            break

if rank == 0:
    print("Evaluation complete!")