#!/usr/bin/env python3
import collections
import copy
import time
import numpy as np

import torch
import torch.nn as nn
import torch.multiprocessing as mp

from tensorboardX import SummaryWriter

from lib import wrappers
from lib import dqn_model


MAX_REWARD_BOUND = 21.0
NOISE_STD = 0.004
POPULATION_SIZE = 800
PARENTS_COUNT = 8
WORKERS_COUNT = 2
SEEDS_PER_WORKER = POPULATION_SIZE // WORKERS_COUNT
MAX_SEED = 2**32 - 1


def evaluate(env, net):
    state = env.reset()
    total_reward = 0.0
    steps = 0

    while True:
        state_a = np.array([state], copy=False)
        state_v = torch.tensor(state_a)
        q_vals_v = net(state_v)
        _, act_v = torch.max(q_vals_v, dim=1)
        action = int(act_v.item())

        state, reward, done, _ = env.step(action)
        total_reward += reward
        steps += 1

        if done:
            break
    
    return total_reward, steps


def mutate_net(net, seed, copy_net=True):
    new_net = copy.deepcopy(net) if copy_net else net
    np.random.seed(seed)
    for p in new_net.parameters():
        noise = np.random.normal(size=p.data.size())
        noise_t = torch.FloatTensor(noise)
        p.data += NOISE_STD * noise_t
    return new_net


def build_net(env, seeds):
    torch.manual_seed(seeds[0])
    net = dqn_model.DQN(env.observation_space.shape,
              env.action_space.n)
    for seed in seeds[1:]:
        net = mutate_net(net, seed, copy_net=False)
    return net


OutputItem = collections.namedtuple(
    'OutputItem', field_names=['seeds', 'total_reward', 'steps'])


def worker_func(input_queue, output_queue):
    env = wrappers.make_env("PongNoFrameskip-v4")
    cache = {}

    while True:
        parents = input_queue.get()

        if parents is None:
            break

        new_cache = {}

        for net_seeds in parents:
            if len(net_seeds) > 1:
                net = cache.get(net_seeds[:-1])
                if net is not None:
                    net = mutate_net(net, net_seeds[-1])
                else:
                    net = build_net(env, net_seeds)
            else:
                net = build_net(env, net_seeds)
            new_cache[net_seeds] = net
            total_reward, steps = evaluate(env, net)
            output_queue.put(OutputItem(
                seeds=net_seeds, total_reward=total_reward, steps=steps))
        cache = new_cache


if __name__ == "__main__":
    print("Starting training...")
    mp.set_start_method('spawn')
    writer = SummaryWriter(comment="-dqn-pong-ga")

    gen_idx = 0
    elite = None
    input_queues = []
    output_queue = mp.Queue(maxsize=WORKERS_COUNT)
    workers = []

    for _ in range(WORKERS_COUNT):
        input_queue = mp.Queue(maxsize=1)
        input_queues.append(input_queue)
        w = mp.Process(target=worker_func, args=(input_queue, output_queue))
        w.start()
        workers.append(w)
        seeds = [(np.random.randint(MAX_SEED),) for _ in range(SEEDS_PER_WORKER)]
        input_queue.put((seeds))

    while True:
        t_start = time.time()
        batch_steps = 0
        population = []

        while len(population) < SEEDS_PER_WORKER * WORKERS_COUNT:
            out_item = output_queue.get()
            population.append((out_item.seeds, out_item.total_reward))
            batch_steps += out_item.steps

        if elite is not None:
            population.append(elite)

        population.sort(key=lambda p: p[1], reverse=True)  # sort by total_reward
        rewards = [p[1] for p in population[:PARENTS_COUNT]]  # display total_reward
        reward_mean = np.mean(rewards)
        reward_max = np.max(rewards)
        reward_std = np.std(rewards)
        writer.add_scalar("reward_mean", reward_mean, gen_idx)
        writer.add_scalar("reward_std", reward_std, gen_idx)
        writer.add_scalar("reward_max", reward_max, gen_idx)
        writer.add_scalar("batch_steps", batch_steps, gen_idx)
        writer.add_scalar("gen_seconds", time.time() - t_start, gen_idx)
        speed = batch_steps / (time.time() - t_start)
        writer.add_scalar("speed", speed, gen_idx)
        print("%d: reward_mean=%.2f, reward_max=%.2f, reward_std=%.2f, speed=%.2f f/s" % (
            gen_idx, reward_mean, reward_max, reward_std, speed))

        elite = population[0]

        if reward_max >= MAX_REWARD_BOUND:
            print("Solved in %d generations!" % gen_idx)
            torch.save(elite, "elite.pt")
            break

        gen_idx += 1

        for worker_queue in input_queues:
            seeds = []
            for _ in range(SEEDS_PER_WORKER):
                parent = np.random.randint(PARENTS_COUNT)
                next_seed = np.random.randint(MAX_SEED)
                s = list(population[parent][0]) + [next_seed]
                seeds.append(tuple(s))
            worker_queue.put((seeds))

    # clean up processes
    writer.close()
    output_queue.close()
    for input_queue in input_queues:
        input_queue.put(None)
        input_queue.close()
    
    for worker in workers:
        worker.join()
