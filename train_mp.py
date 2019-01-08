from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import deepmind_lab
import copy

from agent import RandomAgent
from collections import deque
from model import FP, CPCI_Action_1, CPCI_Action_30
from memory import MemoryBuffer, SubTrajectory, sample_minibatch
from tqdm import tqdm

from setproctitle import setproctitle
from itertools import islice

import numpy as np
import torch
import multiprocessing as mp
from multiprocessing.managers import BaseManager, NamespaceProxy
from multiprocessing import Manager

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))
# mp.set_start_method('spawn')
# mp = mp.get_context('forkserver')
pbar = tqdm(total=int(8e3))

global_step = 1
manager = Manager()
replay_buffer = manager.list()

def run_rollout(model, step, q, args):
    global replay_buffer
    env = deepmind_lab.Lab(args.level, ['RGB', 'DEBUG.POS.TRANS', 'DEBUG.POS.ROT'],
    config={
        'fps': str(args.fps),
        'width': str(args.width),
        'height': str(args.height)
    }, renderer='hardware')
    env.reset()

    agent = RandomAgent(env.action_spec())
    max_steps = args.num_steps
    env.reset()

    sub_trajectory = SubTrajectory(100)

    while step.value < max_steps:
        action = agent.step()
        # for _ in range(np.random.randint(1,5)):
        rgb, pos, orientation = env.observations()['RGB'], env.observations()['DEBUG.POS.TRANS'], env.observations()['DEBUG.POS.ROT']
        reward = env.step(action)
        if (not env.is_running()):
            env.reset()
        else:
            new_rgb, new_pos, new_orientation = env.observations()['RGB'], env.observations()['DEBUG.POS.TRANS'], env.observations()['DEBUG.POS.ROT']
        
        if sub_trajectory.len == 100:
            tmp = copy.deepcopy(sub_trajectory)
            # Send initial belief to replay buffer
            o_0 = torch.from_numpy(tmp.new_rgb[0]).to(dtype=torch.float32).unsqueeze(0)
            a_0 = torch.from_numpy(tmp.action[0]).to(dtype=torch.float32).unsqueeze(0)
            z_0 = model.conv(o_0)
            bgru_input = torch.cat((z_0, a_0), dim=1)
            _, tmp.belief = model.belief_gru.gru1(torch.unsqueeze(bgru_input, 1))
            replay_buffer.append(tmp)
            sub_trajectory.clear()

        sub_trajectory.add(rgb, pos, orientation, action, new_rgb, new_pos, new_orientation)
        step.value += 1
        q.put(global_step)

def train_model(model, device):
    global replay_buffer
    model.to(device)
    print (next(model.conv.parameters()).is_cuda)
    model.pos_optim = torch.optim.Adam(model.eval_mlp.parameters(), lr=0.0005)
    batch = []
    while len(replay_buffer) < 10:
        continue
    while True:
        batch = sample_minibatch(replay_buffer, 10)
        print (len(replay_buffer))
        model.update(batch)

def listener(q, args):
    pbar = tqdm(total = args.num_steps)
    for item in iter(q.get, None):
        pbar.update()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=str, default='CPCI_Action_30',
                        help='Model')
    parser.add_argument('--width', type=int, default=84,
                        help='Horizontal size of the observations')
    parser.add_argument('--height', type=int, default=84,
                        help='Vertical size of the observations')
    parser.add_argument('--fps', type=int, default=60,
                        help='Number of frames per second')
    parser.add_argument('--level', type=str, default='seekavoid_arena_01',
                        help='The environment level script to load')
    parser.add_argument('--batch', type=int, default=64,
                        help='Minibatch size for subtrajectories')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')


    parser.add_argument('--tau', type=int, default=0.99, help='Training hyperparameter')
    parser.add_argument('--gamma', type=int, default=0.99, help='Discounted factor')
    parser.add_argument('--clip-grad-norm', type=int, default=1.0, help='Clip gradient')
    parser.add_argument('--num-steps', type=int, default=int(5e6), help='Number of training episodes')

    """
    Levels to be tested
    -------------------
    fixed: seekavoid_arena_01
    room: contributed/dmlab30/rooms_collect_good_objects_train
    maze: nav_maze_random_goal_01
    terrain: contributed/dmlab30/natlab_fixed_large_map
    """

    # For sharing belief gru across processes, seperate it from model
    # and use share_memory_() for sharing it across processes

    args = parser.parse_args()
    # device = torch.device("cuda:0" if not args.no_cuda and torch.cuda.is_available() else "cpu")
    if args.no_cuda:
        device = "cpu"
    else:
        device = "cuda:0"

    if args.model == 'CPCI_Action_1':
        model = CPCI_Action_1()
    elif args.model == 'FP':
        model = FP()
    elif args.model == 'CPCI_Action_30':
        model = CPCI_Action_30()

    setproctitle('train_mproc [MASTER]')
    model.device = device
    # model.share_memory()
    model.optim = torch.optim.Adam(islice(model.parameters(), 20), lr=0.0001)
    q = mp.Queue()
 
    proc = mp.Process(target=train_model, args=(model, device))
    proc.start()
    proc_bar = mp.Process(target=listener, args=(q, args))
    proc_bar.start()

    model.belief_gru.share_memory()
    step = mp.Value('l', 0)
    workers = []
    workers = [mp.Process(target=run_rollout, args=(model, step, q, args,)) for i in range(3)]

    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()

    proc.join()
    proc_bar.join()