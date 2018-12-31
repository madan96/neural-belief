from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import deepmind_lab
import copy

from agent import RandomAgent
from collections import deque
from model import FP, CPCI_Action_1, CPCI_Action_30
from memory import MemoryBuffer, SubTrajectory
from tqdm import tqdm

from setproctitle import setproctitle
from itertools import islice
from multi_tqdm import parallel_process

import numpy as np
import torch
import torch.multiprocessing as mp

replay_buffer = MemoryBuffer(64)
# pbar = tqdm(total=int(8e3))

global_step = 1

def train(env, model, step, q, args):
    global replay_buffer
    model.optim = torch.optim.Adam(islice(model.parameters(), 20), lr=0.001)
    model.pos_optim = torch.optim.Adam(islice(model.parameters(), 20, None), lr=0.0005)
    replay_buffer = MemoryBuffer(int(args.batch))

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
            replay_buffer.add(tmp)
            sub_trajectory.clear()

        sub_trajectory.add(rgb, pos, orientation, action, new_rgb, new_pos, new_orientation)

        # Train using replay_buffer
        if step.value >= args.batch * 100:
            train_batch = replay_buffer.sample(64)
            if None in train_batch:
                raise Exception("Training batch contains None object")
            model.update(train_batch)

        step.value += 1
        q.put(global_step)

def listener(q):
    pbar = tqdm(total = int(8e3))
    for item in iter(q.get, None):
        pbar.update()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=str, default='CPCI_Action_1',
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

    parser.add_argument('--tau', type=int, default=0.99, help='Training hyperparameter')
    parser.add_argument('--gamma', type=int, default=0.99, help='Discounted factor')
    parser.add_argument('--clip-grad-norm', type=int, default=1.0, help='Clip gradient')
    parser.add_argument('--num-steps', type=int, default=int(8e3), help='Number of training episodes')

    """
    Levels to be tested
    -------------------
    fixed: seekavoid_arena_01
    room: contributed/dmlab30/rooms_collect_good_objects_train
    maze: nav_maze_random_goal_01
    terrain: contributed/dmlab30/natlab_fixed_large_map
    """

    args = parser.parse_args()
    
    env = deepmind_lab.Lab(args.level, ['RGB', 'DEBUG.POS.TRANS', 'DEBUG.POS.ROT'],
    config={
        'fps': str(args.fps),
        'width': str(args.width),
        'height': str(args.height)
    }, renderer='hardware')
    env.reset()

    if args.model == 'CPCI_Action_1':
        model = CPCI_Action_1()
    elif args.model == 'FP':
        model = FP()
    elif args.model == 'CPCI_Action_30':
        model = CPCI_Action_30()

    setproctitle('train_mproc [MASTER]')
    
    model.share_memory()
    model.optim = torch.optim.Adam(islice(model.parameters(), 20), lr=0.0001)
    q = mp.Queue()
 
    proc = mp.Process(target=listener, args=(q,))
    proc.start()
    
    step = mp.Value('l', 0)
    workers = []
    workers = [mp.Process(target=train, args=(env, model, step, q, args,)) for i in range(32)]

    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    proc.join()
    # q = mp.Queue(10)

    # arr = [{"env": env, "model":model, "args":args} for i in range(int(8e3))]
    # p = parallel_process(arr, train, use_kwargs=True)
    # pool = mp.Pool(16)

    # pool.apply_async(train, args=(env, model, step, args,), callback=update)
