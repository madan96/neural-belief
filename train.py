from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import deepmind_lab
import copy

from agent import RandomAgent
from collections import deque
from model import FP, CPCI_Action_1
from memory import MemoryBuffer, SubTrajectory
from tqdm import tqdm

import numpy as np
import torch

from itertools import islice

def replay(replay_buffer):
    replay_buffer.sample(64)
    return

def train(env, model, args):
    model.optim = torch.optim.Adam(islice(model.parameters(), 20), lr=0.0005)
    model.pos_optim = torch.optim.Adam(islice(model.parameters(), 20, None), lr=0.0005)
    replay_buffer = MemoryBuffer(int(args.batch))

    agent = RandomAgent(env.action_spec())
    max_steps = args.num_steps
    env.reset()

    step = 0
    sub_trajectory = SubTrajectory(100)

    pbar = tqdm(total = max_steps)

    while step < max_steps:
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
        if step >= args.batch * 100:
            train_batch = replay_buffer.sample(64)
            if None in train_batch:
                raise Exception("Training batch contains None object")
            model.update(train_batch)

        step += 1
        pbar.update(1)
    
    pbar.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=str, default='FP',
                        help='Model')
    parser.add_argument('--width', type=int, default=84,
                        help='Horizontal size of the observations')
    parser.add_argument('--height', type=int, default=84,
                        help='Vertical size of the observations')
    parser.add_argument('--fps', type=int, default=60,
                        help='Number of frames per second')
    parser.add_argument('--batch', type=int, default=64,
                        help='Minibatch size for subtrajectories')

    parser.add_argument('--level', type=str, default='seekavoid_arena_01',
                        help='The environment level script to load')

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
    
    train(env, model, args)