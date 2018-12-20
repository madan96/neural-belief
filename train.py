from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import deepmind_lab

from agent import RandomAgent
from collections import deque
from model import FP
from memory import MemoryBuffer, SubTrajectory
from tqdm import tqdm

import numpy as np
import torch

def train_FP(env, args):
    model = FP()
    optim = torch.optim.Adam(model.parameters(), lr=0.0005)
    replay_buffer = MemoryBuffer(int(5e4))

    agent = RandomAgent(env.action_spec())
    max_steps = int(5e6)
    env.reset()

    i = 0
    sub_trajectory = SubTrajectory(100)

    pbar = tqdm(total = max_steps)

    while i < max_steps:
        action = agent.step()
        for _ in range(np.random.randint(1,5)):
            rgb, pos, orientation = env.observations()['RGB'], env.observations()['DEBUG.POS.TRANS'], env.observations()['DEBUG.POS.ROT']
            reward = env.step(action)
            if (not env.is_running()):
                env.reset()
            else:
                new_rgb, new_pos, new_orientation = env.observations()['RGB'], env.observations()['DEBUG.POS.TRANS'], env.observations()['DEBUG.POS.ROT']
            
            if sub_trajectory.len > 100:
                replay_buffer.add(sub_trajectory)
                sub_trajectory.clear()

            sub_trajectory.add(rgb, pos, orientation, action, new_rgb, new_pos, new_orientation)

            # Train using replay_buffer
            replay_buffer.sample(64)
            i += 1
            pbar.update(1)
    
    pbar.close()

    
    
    print ("Episodes ran: ", episode, "Sub trajectories: ", count)
        
    """
    * Implement replay buffer
        * Store sub-trajectory as one observation. Each sub-trajectory is 100 steps.
        * Buffer size is 5e4.
        * Class sub-trajectory to make a single sub-trajectory
        * Class replay-buffer (FIFO) to store sub-trajectory objects
        * Sample a mini-batch of 60 sub-trajectories and return the batch 
    """
    


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
    parser.add_argument('--level', type=str, default='seekavoid_arena_01',
                        help='The environment level script to load')

    parser.add_argument('--tau', type=int, default=0.99, help='Training hyperparameter')
    parser.add_argument('--gamma', type=int, default=0.99, help='Discounted factor')
    parser.add_argument('--clip-grad-norm', type=int, default=1.0, help='Clip gradient')
    parser.add_argument('--num-episodes', type=int, default=100, help='Number of training episodes')

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

    # print (env.observations())


    if args.model == 'FP':
        train_FP(env, args)