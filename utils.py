import math
import numpy as np
import random
import torch

def sample_negatives(z_batch, n, f, batch_size):
    sample_range = range(0, n) + range(n + 1, batch_size)
    sample_traj_id = random.choice(sample_range)
    obs_id = random.randint(0, 99)
    obs_neg = z_batch[sample_traj_id][obs_id:obs_id+1]
    return obs_neg.expand(f, -1)

def get_pos_grid(position):
    pos_x, pos_y = position[0], position[1]
    pos_grid = np.zeros((9, 10))
    
    # seek_avoid_arena
    x_min, y_min = -560, -760
    x_max, y_max = 1010, 560

    x_len = x_max - x_min
    y_len = y_max - y_min
    offset_x = math.fabs(x_min)
    offset_y = math.fabs(y_min)
    fac_x = x_len / pos_grid.shape[1]
    fac_y = y_len / pos_grid.shape[0]

    grid_x = math.floor((pos_y + offset_y)/fac_y)
    grid_y = math.floor((pos_x + offset_x)/fac_x)

    pos_grid[int(grid_x)][int(grid_y)] = 1

    return torch.from_numpy(pos_grid).to(dtype=torch.float32)
