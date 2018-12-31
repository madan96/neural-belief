import numpy as np
import random
import torch

def sample_negatives(z_batch, n, f, batch_size):
    sample_range = range(0, n) + range(n + 1, batch_size)
    sample_traj_id = random.choice(sample_range)
    obs_id = random.randint(0, 99)
    obs_neg = z_batch[sample_traj_id][obs_id:obs_id+1]
    return obs_neg.expand(f, -1)
