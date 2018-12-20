import numpy as np
import random

class RandomAgent(object):
  """Basic random agent for DeepMind Lab."""
  def __init__(self, action_spec):
    self.action_spec = action_spec
    self.action_count = len(action_spec)

  def step(self):
    """Choose a random amount of a randomly selected action."""
    action_choice = random.randint(0, self.action_count - 1)
    action_amount = random.randint(self.action_spec[action_choice]['min'],
                                   self.action_spec[action_choice]['max'])
    action = np.zeros([self.action_count], dtype=np.intc)
    action[action_choice] = action_amount
    return action

