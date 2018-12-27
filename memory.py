import numpy as np
import random
from collections import deque
import warnings

class RingBuffer(object):
    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.start = 0
        self.length = 0
        self.data = [None for _ in range(maxlen)]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise KeyError()
        return self.data[(self.start + idx) % self.maxlen]

    def append(self, v):
        if self.length < self.maxlen:
            # We have space, simply increase the length.
            self.length += 1
        elif self.length == self.maxlen:
            # No space, "remove" the first item.
            self.start = (self.start + 1) % self.maxlen
        else:
            # This should never happen.
            raise RuntimeError()
        self.data[(self.start + self.length - 1) % self.maxlen] = v

class MemoryBuffer:

	def __init__(self, size):
		self.buffer = RingBuffer(size)
		self.maxSize = size
		self.len = 0

	def sample(self, count):
		"""
		samples a random batch from the replay memory buffer
		:param count: batch size
		:return: batch (numpy array)
		"""
		# Start sampling after batches are >= mini-batch size
		# For this we need to sample from 0:data_added
		batch = []
		count = min(count, self.len)
		batch = random.sample(self.buffer.data[0:self.buffer.length], count)
		return batch		

	def len(self):
		return self.len

	def add(self, sub_trajectory):
		"""
		adds a particular transaction in the memory buffer
		:param s: current state
		:param a: action taken
		:param r: reward received
		:param s1: next state
		:return:
		"""

		self.len += 1
		if self.len > self.maxSize:
			self.len = self.maxSize
		self.buffer.append(sub_trajectory)

class SubTrajectory(object):
	def __init__(self, size):
		self.rgb = deque(maxlen=size)
		self.pos = deque(maxlen=size)
		self.ori = deque(maxlen=size)
		self.action = deque(maxlen=size)
		self.new_rgb = deque(maxlen=size)
		self.new_pos = deque(maxlen=size)
		self.new_ori = deque(maxlen=size)
		self.belief = None
		self.maxSize = size
		self.len = 0
	
	def len(self):
		return self.len
	
	def add(self, rgb, pos, ori, a, new_rgb, new_pos, new_ori):
		self.len += 1
		self.rgb.append(rgb)
		self.pos.append(pos)
		self.ori.append(ori)
		self.action.append(a)
		self.new_rgb.append(new_rgb)
		self.new_pos.append(new_pos)
		self.new_ori.append(new_ori)
	
	def clear(self):
		self.rgb.clear(), self.pos.clear()
		self.ori.clear(), self.action.clear()
		self.new_rgb.clear(), self.new_pos.clear()
		self.new_ori.clear()
		self.len = 0
