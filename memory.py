import numpy as np
import random
from collections import deque


class MemoryBuffer:

	def __init__(self, size):
		self.buffer = deque(maxlen=size)
		self.maxSize = size
		self.len = 0

	def sample(self, count):
		"""
		samples a random batch from the replay memory buffer
		:param count: batch size
		:return: batch (numpy array)
		"""
		batch = []
		count = min(count, self.len)
		batch = random.sample(self.buffer, count)
		rgb, new_rgb = np.empty((1,3,84,84)), np.empty((1,3,84,84))
		pos, new_pos = np.empty((1,3)), np.empty((1,3))
		ori, new_ori = np.empty((1,3)), np.empty((1,3))
		action = np.empty((1,7))
		for sub_traj in batch:
			rgb = np.concatenate((rgb, sub_traj.rgb))
			pos = np.concatenate((pos, sub_traj.pos))
			ori = np.concatenate((ori, sub_traj.ori))
			new_rgb = np.concatenate((new_rgb, sub_traj.new_rgb))
			new_pos = np.concatenate((new_pos, sub_traj.new_pos))
			new_ori = np.concatenate((new_ori, sub_traj.new_ori))
			action = np.concatenate((action, sub_traj.action))
				
		return rgb, pos, ori, action, new_rgb, new_pos, new_ori

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
