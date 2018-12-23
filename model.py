import numpy as np
import torch
import torch.nn as nn
from network_modules import *

class FP(nn.Module):
    def __init__(self):
        super(FP, self).__init__()
        self.conv = ConvBasic()
        self.gru = beliefGRU()
        self.deconv = BasicDeconv()
        self.mlp = MLP()
        self.optim = None
    
    def update(self, data_batch):
        self.optim.zero_grad()
        obs_batch = np.asarray([np.array(traj_imgs.rgb) for traj_imgs in data_batch])
        # loss = torch.nn.MSELoss()
        # for data in data_batch:
        #     for i in range(len(data.len)):
        #         loss = torch.nn.MSELoss()
        # loss.backward()
        # self.optim.step()
        return

class CPC_1(nn.Module):
    def __init__(self):
        super(CPC_1, self).__init__()

class CPC_30(nn.Module):
    def __init__(self):
        super(CPC_30, self).__init__()

class CPCI_Action_1(nn.Module):
    def __init__(self):
        super(CPCI_Action_1, self).__init__()
        self.conv = ConvBasic()
        self.belief_gru = beliefGRU()
        self.action_gru = actionGRU()
        self.mlp = MLP()
        self.eval_mlp = evalMLP()
        self.optim = None
    
    def update(self, data_batch):
        self.optim.zero_grad()
        obs_batch = np.asarray([np.array(sub_traj.rgb) for sub_traj in data_batch])
        h_0_batch = [sub_traj.belief for sub_traj in data_batch]
        obs_batch= torch.from_numpy(obs_batch).type(torch.FloatTensor)
        z_batch = self.conv(obs_batch.view(-1, 3, 84, 84))
        z_batch = z_batch.view(10, 100, 512)
        a_batch = np.asarray([np.array(sub_traj.action) for sub_traj in data_batch])
        a_batch = torch.from_numpy(a_batch).type(torch.FloatTensor)
        input_batch = torch.cat((z_batch, a_batch), dim=2)
        beliefs = []
        
        """
        * Belief calculation
        * Refer CPCI Algorithm
        * Check loss formulation
        """
        
        for i in range(len(data_batch)):
            h_0 = h_0_batch[i]
            z_a = input_batch[i].unsqueeze(0)
            beliefs.append(self.belief_gru.gru1(z_a, h_0)[0])
        beliefs = torch.stack(beliefs)


        
        # for data in data_batch:
        #     for i in range(len(data.len)):
        #         loss = torch.nn.MSELoss()
        # loss.backward()
        # self.optim.step()
        return

class CPCI_Action_30(nn.Module):
    def __init__(self):
        super(CPCI_Action_30, self).__init__()

