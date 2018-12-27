import numpy as np
import torch
import torch.nn as nn
from network_modules import *
import torchvision.utils as u
from PIL import Image
# from skimage.viewer import ImageViewer

# TODO: seperate optimizers for eval_MLP

class FP(nn.Module):
    def __init__(self):
        super(FP, self).__init__()
        self.conv = ConvBasic()
        self.belief_gru = beliefGRU()
        self.deconv = BasicDeconv()
        self.eval_mlp = evalMLP()
        self.optim = None
        self.pos_optim = None
    
    def forward(self, o_t, a_tm1, a_t, hidden):
        z_t = self.conv(o_t)
        z_a = torch.cat((z_t, a_tm1), dim=1).unsqueeze(0)
        b_t, hidden = self.belief_gru.gru1(z_a, hidden)
        b_t = b_t.squeeze(0)
        b_a = torch.cat((b_t, a_t), dim=1)
        o_tp1 = self.deconv(b_a)
        x_y_theta = self.eval_mlp(b_t.detach())

        return o_tp1, x_y_theta, hidden

    def update(self, data_batch):
        criterion = torch.nn.MSELoss(size_average=True)
        prediction_loss = 0
        for i, data in enumerate(data_batch):
            obs_batch = np.array(data.new_rgb)
            obs_batch= torch.from_numpy(obs_batch).type('torch.FloatTensor')/255.

            a_batch = np.array(data.action)
            a_batch = torch.from_numpy(a_batch).to(dtype=torch.float32)
            hidden = data.belief
            o_tp1, x_y_theta, hidden = self.forward(obs_batch[:data.len-1], a_batch[:data.len-1], a_batch[1:], hidden)
            target_pos, target_ori = np.array(data.new_pos), np.array(data.new_ori)
            target_pos = torch.from_numpy(target_pos).type('torch.FloatTensor')
            target_ori = torch.from_numpy(target_ori).type('torch.FloatTensor')
            target = torch.cat((target_pos, target_ori), dim=1)
            prediction_loss += criterion(o_tp1, obs_batch[1:])
            eval_loss = criterion(x_y_theta, target[1:])

        prediction_loss /= data_batch.__len__()
        eval_loss /= data_batch.__len__()
        print ("Loss: ", prediction_loss.data, " Eval_Loss: ", eval_loss.data)
        self.optim.zero_grad()
        self.pos_optim.zero_grad()
        prediction_loss.backward(retain_graph=True)
        eval_loss.backward()
        self.optim.step()
        self.pos_optim.step()

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
        self.pos_optim = None
    
    def forward(self, data_batch):
        return

    def update(self, data_batch):
        self.optim.zero_grad()
        obs_batch = np.asarray([np.array(sub_traj.new_rgb) for sub_traj in data_batch])
        obs_batch= torch.from_numpy(obs_batch).to(dtype=torch.float32)
        h_0_batch = [sub_traj.belief for sub_traj in data_batch]
        z_batch = self.conv(obs_batch.view(-1, 3, 84, 84))
        z_batch = z_batch.view(64, 100, 512)
        a_batch = np.asarray([np.array(sub_traj.action) for sub_traj in data_batch])
        a_batch = torch.from_numpy(a_batch).to(dtype=torch.float32)
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
        self.conv = ConvBasic()
        self.belief_gru = beliefGRU()
        self.action_gru = actionGRU()
        cpc_clf = [MLP() for i in range(30)]
        self.mlp = nn.ModuleList(cpc_clf)
        self.eval_mlp = evalMLP()
        self.optim = None
    
    def forward(self, data_batch):
        return

