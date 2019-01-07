import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from network_modules import *
import torchvision.utils as u
import utils
from PIL import Image
# from skimage.viewer import ImageViewer

# TODO: evalMLP should return softmax(x,y,theta),
# grid of sigmoids of past (x, y, theta) and obj-
# grid of sigmoids of obj (x, y, theta)

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
        """
        TODO:
        * Belief calculation
        * Refer CPCI Algorithm
        * Check loss formulation
        * Sample negative examples for discriminator
        """
        z_batch = []
        beliefs = []
        for i, data in enumerate(data_batch):
            hidden = data.belief
            obs_batch = np.array(data.new_rgb)
            obs_batch= torch.from_numpy(obs_batch).type('torch.FloatTensor')/255.
            a_batch = np.array(data.action)
            a_batch = torch.from_numpy(a_batch).to(dtype=torch.float32)
            z_t = self.conv(obs_batch)
            z_a = torch.cat((z_t, a_tm1), dim=1).unsqueeze(0)
            b_t, _ = self.belief_gru.gru1(z_a, hidden)

            z_batch.append(z_t)
            beliefs.append(b_t.squeeze(0))
        z_batch = torch.stack(z_batch)
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
        # TODO: Use env grid size to initialize evalMLP input size
        self.eval_mlp = evalMLP()
        self.optim = None
        self.pos_optim = None
    
    def forward(self, b_t, a_t, z_tp1_pos, z_tp1_neg, init_pos_ori, f):
        a_gru, _ = self.action_gru.gru1(a_t.unsqueeze(0), b_t.unsqueeze(0))
        z_a_gru_pos = torch.cat((z_tp1_pos, a_gru[0]), dim=1)
        pred_positive = torch.stack([self.mlp[i](z_a_gru_pos[i].unsqueeze(0)) for i in range(f)], 1)
        z_a_gru_neg = torch.cat((z_tp1_neg, a_gru[0]), dim=1)
        pred_negative = torch.stack([self.mlp[i](z_a_gru_neg[i].unsqueeze(0)) for i in range(f)], 1)
        eval_input = torch.cat((b_t.detach(), init_pos_ori), dim=1)
        pred_xytheta = self.eval_mlp(eval_input)
        
        return pred_positive[0], pred_negative[0], pred_xytheta

    def update(self, data_batch):
        z_batch = []
        beliefs = []
        loss_cr = nn.BCELoss()
        loss_eval = nn.BCELoss()
        for i, data in enumerate(data_batch):
            hidden = data.belief
            obs_batch = np.array(data.new_rgb)
            obs_batch= torch.from_numpy(obs_batch).type('torch.FloatTensor')/255.
            a_batch = np.array(data.action)
            a_batch = torch.from_numpy(a_batch).to(dtype=torch.float32)
            z_t = self.conv(obs_batch)
            z_a = torch.cat((z_t, a_batch), dim=1).unsqueeze(0)
            b_t, _ = self.belief_gru.gru1(z_a, hidden)

            z_batch.append(z_t)
            beliefs.append(b_t.squeeze(0))
        z_batch = torch.stack(z_batch)
        beliefs = torch.stack(beliefs)

        loss_pos, loss_neg = 0, 0
        for i, data in enumerate(data_batch):
            a_batch = np.array(data.action)
            a_batch = torch.from_numpy(a_batch).to(dtype=torch.float32)

            init_pos = utils.get_pos_grid(data.pos[0][:2])
            init_pos = init_pos.view(-1, 90)
            init_ori = utils.discretize_orientation(data.ori[0][1])
            init_pos_ori = torch.cat((init_pos, init_ori), dim=1)
            loss_evalmlp = 0
            for j in range(data.len - 30):
                f = random.randint(1,30)
                z_batch_neg = utils.sample_negatives(z_batch, i, f, len(data_batch))
                curr_pos_gt = utils.get_pos_grid(data.new_pos[j][:2]).view(-1, 90)
                curr_ori_gt = utils.discretize_orientation(data.new_ori[j][1])
                curr_pos_ori_gt = torch.cat((curr_pos_gt, curr_ori_gt), dim=1)
                pred_positive, pred_negative, pred_xytheta = self.forward(beliefs[i][j:j+1], a_batch[j+1:j+f+1], z_batch[i][j+1:j+f+1], z_batch_neg, init_pos_ori, f)

                loss_pos += loss_cr(torch.sigmoid(pred_positive.squeeze(0)), torch.ones((f, 1)))/70
                loss_neg += loss_cr(torch.sigmoid(pred_negative), torch.zeros((f, 1)))/70
                loss_evalmlp += loss_eval(pred_xytheta, curr_pos_ori_gt)
            loss_evalmlp = loss_evalmlp/(data.len - 30)
            self.pos_optim.zero_grad()
            loss_evalmlp.backward()
            self.pos_optim.step()
        loss = (loss_pos + loss_neg)/(2*len(data_batch))
        print ("Loss: ", loss.data, " Eval loss: ", loss_evalmlp.data)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        
