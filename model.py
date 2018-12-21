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
        for data in data_batch:
            for i in range(len(data.len)):
                loss = torch.nn.MSELoss()
        loss.backward()
        self.optim.step()
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
        for data in data_batch:
            for i in range(len(data.len)):
                loss = torch.nn.MSELoss()
        loss.backward()
        self.optim.step()
        return

class CPCI_Action_30(nn.Module):
    def __init__(self):
        super(CPCI_Action_30, self).__init__()

