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

class CPC_1(nn.Module):
    def __init__(self):
        super(CPC_1, self).__init__()

class CPC_30(nn.Module):
    def __init__(self):
        super(CPC_30, self).__init__()

class CPCI_Action_1(nn.Module):
    def __init__(self):
        super(CPCI_Action_1, self).__init__()

class CPCI_Action_30(nn.Module):
    def __init__(self):
        super(CPCI_Action_30, self).__init__()

