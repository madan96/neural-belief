import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBasic(nn.Module):
    def __init__(self):
        super(ConvBasic, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.hidden1 = nn.Linear(3136, 512)
        self.relu4 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu1(self.conv1(x))
        out = self.relu2(self.conv2(out))
        out = self.relu3(self.conv3(out))
        flat_out = out.view(out.size(0), -1)
        out = self.relu4(self.hidden1(flat_out))
        return out

class beliefGRU(nn.Module):
    def __init__(self):
        super(beliefGRU, self).__init__()
        # Check input size
        self.hidden1 = nn.Linear(516, 512)
    
    def forward(self, x):
        out = self.hidden1(x)
        return out

class actionGRU(nn.Module):
    def __init__(self):
        super(actionGRU, self).__init__()
        # Check input size
        self.hidden1 = nn.Linear(512, 512)
    
    def forward(self, x):
        out = self.hidden1(x)
        return out

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        # Check input size
        self.hidden1 = nn.Linear(1024, 512)
        self.relu1 = nn.ReLU(inplace=True)
        self.hidden2 = nn.Linear(512, 1)
    
    def forward(self, x):
        out = self.relu1(self.hidden1(x))
        out = self.hidden2(out)
        return out

class BasicDeconv(nn.Module):
    def __init__(self):
        super(BasicDeconv, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2)
        self.relu2 = nn.ReLU(inplace=True)
        self.deconv3 = nn.ConvTranspose2d(64, 3, kernel_size=8, stride=4)
        self.relu3 = nn.ReLU(inplace=True)
    
    def forward(self, x):
        out = self.relu1(self.deconv1(x))
        out = self.relu2(self.deconv2(out))
        out = self.relu3(self.deconv3(out))

class evalMLP(nn.Module):
    def __init__(self):
        super(evalMLP, self).__init__()
        # Check input size
        self.hidden1 = nn.Linear(1024, 512)
        self.relu1 = nn.ReLU()
        self.hidden2 = nn.Linear(512, 512)
        self.relu2 = nn.ReLU()
    
    def forward(self, x):
        out = self.relu1(self.hidden1(x))
        out = self.relu2(self.hidden2(out))
        return F.softmax(out)