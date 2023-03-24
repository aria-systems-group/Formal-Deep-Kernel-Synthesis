import torch
import torch.nn as nn


# ======================================================================
# 1. Define NN structure and box dataset for crown
# ======================================================================

class DynModelNetTanh3(nn.Module):
    def __init__(self, d=1, width_1=1, width_2=1, out_dim=1):
        super(DynModelNetTanh3, self).__init__()
        self.layer_1 = nn.Linear(d, width_1)
        self.layer_2 = nn.Linear(width_1, width_2)
        self.layer_3 = nn.Linear(width_2, width_2)
        self.tanh = nn.Tanh()
        self.layer_4 = nn.Linear(width_2, out_dim)

    def forward(self, x):
        out = self.layer_1(x)
        out = self.tanh(out)
        out = self.layer_2(out)
        out = self.tanh(out)
        out = self.layer_3(out)
        out = self.tanh(out)
        out = self.layer_4(out)
        return out


class DynModelNetRelu3(nn.Module):
    def __init__(self, d=1, width_1=1, width_2=1, out_dim=1):
        super(DynModelNetRelu3, self).__init__()
        self.layer_1 = nn.Linear(d, width_1)
        self.layer_2 = nn.Linear(width_1, width_2)
        self.layer_3 = nn.Linear(width_2, width_2)
        self.layer_4 = nn.Linear(width_2, out_dim)
        self.ReLU = nn.ReLU()

    def forward(self, x):
        out = self.layer_1(x)
        out = self.ReLU(out)
        out = self.layer_2(out)
        out = self.ReLU(out)
        out = self.layer_3(out)
        out = self.ReLU(out)
        out = self.layer_4(out)
        return out


class DynModelNetRelu2(nn.Module):
    def __init__(self, d=1, width_1=1, width_2=1, out_dim=1):
        super(DynModelNetRelu2, self).__init__()
        self.layer_1 = nn.Linear(d, width_1)
        self.layer_2 = nn.Linear(width_1, width_2)
        self.layer_4 = nn.Linear(width_2, out_dim)
        self.ReLU = nn.ReLU()

    def forward(self, x):
        out = self.layer_1(x)
        out = self.ReLU(out)
        out = self.layer_2(out)
        out = self.ReLU(out)
        out = self.layer_4(out)
        return out


class DynModelNetRelu1(nn.Module):
    def __init__(self, d=1, width_1=1, width_2=1, out_dim=1):
        super(DynModelNetRelu1, self).__init__()
        self.layer_1 = nn.Linear(d, width_1)
        self.layer_2 = nn.Linear(width_1, out_dim)
        self.ReLU = nn.ReLU()

    def forward(self, x):
        out = self.layer_1(x)
        out = self.ReLU(out)
        out = self.layer_2(out)
        return out


class DynModelNetTanh1(nn.Module):
    def __init__(self, d=1, width_1=1, width_2=1, out_dim=1):
        super(DynModelNetTanh1, self).__init__()
        self.layer_1 = nn.Linear(d, width_1)
        self.layer_2 = nn.Linear(width_1, out_dim)
        self.Tanh = nn.Tanh()

    def forward(self, x):
        out = self.layer_1(x)
        out = self.Tanh(out)
        out = self.layer_2(out)
        return out



def simple_box_data(x_max, x_min):
    x1_avg = (x_max[0] - x_min[0]) / 2. + x_min[0]
    x2_avg = (x_max[1] - x_min[1]) / 2. + x_min[1]
    X = torch.tensor([[x1_avg, x2_avg]]).float()
    labels = torch.tensor([0]).long()
    data_max = torch.tensor([[x_max[0], x_max[1]]]).reshape(1, -1)
    data_min = torch.tensor([[x_min[0], x_min[1]]]).reshape(1, -1)
    eps = None
    return X, labels, data_max, data_min, eps


def simple_box_data_nD(x_max, x_min, eps=0):
    dim = len(x_max)
    X = torch.tensor([[(x_max[i] - x_min[i]) / 2. + x_min[i] for i in range(dim)]]).float()
    labels = torch.tensor([0]).long()
    data_max = torch.tensor([[x_max[i] for i in range(dim)]]).reshape(1, -1)
    data_min = torch.tensor([[x_min[i] for i in range(dim)]]).reshape(1, -1)
    eps_used = None
    return X, labels, data_max, data_min, eps_used
