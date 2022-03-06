import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable


class NormalizedCrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, labels):
        new_labels = labels / (torch.sum(labels, dim=1, keepdim=True) + 1e-8)
        loss = -1.0 * torch.mean(torch.sum(Variable(new_labels) * torch.log(pred), dim=1), dim=0)
        return loss


class CategoryCrossEntropy(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.T = T

    def forward(self, pred, soft_label):
        soft_label = F.softmax(soft_label / self.T, -1)
        soft_label = Variable(soft_label.detach().data, requires_grad=False)
        loss = -1.0 * torch.sum(Variable(soft_label) * torch.log_softmax(pred / self.T, -1), dim=-1)
        loss = loss.mean(-1).mean(-1)
        return loss


class AttLoss(nn.Module):
    def __init__(self, s_factor):
        super().__init__()
        self.s = s_factor

    def forward(self, att):
        t = att.size(1)
        max_att_values, _ = torch.topk(att, max(int(t // self.s), 1), -1)
        mean_max_att = max_att_values.mean(1)

        min_att_values, _ = torch.topk(-att, max(int(t // self.s), 1), -1)
        mean_min_att = -min_att_values.mean(1)

        loss = (mean_min_att - mean_max_att).mean(0)

        return loss