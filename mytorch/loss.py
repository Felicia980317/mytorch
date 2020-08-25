import torch
import torch.nn as nn


class MultiMSELoss(nn.Module):
    def __init__(self, scale, device=None):
        super(MultiMSELoss, self).__init__()
        self.mseloss = nn.MSELoss(reduction="sum")
        self.device = device
        if device:
            self.mseloss.to(device)
        self.scale = scale

    def forward(self, outputs_list, targets):
        if not isinstance(outputs_list, list):
            raise TypeError("MultiMSELoss, outputs_list type != list")
        n_output = len(outputs_list)
        loss = torch.zeros(1)
        if self.device:
            loss = loss.to(self.device)
        for outputs in outputs_list:
            loss += self.mseloss(outputs, targets)
        return loss / self.scale


#! not done yet
class FocalLoss(nn.Module):
    def __init__(self, gamma):
        super(FocalLoss, self).__init__()

    def forward(self, outputs, targets):
        pass
