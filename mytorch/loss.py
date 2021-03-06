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


class SmoothCrossEntropyLoss(nn.Module):
    def __init__(self, label_smoothing=0.0, size_average=True):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.size_average = size_average

    def forward(self, input, target):
        if len(target.size()) == 1:
            target = torch.nn.functional.one_hot(target, num_classes=input.size(-1))
            target = target.float().cuda()
        if self.label_smoothing > 0.0:
            s_by_c = self.label_smoothing / len(input[0])
            smooth = torch.zeros_like(target)
            smooth = smooth + s_by_c
            target = target * (1.0 - s_by_c) + smooth

        return cross_entropy(input, target, self.size_average)


def cross_entropy(input, target, size_average=True):
    """ Cross entropy that accepts soft targets
    Args:
         pred: predictions for neural network
         targets: targets, can be soft
         size_average: if false, sum is returned instead of mean
    Examples::
        input = torch.FloatTensor([[1.1, 2.8, 1.3], [1.1, 2.1, 4.8]])
        input = torch.autograd.Variable(out, requires_grad=True)
        target = torch.FloatTensor([[0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
        target = torch.autograd.Variable(y1)
        loss = cross_entropy(input, target)
        loss.backward()
    """
    logsoftmax = torch.nn.LogSoftmax(dim=1)
    if size_average:
        return torch.mean(torch.sum(-target * logsoftmax(input), dim=1))
    else:
        return torch.sum(torch.sum(-target * logsoftmax(input), dim=1))
