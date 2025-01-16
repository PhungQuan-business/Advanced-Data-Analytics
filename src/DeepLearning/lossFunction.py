import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        # inputs = inputs.view(-1)
        # targets = targets.view(-1)
        if inputs.size(1) > 1:  # Multiclass case
            targets = torch.nn.functional.one_hot(
                targets, num_classes=inputs.size(1)).float()
        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth) / \
            (inputs.sum() + targets.sum() + smooth)

        return 1 - dice


class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        # inputs = inputs.view(-1)
        # targets = targets.view(-1)
        if inputs.size(1) > 1:  # Multiclass case
            targets = torch.nn.functional.one_hot(
                targets, num_classes=inputs.size(1)).float()

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.*intersection + smooth) / \
            (inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE


class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        # inputs = inputs.view(-1)
        # targets = targets.view(-1)
        if inputs.size(1) > 1:  # Multiclass case
            targets = torch.nn.functional.one_hot(
                targets, num_classes=inputs.size(1)).float()

        # intersection is equivalent to True Positive count
        # union is the mutually inclusive area of all labels & predictions
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        IoU = (intersection + smooth)/(union + smooth)

        return 1 - IoU


class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()

    def forward(self, inputs, targets, alpha=0.8, gamma=2, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)
        # flatten label and prediction tensors
        # inputs = inputs.view(-1)
        # targets = targets.view(-1)

        if inputs.size(1) > 1:  # Multiclass case
            targets = torch.nn.functional.one_hot(
                targets, num_classes=inputs.size(1)).float()

        # print(inputs.size(), targets.size())
        # first compute binary cross-entropy
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE

        return focal_loss


class TverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(TverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=0.5, beta=0.5):

        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        # inputs = inputs.view(-1)
        # targets = targets.view(-1)
        if inputs.size(1) > 1:  # Multiclass case
            targets = torch.nn.functional.one_hot(
                targets, num_classes=inputs.size(1)).float()

        # True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()

        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)

        return 1 - Tversky


class FocalTverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalTverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=0.5, beta=0.5, gamma=1):

        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        # inputs = inputs.view(-1)
        # targets = targets.view(-1)

        if inputs.size(1) > 1:  # Multiclass case
            targets = torch.nn.functional.one_hot(
                targets, num_classes=inputs.size(1)).float()

        # True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()

        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)
        FocalTversky = (1 - Tversky)**gamma

        return FocalTversky


class ComboLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(ComboLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=0.5, beta=0.5, eps=1e-9):

        # flatten label and prediction tensors
        # inputs = inputs.view(-1)
        # targets = targets.view(-1)

        if inputs.size(1) > 1:  # Multiclass case
            targets = torch.nn.functional.one_hot(
                targets, num_classes=inputs.size(1)).float()

        # True Positives, False Positives & False Negatives
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / \
            (inputs.sum() + targets.sum() + smooth)

        inputs = torch.clamp(inputs, eps, 1.0 - eps)
        out = - (ALPHA * ((targets * torch.log(inputs)) +
                 ((1 - ALPHA) * (1.0 - targets) * torch.log(1.0 - inputs))))
        weighted_ce = out.mean(-1)
        combo = (CE_RATIO * weighted_ce) - ((1 - CE_RATIO) * dice)

        return combo
