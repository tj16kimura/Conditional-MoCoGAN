import torch
import torch.nn.functional as F


def loss_hinge_dis(d_out, fake=False):
    if fake:
        loss = torch.mean(F.relu(1.0 + d_out))
    else:
        loss = torch.mean(F.relu(1.0 - d_out))
    return loss


def loss_hinge_gen(d_out):
    loss = -torch.mean(d_out)
    return loss
