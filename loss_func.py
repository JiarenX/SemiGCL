import torch
import torch.nn.functional as F


def adentropy(F1, feat, lamda=1.0):
    out_t1 = F1(feat, reverse=True, lamda=lamda)
    out_t1 = F.softmax(out_t1, dim=-1)
    loss_adent = torch.mean(torch.sum(out_t1 * (torch.log(out_t1 + 1e-5)), 1))

    return loss_adent