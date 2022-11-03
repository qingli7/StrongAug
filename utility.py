import torch
import torch.nn as nn
import numpy as np
from patchify import patchify, unpatchify


class PNAugment(nn.Module):
    def __init__(self):
        super(PNAugment, self).__init__()

    def L2_dist(self, x, y):
        # x : shape [batch, dim], 64 x 512
        # y : shape [num_classes, dim], C x 512
        # dist : [batch, num_classes], 64 x C
        dist = torch.sqrt(torch.sum(torch.square(x[:, None, :] - y), dim=-1))
        return dist

    def forward(self, feat, P, N):
        pos_dist = self.L2_dist(feat, P)
        neg_dist = self.L2_dist(feat, N)
        
        pnProb = torch.exp(-pos_dist)/(torch.exp(-pos_dist) + torch.exp(-neg_dist))
        # print('prob:',pos_dist.shape, pnProb.shape)
        return pnProb



def fake_trans(inputs):
    #2D image patchify and merge
    inputs = inputs.cpu().numpy()
    for i in range(len(inputs)):
        patches = patchify(inputs[i], (3,16,16), step=1)
        print(patches.shape)
        neg_inputs = unpatchify(patches, inputs[i].shape)
        print(neg_inputs.shape)
    
    return neg_inputs