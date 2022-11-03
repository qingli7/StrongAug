import torch
import torch.nn as nn
from torch.autograd import Variable


def L2_dist(x, y):
    # x : shape [batch, dim]
    # y : shape [num_classes, dim]
    # dist : [batch, num_classes]
    dist = torch.sum(torch.square(x[:, None, :] - y), dim=-1)
    return dist


class PNLoss(nn.Module):
    def __init__(self):
        super(PNLoss, self).__init__()
        self.adversarial_loss = torch.nn.BCELoss()

    def L2_dist(self, x, y):
        # x : shape [batch, dim], 64 x 512
        # y : shape [num_classes, dim], C x 512
        # dist : [batch, num_classes], 64 x C
        dist = torch.sqrt(torch.sum(torch.square(x[:, None, :] - y), dim=-1))
        return dist
    
    def forward(self, valided, valided_f, P):
        valided = torch.sigmoid(-torch.sum(self.L2_dist(valided, P), -1))
        valided_f = torch.sigmoid(-torch.sum(self.L2_dist(valided_f, P), -1))
        
        # valided = torch.gather(valided, dim=1, index=labels.unsqueeze(-1).to(torch.int64))
        # valided_f = torch.gather(valided_f, dim=1, index=labels.unsqueeze(-1).to(torch.int64))
        # print(valided.shape, valided_f.shape, valided[0],valided_f[0])
        
        valid = Variable(torch.FloatTensor(valided.shape[0]).fill_(1.0), requires_grad=False).cuda()
        fake = Variable(torch.FloatTensor(valided.shape[0]).fill_(0.0), requires_grad=False).cuda()
        # Measure discriminator's ability to classify real from generated samples
        real_loss = self.adversarial_loss(valided, valid)
        fake_loss = self.adversarial_loss(valided_f, fake)
        d_loss = real_loss + fake_loss 
        return d_loss
    

class PLoss(nn.Module):
    def __init__(self):
        super(PLoss, self).__init__()

    def L2_dist(self, x, y):
        # x : shape [batch, dim], 64 x 512
        # y : shape [num_classes, dim], C x 512
        # dist : [batch, num_classes], 64 x C
        dist = torch.sqrt(torch.sum(torch.square(x[:, None, :] - y), dim=-1))
        return dist

    def forward(self, feat, prototype, label):
        dist = self.L2_dist(feat, prototype)
        # gather利用index来索引input特定位置的数值，unsqueeze(-1)拆分元素
        pos_dist = torch.gather(dist, dim=1, index=label.unsqueeze(-1).to(torch.int64))
        pl = torch.mean(pos_dist)
        return pl
    
    
class CLoss(nn.Module):
    def __init__(self):
        super(CLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        loss = self.loss(logits, labels)
        return loss
       

if __name__ == '__main__':
    features = torch.rand((30, 2)).cuda()
    features2 = torch.rand((30, 2)).cuda()
    prototypes = torch.rand((10, 2)).cuda()
    # labels = torch.rand((30,)).long().cuda()
    pnAug = PNLoss()
    loss = pnAug(features, features2, prototypes)
    print(loss)
