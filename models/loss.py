import torch
import torch.nn as nn
from torch.nn import init
import torchvision
from torch.optim import lr_scheduler
import torch.nn.functional as F

def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        lambda_rule = lambda epoch: opt.lr_gamma ** ((epoch+1) // opt.lr_decay_epochs)
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer,step_size=opt.lr_decay_iters, gamma=0.1)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler

class SegmantationLoss(nn.Module):
    def __init__(self, class_weights=None):
        super(SegmantationLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss(weight=class_weights)
    def __call__(self, output, target, pixel_average=True):
        if pixel_average:
            return self.loss(output, target) #/ target.data.sum()
        else:
            return self.loss(output, target)
