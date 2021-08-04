import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class FPN(nn.module):
    def __init__(self, block, layers):
        super(FPN, self).__init__()
        