import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class SoftNLL(nn.Module):
    def __init__(self):
        """The `soft' version of negative_log_likelihood, where y is a distribution
                over classes rather than a one-hot coding
            """
        super(SoftNLL, self).__init__()

    def forward(self, input, target):
        return -torch.mean(torch.sum(torch.log(input) * target, dim=1))

