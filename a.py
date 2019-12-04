import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from functools import reduce, partial
from detection.layers import style_pool2d

if __name__ == '__main__':
    a = torch.randn(1, 32, 5, 5)

    y1 = style_pool2d(a, kernel_size=5)

    var, mean = torch.var_mean(a, dim=(2, 3), unbiased=False, keepdim=True)  # (n, c, new_h * new_w)
    std = torch.sqrt(var + 1e-12)

    windows = torch.cat((mean, std), dim=1)

    print(torch.allclose(y1, windows))
    print((y1 == windows))
