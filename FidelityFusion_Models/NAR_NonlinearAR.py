import torch
import torch.nn as nn
# import kernel as kernel
import base.kernel as kernel
from base.gp_basic import GP_basic as CIGP
import MF_pack as mf
import matplotlib.pyplot as plt

##TODO: finish the training part

# demo 
if __name__ == "__main__":
    torch.manual_seed(1)
    # generate the data
    x_all = torch.rand(500, 1) * 20
    xlow_indices = torch.randperm(500)[:300]
    x_low = x_all[xlow_indices]
    xhigh_indices = torch.randperm(500)[:300]
    x_high = x_all[xhigh_indices]
    x_test = torch.linspace(0, 20, 100).reshape(-1, 1)

    y_low = torch.sin(x_low) + torch.rand(300, 1) * 0.6 - 0.3
    y_high = torch.sin(x_high) + torch.rand(300, 1) * 0.2 - 0.1
    y_test = torch.sin(x_test)

    x_train = [x_low, x_high]
    y_train = [y_low, y_high]

