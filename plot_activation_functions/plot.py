import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
#import pandas as pd

writer = SummaryWriter()
stepsize = 400
x_axis = torch.linspace(-3, 3, steps = stepsize)
print(x_axis)

for x in x_axis:
    #rel = nn.ReLU()
    tanh = nn.Tanh()
    softp = nn.Softplus()
    #writer.add_scalars('activation functions', {'relu':rel(x), 'tanh': tanh(x), 'softplus':softp(x)}, x* stepsize)
    writer.add_scalars('activation functions', {'tanh': tanh(x), 'softplus':softp(x)}, x* stepsize)



writer.close()
