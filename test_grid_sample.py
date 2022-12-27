import torch
import torch.nn as nn
import torch.nn.functional as F
inp = torch.arange(4*4).view(1, 1, 4, 4).float()
d = torch.linspace(-1, 1, 8)
meshx, meshy = torch.meshgrid((d, d))
grid = torch.stack((meshy, meshx), 2).unsqueeze(0)
output = F.grid_sample(inp, grid, align_corners=False)
