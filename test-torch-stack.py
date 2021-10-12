import torch

x = torch.rand(3,3)
y = torch.rand(3,3)
print(f"x shape is:{x.shape}")
z= torch.stack([x,y],dim= 2)
print(f"z shape is:{z.shape}")
