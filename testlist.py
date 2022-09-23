#%%
import os

points=[(1,2),(2.1,2.2),(3.1,3.2)]

#print(points)
fac = 2
new_points = [(p[0] /2,p[1] /2) for p in points]
print(list(new_points))


# %%
import torch
a = torch.randn(28, 28)
a.unsqueeze(0).shape
a.unsqueeze(0).shape
a.shape
#%%
a = torch.unsqueeze(a,(0,1))
a.shape