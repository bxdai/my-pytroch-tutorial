import torch
import numpy as np

data = [[1,2],[3,4]]
x_data = torch.tensor(data)
print(x_data)

#%%
import torch
a = torch.Tensor([[1.5,2.2,0.3],[2.2,3.1,0.8]])
# %%
a
# %%
b = torch.sigmoid(a)
# %%
b
# %%
c = torch.softmax(a,dim=0)
a.shape
# %%
c
# %%
import torch
x = torch.tensor([[1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0.]])
x = x.reshape(-1, 3, 4)
# %%
x
# %%
x.shape