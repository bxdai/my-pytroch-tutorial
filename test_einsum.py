#%%
import torch
from torch import einsum
a = torch.ones(3,4)
b = torch.ones(4,5)
c = torch.ones(6,7,8)
d = torch.ones(3,4)
x, y = torch.randn(5), torch.randn(5)
#%%
einsum('i,j', a)   # 等价于einsum('i,j->', a)
#%%
einsum('i,j,k', c)

# %%
einsum('ii->i', a)

# %%
a = torch.rand(2,3)
b = torch.rand(3,4)
c = torch.einsum("ik,kj->ij", [a, b])
# 等价操作 torch.mm(a, b)
# %%
print(c)
# %%
import torch
import numpy as np

a = torch.randn(2,3)
b = torch.randn(5,3,7)
c = torch.randn(2,7)
# i = 2, k = 3, j = 5, l = 7
torch_ein_out = torch.einsum('ik,jkl,il->ij', [a, b, c]).numpy()
# %%
torch_ein_out.shape
# %%
torch_ein_out = torch.einsum('ik,jkl,il->ijl', [a, b, c]).numpy()
# %%
torch_ein_out.shape
# %%
