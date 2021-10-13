from numpy.core.fromnumeric import shape
import torch
import torch.nn as nn
import torch.nn.functional as F


x = torch.randn(1,572,572)
print(x.shape)

kernel = torch.randn(1,1,1)

x2 = F.conv2d(x,kernel,stride=1)
# x = torch.linspace(3,10,5)
# print(x)

# y = torch.rand((1,3,5))
# y = torch.flatten(y,2)
# print(y.shape)

# a = 5
# if isinstance(a,int):
#     print("int")
# type(a)


# for epoch in range(10):
#      print("-" * 10)
#      print(f"epoch {epoch + 1}/{10}")