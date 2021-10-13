import torch
import torch.nn as nn

bn = nn.BatchNorm2d(3)
x = torch.randn(4, 3, 2, 2)
y = bn(x)
a = (x[0, 0, :, :] + x[1, 0, :, :] + x[2, 0, :, :] + x[3, 0, :, :]).sum() / 16
b = (x[0, 1, :, :] + x[1, 1, :, :] + x[2, 1, :, :] + x[3, 1, :, :]).sum() / 16
c = (x[0, 2, :, :] + x[1, 2, :, :] + x[2, 2, :, :] + x[3, 2, :, :]).sum() / 16
print('The mean value of the first channel is %f' % a.data)
print('The mean value of the first channel is %f' % b.data)
print('The mean value of the first channel is %f' % c.data)
print('The output mean value of the BN layer is %f, %f, %f' % (bn.running_mean.data[0], bn.running_mean.data[1], bn.running_mean.data[2]))
