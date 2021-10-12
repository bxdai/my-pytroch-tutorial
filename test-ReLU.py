import torch
import torch.nn as nn
m = torch.randn(2)
print(m)
m = nn.ReLU()
input = torch.randn(2).unsqueeze(0)
print('input:\n{}'.format(input))
output = torch.cat((m(input),m(-input)))
print('output:\n{}'.format(output))

m = nn.Sigmoid()
input = torch.randn(2)
output = m(input)
print('output:\n{}'.format(output))
