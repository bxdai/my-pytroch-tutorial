from numpy.core.fromnumeric import shape
import torch
import numpy as np
from torch.functional import Tensor

#%%
#Tensor Initialization
#1.直接从data创建，数据类型会被自动推导
data =[[1,2],[3,4]]
print(f"data type {type(data)}")
x_data=torch.tensor(data)
#print(x_data)

#2.从numpy array 中创建tensor
np_array = np.array(data)
x_np = torch.tensor(np_array)
#print(x_np)
print(x_np.shape)

#3.从其他tensor创建，新的tensor会保留数据类型和形状，除非明确覆盖
x_one = torch.ones_like(x_data)
#print("one tensor: \n{} \n".format(x_one)) # retains the properties of x_data
x_rand = torch.rand_like(x_data,dtype=torch.float32) # overrides the datatype of x_data
#print(f"Random Tensor: \n {x_rand} \n")
#%%

#shape 是tensor尺寸的一个元组
shape = (2,3)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")
print("ones_tensor.shape:  {}\n".format(ones_tensor.shape))

#Tensor Attributes
#Tensor attributes describe their shape, datatype, and the device on which they are stored.
tensor = torch.rand(3,4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

#%%
#Tensor Operations
#tensor operations doc https://pytorch.org/docs/stable/torch.html\


if torch.cuda.is_available():
    tensor = tensor.to('cuda')
#标准的索引和切片一个tensor
tensor = torch.ones(4,4)
tensor[:,1] = 0
print(tensor)
#1.连接tensor
tensor_clone = tensor.clone()
tensor_clone = torch.stack([tensor, tensor, tensor],dim=1)
print("tensor_clone:{}\n".format(tensor_clone))
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

#2.tensor相乘
# This computes the element-wise product
print(f"tensor.mul(tensor) \n {tensor.mul(tensor)} \n")
# Alternative syntax:
print(f"tensor * tensor \n {tensor * tensor}")
#3.在两个张量中使用矩阵乘法
t2 = tensor.matmul(tensor.T)
print("tensor.matmul(tensor.T) \n {} \n".format(t2))
#等效语法如下
print(f"tensor @ tensor.T \n {tensor @ tensor.T}")
#4.原地操作
#语法上有_的操作就是原地操作
print(tensor,"\n")
tensor.add_(5)
print(tensor)
#5.CPU 和 NumPy 数组上的张量可以共享其底层内存位置，更改一个将更改另一个
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")
#张量的变化反映在 NumPy 数组中
t.add_(1)
print(f"t: {t}")
print(f"n: {n}")
