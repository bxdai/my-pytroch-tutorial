#pytroch 自动微分求导 https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html
#在训练神经网络时，最常用的算法是反向传播。在该算法中，参数（模型权重）根据损失函数相对于给定参数的梯度进行调整
#为了计算这些梯度，PyTorch 有一个名为 torch.autograd 的内置微分引擎。它支持任何计算图的梯度自动计算
#考虑最简单的一层神经网络，输入 x，参数 w 和 b，以及一些损失函数。可以通过以下方式在 PyTorch 中定义
from numpy.lib.twodim_base import triu_indices_from
import torch
x=torch.rand(5)# input tensor
y=torch.rand(3)# expected output
w=torch.randn(5,3,requires_grad=True)
#w.requires_grad = True
b=torch.randn(3,requires_grad=True)
z = torch.matmul(x,w)+b


loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
#我们应用于张量来构建计算图的函数实际上是类 Function 的对象。该对象知道如何在前向计算函数，以及如何在反向传播步骤中计算其导数。
# 对反向传播函数的引用存储在张量的 grad_fn 属性中。您可以在文档中找到有关 Function 的更多信息。
print('Gradient function for z =',z.grad_fn)
print('Gradient function for loss =', loss.grad_fn)
loss.backward()
print(w.grad)
print(b.grad)

#禁用梯度跟踪
#法1
print(z.requires_grad)
with torch.no_grad():
    z = torch.matmul(x,w)+b
print(z.requires_grad)
#法二
#z = torch.matmul(x, w)+b
z_det = z.detach()
print(z_det.requires_grad)

#什么时候需要禁用梯度跟踪
#将神经网络中的某些参数标记为冻结参数。这是微调预训练网络的一个非常常见的场景
#在仅进行前向传递时加快计算速度，因为对不跟踪梯度的张量进行计算会更有效。


#More on Computational Graphs
#从概念上讲，autograd 在由 Function 对象组成的有向无环图 (DAG) 
# 中保存数据（张量）和所有已执行操作（以及生成的新张量）的记录。在这个 DAG 中，
# 叶子是输入张量，根是输出张量。通过从根到叶跟踪此图，您可以使用链式法则自动计算梯度。

#在前向传递中，autograd 同时做两件事
#---运行请求的操作来计算结果张量
#---在 DAG 中维护操作的梯度函数。

#当在 DAG root上调用 .backward() 时，反向传递开始。 autograd 然后：
#----计算每个 .grad_fn 的梯度，
#----将它们累积在各自张量的 .grad 属性中
#----使用链式法则，一直传播到叶张量。


#Note
#PyTorch 中的 DAG 是动态的 需要注意的重要一点是，
# 该图是从头开始重新创建的；在每次 .backward() 调用之后，autograd 开始填充一个新图形。
# 这正是允许您在模型中使用控制流语句的原因；如果需要，您可以在每次迭代时更改形状、大小和操作
#


#张量梯度和雅可比积
#---在很多情况下，我们有一个标量损失函数，我们需要计算一些参数的梯度。
#---但是，有些情况下输出函数是任意张量。在这种情况下，PyTorch 允许您计算所谓的雅可比乘积，而不是实际的梯度




inp = torch.eye(5, requires_grad=True)#对角线为1其他为0
out = (inp+1).pow(2)
print("out:{}\n".format(out))
out.backward(torch.ones_like(inp), retain_graph=True)
print("First call\n", inp.grad)
out.backward(torch.ones_like(inp), retain_graph=True)
print("\nSecond call\n", inp.grad)
out.backward(torch.ones_like(inp), retain_graph=True)
print("\nThree call\n",inp.grad)
inp.grad.zero_()
out.backward(torch.ones_like(inp), retain_graph=True)
print("\nCall after zeroing gradients\n", inp.grad)

