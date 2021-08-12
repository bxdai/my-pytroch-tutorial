#作者：Jeremy Howard，fast.ai。感谢雷切尔·托马斯和弗朗西斯科·英厄姆。
#我们建议将本教程作为笔记本而不是脚本运行。要下载笔记本 (.ipynb) 文件，请单击页面顶部的链接。
#PyTorch 提供了设计精美的模块和类 torch.nn 、 torch.optim 、 Dataset 和 DataLoader 来帮助您创建和训练神经网络。为了充分利用它们的力量并针对您的问题定制它们，您需要真正了解它们在做什么。为了加深理解，我们将首先在 MNIST 数据集上训练基本神经网络，而不使用这些模型的任何特征；我们最初将只使用最基本的 PyTorch 张量功能。然后，我们将一次从 torch.nn、torch.optim、Dataset 或 DataLoader 增量添加一个功能，准确显示每个部分的作用，以及它如何工作以使代码更简洁或更灵活。
#本教程假设您已经安装了 PyTorch，并且熟悉张量操作的基础知识。 （如果您熟悉 Numpy 数组操作，您会发现此处使用的 PyTorch 张量操作几乎相同）。

#1 MNIST data setup
#我们将使用经典的 MNIST 数据集，该数据集由手绘数字（0 到 9 之间）的黑白图像组成。
#我们将使用 pathlib 处理路径（Python 3 标准库的一部分），并将使用请求下载数据集。我们只会在使用时导入模块，因此您可以确切地看到每个点正在使用的内容。

from pathlib import Path
import requests

DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"

PATH.mkdir(parents=True,exist_ok=True)

URL = "https://github.com/pytorch/tutorials/raw/master/_static/"
FILENAME = "mnist.pkl.gz"
if not (PATH / FILENAME).exists():
        content = requests.get(URL + FILENAME).content
        (PATH / FILENAME).open("wb").write(content)

#该数据集采用 numpy 数组格式，并使用 pickle 存储，pickle 是一种用于序列化数据的 Python 特定格式。
import pickle
import gzip

with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")

#每个图像都是 28 x 28，并且被存储为长度为 784 (=28x28) 的扁平行。我们来看一个；我们需要先将其重塑为 2d。
from matplotlib import pyplot
import numpy as np

pyplot.imshow(x_train[0].reshape((28, 28)), cmap="gray")
print(x_train.shape)

#PyTorch 使用 torch.tensor，而不是 numpy 数组，因此我们需要转换我们的数据。
import torch

x_train, y_train, x_valid, y_valid = map(
    torch.tensor, (x_train, y_train, x_valid, y_valid)
)
n, c = x_train.shape
print(x_train, y_train)
print(x_train.shape)
print(y_train.min(), y_train.max())

# x_train = torch.tensor(x_train)
# y_train = torch.tensor(y_train)
# print(x_train, y_train)

#2 从零开始的神经网络（没有 torch.nn） 

#让我们首先使用 PyTorch 张量操作创建一个模型。 我们假设您已经熟悉神经网络的基础知识。 （如果你不是，你可以在 course.fast.ai 上学习它们）。

#PyTorch 提供了创建随机张量或零填充张量的方法，我们将使用这些方法为简单的线性模型创建权重和偏差。 这些只是常规张量，还有一个非常特殊的补充：我们告诉 PyTorch 它们需要梯度。 这会导致 PyTorch 记录对张量所做的所有操作，以便它可以在反向传播期间自动计算梯度！

#对于权重，我们在初始化之后设置 requires_grad，因为我们不希望该步骤包含在梯度中。 （请注意，PyTorch 中的尾随 _ 表示操作是就地执行的。） 

import math

weights = torch.randn(784,10)/math.sqrt(784)
weights.requires_grad_()
bias = torch.zeros(10,requires_grad=True)


#由于 PyTorch 能够自动计算梯度，我们可以使用任何标准 Python 函数（或可调用对象）作为模型！ 
# 所以让我们写一个简单的矩阵乘法和广播加法来创建一个简单的线性模型。 我们还需要一个激活函数，因此我们将编写 log_softmax 并使用它。
#  请记住：尽管 PyTorch 提供了许多预先编写的损失函数、激活函数等，但您可以使用纯 Python 轻松编写自己的函数。
#  PyTorch 甚至会自动为您的函数创建快速 GPU 或矢量化 CPU 代码。

def log_softmax(x):
    return x-x.exp().sum(-1).log().unsqueeze(-1)

def model(xb):
    return log_softmax(xb@weights+bias)

#在上面，@ 代表点积操作。 我们将对一批数据（在本例中为 64 张图像）调用我们的函数。
#  这是一次向前传递。 请注意，在此阶段我们的预测不会比随机更好，因为我们从随机权重开始。 

bs = 64  # batch size

xb = x_train[0:bs]  # a mini-batch from x
preds = model(xb)  # predictions
preds[0], preds.shape
print(preds[0], preds.shape)

#如您所见，preds 张量不仅包含张量值，还包含梯度函数。稍后我们将使用它来进行反向传播。
#让我们实现负对数似然作为损失函数（同样，我们可以只使用标准 Python）

def nll(input,target):
    return -input[range(target.shape[0]),target].mean()

loss_func = nll

#让我们用我们的随机模型检查我们的损失，这样我们就可以看看我们在反向传播后是否有所改进

yb = y_train[0:bs]
print(loss_func(preds, yb))

#让我们也实现一个函数来计算我们模型的准确性。对于每个预测，如果具有最大值的索引与目标值匹配，则预测是正确的。
def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()

#让我们检查一下我们的随机模型的准确性，这样我们就可以看到我们的准确性是否随着损失的提高而提高。
print(accuracy(preds, yb))

#我们现在可以运行一个训练循环。对于每次迭代，我们将:
###选择一小批数据（大小为 bs）
###使用模型进行预测
###计算损失
###loss.backward() 更新模型的梯度，在本例中为权重和偏差。

#您可以使用标准的 python 调试器来单步调试 PyTorch 代码，允许您在每一步检查各种变量值。取消下面的 set_trace() 注释以进行尝试。
from IPython.core.debugger import set_trace

lr = 0.5  # learning rate
epochs = 2  # how many epochs to train for

for epoch in range(epochs):
    for i in range((n - 1) // bs + 1):
        #         set_trace()
        start_i = i * bs
        end_i = start_i + bs
        xb = x_train[start_i:end_i]
        yb = y_train[start_i:end_i]
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        with torch.no_grad():
            weights -= weights.grad * lr
            bias -= bias.grad * lr
            weights.grad.zero_()
            bias.grad.zero_()

#就是这样：我们完全从头开始创建并训练了一个最小的神经网络（在这种情况下，是逻辑回归，因为我们没有隐藏层）！

#让我们检查损失和准确性，并将它们与我们之前得到的进行比较。我们预计损失会减少，准确度会增加，他们确实做到了。

print(loss_func(model(xb), yb), accuracy(model(xb), yb))


#3 Using torch.nn.functional

#我们现在将重构我们的代码，使其与以前做同样的事情，只是我们将开始利用 PyTorch 的 nn 类使其更加简洁和灵活。
# 从这里开始的每一步，我们都应该使我们的代码中的一个或多个：更短、更易于理解和/或更灵活。
#第一步也是最简单的一步是通过将我们手写的激活和损失函数替换为来自 torch.nn.functional（通常按照惯例导入到命名空间 F 中）的那些函数来缩短我们的代码。
# 该模块包含 torch.nn 库中的所有函数（而库的其他部分包含类）。
# 除了各种损失和激活函数外，您还会在此处找到一些用于创建神经网络的便捷函数，例如池化函数。 （也有用于进行卷积、线性层等的函数，但正如我们将看到的，使用库的其他部分通常可以更好地处理这些。）
#如果你使用负对数似然损失和对数 softmax 激活，那么 Pytorch 提供了一个单一的函数 F.cross_entropy 将两者结合起来。
# 所以我们甚至可以从我们的模型中删除激活函数。

import torch.nn.functional as F

loss_func = F.cross_entropy

def model(xb):
    return xb @ weights + bias

#请注意，我们不再在模型函数中调用 log_softmax。让我们确认一下我们的损失和准确率和之前一样

print(loss_func(model(xb), yb), accuracy(model(xb), yb))

#4 使用 nn.Module 重构
##接下来，我们将使用 nn.Module 和 nn.Parameter，以获得更清晰、更简洁的训练循环。我们将 nn.Module 子类化（它本身是一个类并且能够跟踪状态）。
# 在这种情况下，我们想要创建一个类来保存我们的权重、偏差和前进步骤的方法。 nn.Module 有许多我们将使用的属性和方法（例如 .parameters() 和 .zero_grad()）。

#注意：nn.Module（大写 M）是 PyTorch 特定的概念，是一个我们会经常使用的类。
# 不要将 nn.Module 与（小写的 m）模块的 Python 概念混淆，后者是可以导入的 Python 代码文件。

from torch import nn

class Mnist_Logistic(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weights = nn.Parameter(torch.randn(784, 10) / math.sqrt(784))
        self.bias = nn.Parameter(torch.zeros(10))

    def forward(self, xb):
        return xb @ self.weights + self.bias

#由于我们现在使用的是对象而不是函数，因此我们首先必须实例化我们的模型：


model = Mnist_Logistic()

#现在我们可以像以前一样计算损失。请注意， 
# nn.Module 对象被用作函数（即它们是可调用的），但在幕后 Pytorch 会自动调用我们的 forward 方法。

print(loss_func(model(xb), yb))

#以前，对于我们的训练循环，我们必须按名称更新每个参数的值，并分别手动将每个参数的梯度归零，如下所示：

with torch.no_grad():
    weights -= weights.grad * lr
    bias -= bias.grad * lr
    weights.grad.zero_()
    bias.grad.zero_()

#我们将我们的小训练循环包装在一个 fit 函数中，以便我们稍后再次运行它

def fit():
    for epoch in range(epochs):
        for i in range((n - 1) // bs + 1):
            start_i = i * bs
            end_i = start_i + bs
            xb = x_train[start_i:end_i]
            yb = y_train[start_i:end_i]
            pred = model(xb)
            loss = loss_func(pred, yb)

            loss.backward()
            with torch.no_grad():
                for p in model.parameters():
                    p -= p.grad * lr
                model.zero_grad()

fit()

#让我们仔细检查一下我们的损失是否已经减少
print(loss_func(model(xb), yb))


#5 使用 nn.Linear 重构
#我们继续重构我们的代码。我们不是手动定义和初始化 self.weights 和 self.bias，并计算 xb @ self.weights + self.bias，
# 而是使用 Pytorch 类 nn.Linear 作为线性层，它为我们完成所有这些。
#  Pytorch 有许多类型的预定义层，可以极大地简化我们的代码，并且通常也可以使其更快。

class Mnist_Logistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(784, 10)

    def forward(self, xb):
        return self.lin(xb)

#我们实例化我们的模型并以与以前相同的方式计算损失
model = Mnist_Logistic()
print(loss_func(model(xb), yb))

#我们仍然可以使用与以前相同的拟合方法

fit()

print(loss_func(model(xb), yb))

#6 使用优化重构 Refactor using optim

#Pytorch 还有一个包含各种优化算法的包，torch.optim。我们可以使用优化器中的 step 方法向前迈进，而不是手动更新每个参数。
#这将让我们替换我们之前手动编码的优化步骤:
#with torch.no_grad():
#    for p in model.parameters(): p -= p.grad * lr
#    model.zero_grad()

#而是使用：
#opt.step()
#opt.zero_grad()

from torch import optim

def get_model():
    model = Mnist_Logistic()
    return model, optim.SGD(model.parameters(), lr=lr)

model, opt = get_model()
print(loss_func(model(xb), yb))

for epoch in range(epochs):
    for i in range((n - 1) // bs + 1):
        start_i = i * bs
        end_i = start_i + bs
        xb = x_train[start_i:end_i]
        yb = y_train[start_i:end_i]
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        opt.step()
        opt.zero_grad()

print(loss_func(model(xb), yb))

#7 Refactor using Dataset
#PyTorch 有一个抽象的 Dataset 类。数据集可以是任何具有 __len__ 函数（由 Python 的标准 len 函数调用）和 __getitem__ 函数作为索引方式的任何东西。
# 本教程通过一个很好的示例来创建自定义 FacialLandmarkDataset 类作为 Dataset 的子类。
#PyTorch 的 TensorDataset 是一个 Dataset 包装张量。通过定义长度和索引方式，这也为我们提供了一种沿张量的第一维迭代、索引和切片的方法。这将使我们在训练的同一行中更容易访问自变量和因变量。

from torch.utils.data import TensorDataset

#x_train 和 y_train 都可以组合在一个 TensorDataset 中，这将更容易迭代和切片。
train_ds = TensorDataset(x_train, y_train)
#Previously, we had to iterate through minibatches of x and y values separately
# xb = x_train[start_i:end_i]
# yb = y_train[start_i:end_i]

#现在，我们可以一起做这两个步骤
xb,yb = train_ds[i*bs : i*bs+bs]

model, opt = get_model()

for epoch in range(epochs):
    for i in range((n - 1) // bs + 1):
        xb, yb = train_ds[i * bs: i * bs + bs]
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        opt.step()
        opt.zero_grad()

print(loss_func(model(xb), yb))

#8 使用 DataLoader 重构

#Pytorch的的DataLoader负责管理批次。您可以从任何一个数据集的DataLoader。
# 的DataLoader使得它更容易遍历批次。而不必使用train_ds [i * BS：i* BS + BS]，所述的DataLoader自动给我们每个minibatch。

from torch.utils.data import DataLoader

train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size=bs)

#以前，我们的循环像这样迭代批次 (xb, yb)：
# for i in range((n-1)//bs + 1):
#     xb,yb = train_ds[i*bs : i*bs+bs]
#     pred = model(xb)

#现在，我们的循环更加清晰，因为 (xb, yb) 是从数据加载器自动加载的：

for xb,yb in train_dl:
    pred = model(xb)

model, opt = get_model()

for epoch in range(epochs):
    for xb, yb in train_dl:
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        opt.step()
        opt.zero_grad()

print(loss_func(model(xb), yb))

#感谢 Pytorch 的 nn.Module、nn.Parameter、Dataset 和 DataLoader，
# 我们的训练循环现在明显更小且更容易理解。现在让我们尝试添加在实践中创建有效模型所需的基本功能。

#9 添加验证

#在第 1 节中，我们只是试图为我们的训练数据设置一个合理的训练循环。实际上，您始终还应该有一个验证集，以便确定您是否过度拟合。
#混洗训练数据对于防止批次和过度拟合之间的相关性很重要。另一方面，无论我们是否对验证集进行洗牌，验证损失都是相同的。
# 由于混洗需要额外的时间，因此混洗验证数据是没有意义的。
#我们将使用比训练集大两倍的验证集批量大小。这是因为验证集不需要反向传播，因此占用的内存更少（不需要存储梯度）。
# 我们利用这一点来使用更大的批量大小并更快地计算损失。

train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)

valid_ds = TensorDataset(x_valid, y_valid)
valid_dl = DataLoader(valid_ds, batch_size=bs * 2)

#我们将在每个 epoch 结束时计算并打印验证损失。
#（请注意，我们总是在训练之前调用 model.train()，在推理之前调用 model.eval()，
#因为它们被 nn.BatchNorm2d 和 nn.Dropout 等层使用，以确保这些不同阶段的适当行为。）

model, opt = get_model()

for epoch in range(epochs):
    model.train()
    for xb, yb in train_dl:
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        opt.step()
        opt.zero_grad()

    model.eval()
    with torch.no_grad():
        valid_loss = sum(loss_func(model(xb), yb) for xb, yb in valid_dl)

    print(epoch, valid_loss / len(valid_dl))

#10 创建fit()和get_data()

#我们现在将自己进行一些重构。由于我们经历了两次计算训练集和验证集损失的类似过程，让我们将其放入自己的函数 loss_batch 中，该函数计算一批的损失。
#我们为训练集传入一个优化器，并使用它来执行反向传播。对于验证集，我们不传递优化器，因此该方法不执行反向传播

def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)

#fit 运行必要的操作来训练我们的模型并计算每个 epoch 的训练和验证损失。

import numpy as np

def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb, yb, opt)

        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]
            )
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

        print(epoch, val_loss)
#get_data返回训练和验证集的数据加载器。

def get_data(train_ds, valid_ds, bs):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs * 2),
    )

#现在，我们获取数据加载器和拟合模型的整个过程可以在 3 行代码中运行

train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
model, opt = get_model()
fit(epochs, model, loss_func, opt, train_dl, valid_dl)

#Switch to CNN 切换到CNN
#您可以使用这 3 行基本代码来训练各种模型。让我们看看我们是否可以使用它们来训练卷积神经网络 (CNN)！

#现在，我们要建立我们的神经网络有三个卷积层。因为上一节中的函数都没有假设任何关于模型形式的东西，我们将能够使用它们来训练 CNN，而无需任何修改。
#我们将使用Pytorch的预定义Conv2d类作为我们的卷积层。我们定义了一个CNN 3个卷积层。每一圈之后是RELU。最后，我们执行的平均池。 （请注意，观点PyTorch的版本numpy的的重塑）

class Mnist_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1)

    def forward(self, xb):
        xb = xb.view(-1, 1, 28, 28)
        xb = F.relu(self.conv1(xb))
        xb = F.relu(self.conv2(xb))
        xb = F.relu(self.conv3(xb))
        xb = F.avg_pool2d(xb, 4)
        return xb.view(-1, xb.size(1))

lr = 0.1

#动量是随机梯度下降的一种变体，它也考虑了先前的更新，通常可以加快训练速度。

model = Mnist_CNN()
opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

fit(epochs, model, loss_func, opt, train_dl, valid_dl)

#10 nn.Sequential

#torch.nn还有另一个方便的类，可以用来简化我们的代码：Sequential。 Sequential对象以顺序方式运行其中包含的每个模块。 这是编写神经网络的一种简单方法。

#为了利用这一点，我们需要能够从给定的函数轻松定义自定义层。 例如，PyTorch 没有视层，我们需要为我们的网络创建一个层。 Lambda将创建一个层，然后在使用Sequential定义网络时可以使用该层。

class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)

def preprocess(x):
    return x.view(-1, 1, 28, 28)

model = nn.Sequential(
    Lambda(preprocess),
    nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.AvgPool2d(4),
    Lambda(lambda x: x.view(x.size(0), -1)),
)

opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

fit(epochs, model, loss_func, opt, train_dl, valid_dl)

#包装DataLoader

#Our CNN is fairly concise, but it only works with MNIST, because:

#假设输入为28 * 28长向量
#假设 CNN 的最终网格尺寸为4 * 4（因为这是平均值
#我们使用的合并核大小）

#让我们摆脱这两个假设，因此我们的模型适用于任何 2d 单通道图像。 首先，我们可以删除初始的 Lambda 层，但将数据预处理移至生成器中：

def preprocess(x, y):
    return x.view(-1, 1, 28, 28), y

class WrappedDataLoader:
    def __init__(self, dl, func):
        self.dl = dl
        self.func = func

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        batches = iter(self.dl)
        for b in batches:
            yield (self.func(*b))

train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
train_dl = WrappedDataLoader(train_dl, preprocess)
valid_dl = WrappedDataLoader(valid_dl, preprocess)

#接下来，我们可以将nn.AvgPool2d替换为nn.AdaptiveAvgPool2d，
# 这使我们能够定义所需的输出张量的大小，而不是所需的输入张量的大小。 结果，我们的模型将适用于任何大小的输入。

model = nn.Sequential(
    nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d(1),
    Lambda(lambda x: x.view(x.size(0), -1)),
)

opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

fit(epochs, model, loss_func, opt, train_dl, valid_dl)

#12 使用您的 GPU

#如果您足够幸运地能够使用具有 CUDA 功能的 GPU（可以从大多数云提供商处以每小时 0.50 美元的价格租用一个），
# 则可以使用它来加速代码。 首先检查您的 GPU 是否在 Pytorch 中正常工作：
print(torch.cuda.is_available())

#然后为其创建一个设备对象：

dev = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")

#让我们更新preprocess，将批量移至 GPU：
def preprocess(x, y):
    return x.view(-1, 1, 28, 28).to(dev), y.to(dev)

train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
train_dl = WrappedDataLoader(train_dl, preprocess)
valid_dl = WrappedDataLoader(valid_dl, preprocess)

#最后，我们可以将模型移至 GPU

model.to(dev)
opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

#您应该发现它现在运行得更快：
fit(epochs, model, loss_func, opt, train_dl, valid_dl)

#总结
#现在，我们有了一个通用的数据管道和训练循环，您可以将其用于使用 Pytorch 训练许多类型的模型。 要了解现在可以轻松进行模型训练，请查看mnist_sample示例笔记本。

# 当然，您需要添加很多内容，例如数据扩充，超参数调整，监控训练，迁移学习等。 这些功能可在 fastai 库中使用，该库是使用本教程中所示的相同设计方法开发的，为希望进一步推广其模型的从业人员提供了自然的下一步。

# 我们承诺在本教程开始时将通过示例分别说明torch.nn，torch.optim，Dataset和DataLoader。 因此，让我们总结一下我们所看到的：

# torch.nn
# Module：创建一个行为类似于函数的可调用对象，但也可以包含状态（例如神经网络层权重）。 它知道其中包含的 Parameter ，并且可以将其所有坡度归零，遍历它们以进行权重更新等。
# Parameter：张量的包装器，用于告知 Module 具有在反向传播期间需要更新的权重。 仅更新具有require_grad属性集的张量
# functional：一个模块（通常按照惯例导入到 F 名称空间中），其中包含激活函数，损失函数等。 以及卷积和线性层等层的无状态版本。
# torch.optim：包含诸如 SGD 的优化程序，这些优化程序在后退步骤
# Dataset 中更新 Parameter 的权重。 具有 __len__ 和 __getitem__ 的对象，包括 Pytorch 提供的类，例如 TensorDataset
# DataLoader：获取任何 Dataset 并创建一个迭代器，该迭代器返回批量数据。

#参考文档 https://pytorch.apachecn.org/docs/1.7/16.html
#官网文档 https://pytorch.org/tutorials/beginner/nn_tutorial.html#

