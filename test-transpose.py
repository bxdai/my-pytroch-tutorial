import torch

a = torch.randn(3, 5,4,6)
c = a.transpose(0,1)
b = a.permute(2, 3,1,0)

print(f"a shape:{a.shape}")
print(f"b shape:{b.shape}")

print(f"c shape:{c.shape}")

#coding=utf-8
import  torch

def change_tensor_shape():
    x=torch.randn(2,4,3)
    s=x.transpose(1,2) #shape=[2,3,4]
    y=x.view(2,3,4) #shape=[2,3,4]
    z=x.permute(0,2,1) #shape=[2,3,4]

    #tensor.t()只能转化 a 2D tensor
    m=torch.randn(2,3)#shape=[2,3]
    n=m.t()#shape=[3,2]
    print(f"m shape:{m.shape}")
    print(f"n shape:{n.shape}")

    #返回当前张量在某个维度为1扩展为更大的张量
    x = torch.Tensor([[1], [2], [3]])#shape=[3,1]
    t=x.expand(3, 4)
    print(t)
    '''
    tensor([[1., 1., 1., 1.],
        [2., 2., 2., 2.],
        [3., 3., 3., 3.]])
    '''

    #沿着特定的维度重复这个张量
    x=torch.Tensor([[1,2,3]])
    t=x.repeat(3, 2)
    print(t)
    '''
    tensor([[1., 2., 3., 1., 2., 3.],
        [1., 2., 3., 1., 2., 3.],
        [1., 2., 3., 1., 2., 3.]])
    '''
    x = torch.randn(2, 3, 4)
    t=x.repeat(2, 1, 3) #shape=[4, 3, 12]


def cat_and_stack():

    x = torch.randn(2,3,6)
    y = torch.randn(2,4,6)
    c=torch.cat((x,y),1)
    #c=(2*7*6)
    print(c.size)

    """
    而stack则会增加新的维度。
    如对两个1*2维的tensor在第0个维度上stack，则会变为2*1*2的tensor；在第1个维度上stack，则会变为1*2*2的tensor。
    """
    a = torch.rand((1, 2))
    b = torch.rand((1, 2))
    c = torch.stack((a, b), 0)
    print(c.size())#torch.Size([2, 1, 2])

if __name__=='__main__':
    change_tensor_shape()
    cat_and_stack()

