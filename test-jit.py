#%%
import torch  # This is all you need to use both PyTorch and TorchScript!
print(torch.__version__)
#%%

#%%
import torch
import torchvision

# An instance of your model.
model = torchvision.models.resnet18()

# An example input you would normally provide to your model's forward() method.
example = torch.rand(1, 3, 224, 224)

# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
traced_script_module = torch.jit.trace(model, example)
# %%
output = traced_script_module(torch.ones(1, 3, 224, 224))
# %%
output[0, :5]
# %%
traced_script_module.save("traced_resnet_model.pt")
# %%
class MyModule(torch.nn.Module):
    def __init__(self, N, M):
        super(MyModule, self).__init__()
        self.weight = torch.nn.Parameter(torch.rand(N, M))

    def forward(self, input):
        if input.sum() > 0:
          output = self.weight.mv(input)
        else:
          output = self.weight + input
        return output

my_module = MyModule(10,20)
sm = torch.jit.script(my_module)
#%%
sm.save("traced_resnet_model.pt")