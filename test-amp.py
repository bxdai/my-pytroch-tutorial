#%%
import torch
# Creates some tensors in default dtype (here assumed to be float32)
a_float32 = torch.rand((8, 8), device="cuda")
b_float32 = torch.rand((8, 8), device="cuda")
c_float32 = torch.rand((8, 8), device="cuda")
d_float32 = torch.rand((8, 8), device="cuda")

with torch.cuda.amp.autocast_mode.autocast(True):
    e_float16 = torch.mm(a_float32, b_float32)
    print(e_float16.type,e_float16.dtype)
    with torch.cuda.amp.autocast_mode.autocast(enabled=False):
        # Calls e_float16.float() to ensure float32 execution
        # (necessary because e_float16 was created in an autocasted region)
        f_float32 = torch.mm(c_float32, e_float16.float())
        print(f_float32.type,f_float32.dtype)
    # No manual casts are required when re-entering the autocast-enabled region.
    # torch.mm again runs in float16 and produces float16 output, regardless of input types.
    g_float16 = torch.mm(d_float32, f_float32)
    print(g_float16.type,g_float16.dtype)
# %%
