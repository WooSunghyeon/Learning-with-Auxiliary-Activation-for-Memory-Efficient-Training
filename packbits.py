# PyTorch bit packing inspired by np.packbits / np.unpackbits. Feature request: https://github.com/pytorch/pytorch/issues/32867

import torch

'''
packbits_one_bits is used to packing 1-byte Tesnor whose components are only 0 or 1.
unpackbits_one_bits restore packed Tensor.
In unpackbits_one_bits function, you have to use relu=True if you want to restore the value of Tensor with 0 or 1.
Otherwise, it restores the value of Tensor with -1 or 1.
'''

def packbits_one_bits(x):
    x_size = x.size()
    # Float to Int
    x = x.int() 
    # Packbits
    x = x.reshape(-1, 8) 
    y = x[:,0] 
    for i in range(1,8):
        y = torch.bitwise_or(y, x[:,i]<<i)
    # Int to Byte
    y = y.byte()
    return y.detach().clone(), x_size

def unpackbits_one_bits(x, x_size, relu=False):
    # Byte to int
    x = x.int() 
    # Unpackbits
    y = torch.bitwise_and(x, 1).unsqueeze(0)
    for i in range(1,8):
        z = torch.bitwise_and(x>>i, 1).unsqueeze(0)
        y = torch.cat([y,z])
        
    y = torch.flatten(y.permute(1,0)).view(x_size)
    if not(relu):
        y[y == 0] = -1
    # Int to float tensor
    y = y.float()
    return y.detach().clone()

