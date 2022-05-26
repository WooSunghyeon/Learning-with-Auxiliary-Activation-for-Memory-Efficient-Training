import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.cpp_extension import load
from packbits import packbits_one_bits, unpackbits_one_bits
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

cudnn_convolution = load(name="cudnn_convolution", sources=["../cudnn_conv.cpp"], verbose=True)
cudnn_batch_norm = load(name="cudnn_batchnorm", sources=["../batchnorm.cpp"], verbose=True)

#%%
def learning_indicator(real_act, alternative_act, convnet=False):
    '''
    This function can calculate learning_indicator by using real_act and alternative_act. 
    '''
    if convnet:
        bp_direction = real_act*real_act
        as_direction = alternative_act * (2*real_act - alternative_act)
        bp_direction = torch.sum(bp_direction.view(bp_direction.size(0),-1), dim=1) + 1e-6
        as_direction = torch.sum(as_direction.view(as_direction.size(0),-1), dim=1)
        li = torch.flatten(as_direction/bp_direction).view(1,-1)
        li.flatten()[:50].detach().clone()
    else:
        bp_direction = torch.sum(real_act*real_act, dim=2) + 1e-6
        as_direction = torch.sum(alternative_act * (2*real_act - alternative_act), dim=2)
        li = torch.flatten(as_direction/bp_direction).tolist()
    return li


#%%
class Conv2d_ARA(nn.Conv2d):
    '''
    Conv2d_ARA uses auxilairy reidual activation (ARA) to update weights in backward propagation.
    You can use get_li=True if you want to extract learning indicator.
    In this layer, we do not ARA to input because it has been already added outside of the module in the resnet.
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=False, get_li=False):
        super(Conv2d_ARA, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=groups)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.li=None
        self.get_li = get_li
        
    def forward(self, input, ARA):
        if self.get_li and self.training:
            self.li = learning_indicator(input, ARA, convnet=True)
        return conv2d_ara.apply(input, ARA, self.weight, self.bias, self.stride, self.padding, self.groups) 
    
class conv2d_ara(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, ARA, weight, bias, stride=1, padding=0, groups=1):
        output = cudnn_convolution.convolution(input, weight, bias, stride, padding, (1, 1), groups, False, False)
        ctx.stride = stride
        ctx.padding = padding
        ctx.groups = groups
        ctx.save_for_backward(weight, bias)
        return output, ARA.clone()
    
    @staticmethod
    def backward(ctx, grad_output, ARA):
        stride = ctx.stride
        padding = ctx.padding
        groups = ctx.groups
        weight, bias = ctx.saved_tensors
        grad_weight = cudnn_convolution.convolution_backward_weight(ARA, weight.shape, grad_output, stride, padding, (1, 1), groups, False, False, False)
        grad_input = cudnn_convolution.convolution_backward_input(ARA.shape, weight, grad_output, stride, padding, (1, 1), groups, False, False, False)
        if bias is not None:
            grad_bias = grad_output.sum(dim=[0,2,3])
        return grad_input, ARA, grad_weight, grad_bias, None, None, None, None,None
    
class Distribute_ARA(nn.Module):
    '''
    When using stride to apply ARA, this layer is needed.
    If you deploy with x-> ARA_Conv2d - ARA_Conv2d - ARA_Conv2d -> y -> Distribute_ARA(y, x),
    the following three ARA_Conv2d will be trained using x.
    In this case, fisrt ARA_Conv2d act like backpropagation because it uses x which is input activation of itself. 
    '''
    def __init__(self, ):
        super(Distribute_ARA, self).__init__()
        
    def forward(self, input, auxiliary_activation):
        return distribute_ara.apply(input, auxiliary_activation)

class distribute_ara(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, auxiliary_activation):
        output = input.clone()
        ctx.save_for_backward(auxiliary_activation)
        return output, output.detach()
    
    @staticmethod
    def backward(ctx, grad_output, auxiliary_activation):
        auxiliary_activation, = ctx.saved_tensors
        return grad_output.clone(), auxiliary_activation.clone(), None


#%%
class Linear_ABA(nn.Linear):
    '''
    Linear_ABA uses auxiliary bias activation (ABA) to update weights in backward propagation.
    You can use get_li=True if you want to extract learning indicator.
    The learning rate can be enlarged by setting lr_expansion.
    In this layer, we do not auxiliary_activation to input because it has been already added outside of the module.
    '''
    def __init__(self, in_features, out_features, bias=True, get_li=False, lr_expansion= 100):
        super(Linear_ABA, self).__init__(in_features, out_features, bias=bias)
        self.ABA = nn.Parameter(torch.ones(in_features), requires_grad=True)
        bound1 = 1 / math.sqrt(in_features) if in_features > 0 else 0
        nn.init.uniform_(self.ABA, -bound1, bound1)
        self.get_li = get_li
        self.lr_expansion = lr_expansion
        
    def forward(self, input):
        if self.get_li and self.training:
            self.li = learning_indicator(input+self.ABA, self.ABA)
        return linear_aba.apply(input, self.ABA, self.weight, self.bias, self.lr_expansion)
        
class linear_aba(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, ABA, weight, bias, lr_expansion):
        output = F.linear(input + ABA, weight, bias)
        ctx.input_size = input.size()
        ctx.lr_expansion = lr_expansion
        ctx.save_for_backward(ABA, bias, weight)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        ABA, bias, weight = ctx.saved_tensors
        grad_output_size = grad_output.size()
        grad_output = grad_output.reshape(-1, grad_output_size[-1]) 
        ABA = ABA.unsqueeze(dim=0).repeat(grad_output.size(0),1)
        grad_weight = F.linear(ABA.t(), grad_output.t()).t()
        grad_weight *= ctx.lr_expansion
        grad_bias = None
        if bias is not None:
            grad_bias = grad_output.sum(0).squeeze(0)
        grad_input = F.linear(grad_output, weight.t())
        grad_ABA = grad_input.sum(0).squeeze(0)
        return grad_input.reshape(ctx.input_size), grad_ABA, grad_weight, grad_bias, None


class Conv2d_ABA(nn.Conv2d):
    '''
    Conv2d_ABA uses auxiliary bias activation (ABA) to update weights in backward propagation.
    You can use get_li=True if you want to extract learning indicator.
    The learning rate can be enlarged by setting lr_expansion.
    In this layer, we do not auxiliary_activation to input because it has been already added outside of the module.
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=False, get_li=False, lr_expansion=100):
        super(Conv2d_ABA, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=groups)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.ABA = nn.Parameter(torch.ones(in_channels),requires_grad=True)
        stdv = 1. / math.sqrt(in_channels)
        self.ABA.data.uniform_(-stdv, stdv)
        self.li=None
        self.get_li=get_li
        self.lr_expansion=lr_expansion
        
    def forward(self, input):
        if self.get_li and self.training:
            self.li = learning_indicator(input+self.ABA.reshape(1,-1,1,1), self.ABA.reshape(1,-1,1,1), convnet=True)
        
        return conv2d_aba.apply(input, self.ABA, self.weight, self.bias, self.stride, self.padding, self.groups, self.lr_expansion) 

class conv2d_aba(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, ABA, weight, bias, stride=1, padding=0, groups=1, lr_expansion=100):
        output = cudnn_convolution.convolution(input+ABA.reshape(1,-1,1,1), weight, bias, stride, padding, (1, 1), groups, False, False)
        ctx.stride = stride
        ctx.padding = padding
        ctx.groups = groups
        ctx.input_size = input.size()
        ctx.lr_expansion=lr_expansion
        ctx.save_for_backward(ABA, weight, bias)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        stride = ctx.stride
        padding = ctx.padding
        groups = ctx.groups
        ABA, weight, bias = ctx.saved_tensors
        ABA = ABA.view(1,-1,1,1).repeat(ctx.input_size[0],1,ctx.input_size[2],ctx.input_size[3])
        grad_weight = cudnn_convolution.convolution_backward_weight(ABA, weight.shape, grad_output, stride, padding, (1, 1), groups, False, False,False)
        grad_weight *= ctx.lr_expansion
        grad_input = cudnn_convolution.convolution_backward_input(ABA.shape, weight, grad_output, stride, padding, (1, 1), groups, False, False,False)
        if bias is not None:
            grad_bias = grad_output.sum(dim=[0,2,3])
        grad_ABA = grad_input.sum(dim=[0,2,3])
        return grad_input, grad_ABA, grad_weight, grad_bias, None, None, None, None,None,None

#%%
class Linear_ASA(nn.Linear):
    '''
    Linear_ABA uses auxiliary sign activation (ASA) to update weights in backward propagation.
    You can use get_li=True if you want to extract learning indicator.
    You have to use relu=True if Linear_ASA is displayed after ReLU function.
    The epsilon controls the magnitude of ARA to make comparable value to output activation.
    The learning rate can be enlarged by setting lr_expansion.
    In this layer, we do not auxiliary_activation to input because it has been already added outside of the module.
    '''
    def __init__(self, in_features, out_features, bias=True, get_li=False, relu=False, epsilon = 0.01, lr_expansion=100):
        super(Linear_ASA, self).__init__(in_features, out_features, bias=bias)
        self.get_li = get_li
        self.epsilon = epsilon
        self.relu = relu
        self.lr_expansion = lr_expansion
        
    def forward(self, input):
        ASA = input.sign().detach().clone()
        if self.get_li and self.training:
            self.li = learning_indicator(input+ASA*self.epsilon, ASA*self.epsilon)
        return linear_asa.apply(input, ASA, self.weight, self.bias, self.relu, self.epsilon, self.lr_expansion)
        
class linear_asa(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, ASA, weight, bias, relu=False, epsilon=0.01, lr_expansion=100):
        output = F.linear(input + ASA*epsilon, weight, bias)
        ctx.input_size = input.size()
        ctx.relu=relu
        #packbits
        ASA = (ASA>0).to(ASA.device)
        ASA, restore_size = packbits_one_bits(ASA)
        ctx.restore_size = restore_size
        ctx.epsilon=epsilon
        ctx.lr_expansion=lr_expansion
        ctx.save_for_backward(ASA, bias, weight)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        ASA, bias, weight = ctx.saved_tensors
        #unpackbit
        ASA = unpackbits_one_bits(ASA, ctx.restore_size, relu=ctx.relu)
        grad_output_size = grad_output.size()
        grad_output = grad_output.reshape(-1, grad_output_size[-1]) 
        ASA = ASA.reshape(-1,ASA.size(2))
        grad_weight = F.linear(ASA.t()*ctx.epsilon, grad_output.t()).t()
        grad_weight *= ctx.lr_expansion
        grad_bias = None
        if bias is not None:
            grad_bias = grad_output.sum(0).squeeze(0)
        grad_input = F.linear(grad_output, weight.t())
        return grad_input.reshape(ctx.input_size), None, grad_weight, grad_bias, None, None, None
    
class Conv2d_ASA(nn.Conv2d):
    '''
    Conv2d_ABA uses auxiliary sign activation (ASA) to update weights in backward propagation.
    You can use get_li=True if you want to extract learning indicator.
    The epsilon controls the magnitude of ARA to make comparable value to output activation.
    The learning rate can be enlarged by setting lr_expansion.
    In this layer, we do not auxiliary_activation to input because it has been already added outside of the module.
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=False, epsilon = 0.01, lr_expansion=100, get_li=False):
        super(Conv2d_ASA, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=groups)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.epsilon = epsilon
        self.lr_expansion=lr_expansion
        self.li=None
        self.get_li=get_li
        
    def forward(self, input):
        ASA = input.sign().detach().clone()
        if self.get_li and self.training:
            self.li = learning_indicator(input+self.epsilon*ASA, self.epsilon*ASA, convnet=True)
        return conv2d_asa.apply(input, ASA, self.weight, self.bias, self.stride, self.padding, self.groups, self.epsilon, self.lr_expansion) 

class conv2d_asa(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, ASA, weight, bias, stride=1, padding=0, groups=1, epsilon=0.01, lr_expansion=100):
        output = cudnn_convolution.convolution(input+ASA*epsilon, weight, bias, stride, padding, (1, 1), groups, False, False)
        ctx.stride = stride
        ctx.padding = padding
        ctx.groups = groups
        #packbits
        ASA = (ASA>0).to(ASA.device)
        ASA, restore_size = packbits_one_bits(ASA)
        ctx.restore_size = restore_size
        ctx.epsilon=epsilon
        ctx.lr_expansion=lr_expansion
        ctx.save_for_backward(ASA, weight, bias)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        stride = ctx.stride
        padding = ctx.padding
        groups = ctx.groups
        ASA, weight, bias = ctx.saved_tensors
        #unpackbit
        ASA = unpackbits_one_bits(ASA, ctx.restore_size, relu=True)
        grad_weight = cudnn_convolution.convolution_backward_weight(ASA*ctx.epsilon, weight.shape, grad_output, stride, padding, (1, 1), groups, False, False,False)
        grad_weight *= ctx.lr_expansion
        grad_input = cudnn_convolution.convolution_backward_input(ASA.shape, weight, grad_output, stride, padding, (1, 1), groups, False, False,False)
        if bias is not None:
            grad_bias = grad_output.sum(dim=[0,2,3])
        return grad_input, None, grad_weight, grad_bias, None, None, None, None,None, None, None


