import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.cpp_extension import load
from packbits import packbits_one_bits, unpackbits_one_bits
from Auxiliary_Activation_Learning import learning_indicator

cudnn_convolution = load(name="cudnn_convolution", sources=["../cudnn_conv.cpp"], verbose=True)
cudnn_batch_norm = load(name="cudnn_batchnorm", sources=["../batchnorm.cpp"], verbose=True)

#%%
'''
We make custom batchnorm layer.
Batchnorm2d can replace nn.BatchNorm2d
'''
class BatchNorm2d(torch.nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        self.momentum = momentum
        self.one_minus_momentum = 1-momentum
    
    def forward(self, input):
        self._check_input_dim(input)
        return batch_norm2d.apply(input, self.running_mean, self.running_var, self.weight, self.bias,
                self.training, self.track_running_stats, self.momentum, self.eps)

class batch_norm2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, running_mean, running_var, weight, bias, training, track_running_stats, momentum, eps):
        output, save_mean, save_var, reservedspace = cudnn_batch_norm.batch_norm(input, weight, bias, running_mean, running_var, training, momentum, eps)
        ctx.save_for_backward(input, weight, running_mean, running_var, save_mean, save_var, reservedspace)
        ctx.eps =eps
        return output 
    
    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        input, weight, running_mean, running_var, save_mean, save_var, reservedspace = ctx.saved_tensors
        grad_input, grad_weight, grad_bias = cudnn_batch_norm.batch_norm_backward(input, grad_output, weight, running_mean, running_var, save_mean, save_var, eps, reservedspace)
        return grad_input, None, None, grad_weight, grad_bias, None, None, None, None

def batchnorm_forward(input, weight, bias, training, track_running_stats, running_mean, running_var, momentum, eps, backward = False):
    N = input.size(1) # channel
    input = input.permute(0,2,3,1).contiguous()
    input_shape = input.shape
    input = input.view(-1,N)
    if training:
        mu = input.mean(0)
        var = torch.var(input,0, unbiased=False)
        if track_running_stats and not(backward):
            running_mean.data = running_mean.mul(1-momentum).add(mu.mul(momentum)).data
            running_var.data = running_var.mul(1-momentum).add(var.mul(momentum)).data
        sqrt = torch.sqrt(var+eps).reciprocal()
        mu = mu.mul(sqrt)
        weight_div_sqrt = weight.mul(sqrt)
        y = input * weight_div_sqrt + bias.add(-mu*weight)
        return y.view(input_shape).permute(0,3,1,2).contiguous(), mu, weight_div_sqrt, sqrt
        
    else:
        y = input * weight.div(torch.sqrt(running_var+eps)) \
            + bias.add(-running_mean.div(torch.sqrt(running_var+eps)).mul(weight))
        return y.view(input_shape).permute(0,3,1,2).contiguous(), None, None, None


def batchnorm_backward(out, weight, bias, grad_output, mu_div_sqrt, weight_div_sqrt, sqrt, approximate_input = False):
    N = out.size(1) # channel
    out = out.permute(0,2,3,1).contiguous()
    out = out.view(-1,N)
    
    if approximate_input:
        out *= sqrt
        out -= mu_div_sqrt
    else:
        out -= bias
        out /= weight
        
    grad_out = grad_output.permute(0,2,3,1).contiguous()
    grad_shape = grad_out.shape
    grad_out = grad_out.view(-1, N)

    grad_bias = torch.sum(grad_out, 0)
    grad_weight = torch.sum(out*grad_out, 0)
    grad_input = weight_div_sqrt*(grad_out - grad_weight*out/grad_out.size(0) - grad_bias/grad_out.size(0) )
    
    grad_input = grad_input.view(grad_shape).permute(0,3,1,2).contiguous()
    return grad_input, grad_weight, grad_bias

#%%
'''
We make custom relu layer.
ReLU can replace nn.ReLU.
When packbits=True, the nn.ReLU just store packed Tensor instead of 1byte Tensor.
'''
class ReLU(nn.Module):
    def __init__(self, packbits = True):
        super(ReLU, self).__init__()
        self.packbits = packbits
        
    def forward(self, input):
        return relu.apply(input, self.packbits)

class relu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, packbits):
        input = input.clamp(min=0)
        output = input.clone()
        if packbits:
            input = (input>0).to(input.device)
            input, size = packbits_one_bits(input)
            ctx.size = size
        ctx.packbits = packbits
        ctx.save_for_backward(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        packbits = ctx.packbits
        input, =  ctx.saved_tensors
        if packbits:
            size = ctx.size
            input = unpackbits_one_bits(input, size)
        grad_input = grad_output.clone()
        grad_input[input <= 0] = 0
        return grad_input, None

#%%
'''
The torch.utils.checkpoint for gradinet checkpointing is slow in ResNet.
Therfore, we make custom gradient checkpoint layer
:BnReLUConv, BnReLUConvBn, ConvBn_ARA
It only stores inpuy activation of layer and do recomputation during backward propagation
''' 
class BnReLUConv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, momentum=0.1, eps=1e-5, track_running_stats=True, bias=False):
        super(BnReLUConv, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=groups)
        self.bn_weight = nn.Parameter(torch.ones(in_channels))
        self.bn_bias = nn.Parameter(torch.zeros(in_channels))
        self.register_buffer('running_mean', torch.zeros(in_channels))
        self.register_buffer('running_var', torch.zeros(in_channels))
        self.momentum = momentum
        self.eps = eps
        self.track_running_stats = track_running_stats
        
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    
    def forward(self, input):
        return bnreluconv.apply(input, self.weight, self.bias, self.bn_weight, self.bn_bias, self.running_mean, self.running_var
                                ,self.stride, self.padding, self.groups, self.momentum, self.eps, self.track_running_stats, self.training) 
        

class bnreluconv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, bn_weight, bn_bias, running_mean, running_var,
                stride=1, padding=0, groups=1, momentum=0.1, eps=1e-5, track_running_stats = True, training = True):
        ############# Doing batchnorm ##################
        #out, input, mu, weight_div_sqrt, sqrt = batchnorm_forward(input, bn_weight, bn_bias, training, track_running_stats, running_mean, running_var, momentum, eps)    
        out, save_mean, save_var, reservedspace = cudnn_batch_norm.batch_norm(input, bn_weight, bn_bias, running_mean, running_var, training, momentum, eps)
        
        ############# Doing ReLU ##################        
        out = out.clamp(min=0)
        
        ############# Doing Conv2d_AS ##################
        out = cudnn_convolution.convolution(out, weight, bias, stride, padding, (1, 1), groups, False, False)

        ############# Save for backward ##################
        ctx.stride = stride
        ctx.padding = padding
        ctx.groups = groups
        ctx.training = training
        ctx.momentum = momentum
        ctx.eps = eps
        ctx.save_for_backward(input, bn_weight, bn_bias, running_mean, running_var, save_mean, save_var, reservedspace, weight, bias)  
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        ############# Load saved tensors & recomputation ##################
        stride = ctx.stride
        padding = ctx.padding
        groups = ctx.groups
        training = ctx.training
        momentum = ctx.momentum
        eps = ctx.eps
        
        input, bn_weight, bn_bias, running_mean, running_var, save_mean, save_var, reversedspace, weight, bias = ctx.saved_tensors
        running_mean_d = torch.zeros_like(running_mean).to(input.device)
        running_var_d = torch.zeros_like(running_var).to(input.device)
 
        out, save_mean, save_var, reservedspace = cudnn_batch_norm.batch_norm(input, bn_weight, bn_bias, running_mean_d, running_var_d, training, momentum, eps)
        out = out.clamp(min=0)
        
        ############# Doing Conv2d_AS Backward ##################
        grad_weight = cudnn_convolution.convolution_backward_weight(out, weight.shape, grad_output, stride, padding, (1, 1), groups, False, False, False)
        if bias is not None:
            grad_bias = grad_output.sum(dim=[0,2,3])
        grad_input = cudnn_convolution.convolution_backward_input(out.shape, weight, grad_output, stride, padding, (1, 1), groups, False, False, False)
         
        ############# Doing ReLU Backward ##################
        grad_input = grad_input.clone()
        grad_input[out <= 0] = 0
        
        ############# Doing batchnorm ##################
        grad_input, grad_bn_weight, grad_bn_bias = cudnn_batch_norm.batch_norm_backward(input, grad_input, bn_weight, running_mean, running_var, save_mean, save_var, eps, reservedspace)
       
        return grad_input, grad_weight, grad_bias, grad_bn_weight, grad_bn_bias, None, None, None, None,None, None, None, None, None        


class BnReLUConvBn(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, momentum=0.1, eps=1e-5, track_running_stats=True, bias=False):
        super(BnReLUConvBn, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=groups)
        self.bn_weight = nn.Parameter(torch.ones(in_channels))
        self.bn_bias = nn.Parameter(torch.zeros(in_channels))
        self.register_buffer('running_mean', torch.zeros(in_channels))
        self.register_buffer('running_var', torch.zeros(in_channels))
        self.obn_weight = nn.Parameter(torch.ones(out_channels))
        self.obn_bias = nn.Parameter(torch.zeros(out_channels))
        self.register_buffer('running_mean_o', torch.zeros(out_channels))
        self.register_buffer('running_var_o', torch.zeros(out_channels))
        self.momentum = momentum
        self.eps = eps
        self.track_running_stats = track_running_stats
        
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
    def forward(self, input):
        return bnreluconvbn.apply(input, self.weight, self.bias
                                  ,self.bn_weight, self.bn_bias, self.obn_weight, self.obn_bias
                                  ,self.running_mean, self.running_var, self.running_mean_o, self.running_var_o
                                ,self.stride, self.padding, self.groups, self.momentum, self.eps, self.track_running_stats, self.training) 


class bnreluconvbn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, bn_weight, bn_bias, obn_weight, obn_bias, running_mean, running_var,  running_mean_o, running_var_o,
                stride=1, padding=0, groups=1, momentum=0.1, eps=1e-5, track_running_stats = True, training = True):
        ############# Doing batchnorm ##################
        out_b, save_mean, save_var, reservedspace = cudnn_batch_norm.batch_norm(input, bn_weight, bn_bias, running_mean, running_var, training, momentum, eps)
        
        ############# Doing ReLU ##################        
        out_r = out_b.clamp(min=0)
        
        ############# Doing Conv2d_AS ##################
        out_bb = cudnn_convolution.convolution(out_r, weight, bias, stride, padding, (1, 1), groups, False, False)
        
        ############# Doing out batchnorm ##################
        out, save_mean_o, save_var_o, reservedspace_o = cudnn_batch_norm.batch_norm(out_bb, obn_weight, obn_bias, running_mean_o, running_var_o, training, momentum, eps)
        
        ############# Save for backward ##################
        ctx.stride = stride
        ctx.padding = padding
        ctx.groups = groups
        ctx.training = training
        ctx.track_running_statas = track_running_stats
        ctx.momentum = momentum
        ctx.eps = eps
        
        ctx.save_for_backward(input, bn_weight, bn_bias, running_mean, running_var, save_mean, save_var, reservedspace,
                                                                 # save for relu : None (do recomputation by input)
                              weight, bias,  # save for conv_as : mainly None(do recomputation by input) 
                              obn_weight, obn_bias, running_mean_o, running_var_o, save_mean_o, save_var_o, reservedspace_o,
                              )  # mainly None(do recomputation by input)
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        ############# Load saved tensors & recomputation ##################
        stride = ctx.stride
        padding = ctx.padding
        groups = ctx.groups
        training = ctx.training
        momentum = ctx.momentum
        eps = ctx.eps
        
        input, bn_weight, bn_bias, running_mean, running_var, save_mean, save_var, reservedspace, weight, bias, obn_weight, obn_bias, running_mean_o, running_var_o, save_mean_o, save_var_o, reservedspace_o = ctx.saved_tensors
        running_mean_d = torch.zeros_like(running_mean).to(input.device)
        running_var_d = torch.zeros_like(running_var).to(input.device)
        out, save_mean, save_var, reservedspace = cudnn_batch_norm.batch_norm(input, bn_weight, bn_bias, running_mean_d, running_var_d, training, momentum, eps)
        out_ = out.clamp(min=0)
        out_b = cudnn_convolution.convolution(out_, weight, bias, stride, padding, (1, 1), groups, False, False)
        
        ############# Doing batchnorm ##################
        grad_input, grad_obn_weight, grad_obn_bias = cudnn_batch_norm.batch_norm_backward(out_b, grad_output, obn_weight, running_mean_o, running_var_o, save_mean_o, save_var_o, eps, reservedspace_o)
        
        ############# Doing Conv2d_AS Backward ##################
        grad_weight = cudnn_convolution.convolution_backward_weight(out_, weight.shape, grad_input, stride, padding, (1, 1), groups, False, False, False)
        if bias is not None:
            grad_bias = grad_input.sum(dim=[0,2,3])
        grad_input = cudnn_convolution.convolution_backward_input(out_.shape, weight, grad_input, stride, padding, (1, 1), groups, False, False, False)
            
        ############# Doing ReLU Backward ##################
        grad_input = grad_input.clone()
        grad_input[out <= 0] = 0
        
        ############# Doing batchnorm ##################
        grad_input, grad_bn_weight, grad_bn_bias = cudnn_batch_norm.batch_norm_backward(input, grad_input, bn_weight, running_mean, running_var, save_mean, save_var, eps, reservedspace)
        
        return grad_input, grad_weight, grad_bias, grad_bn_weight, grad_bn_bias, grad_obn_weight, grad_obn_bias, None, None, None, None,None, None, None, None, None, None, None , None

class ConvBn_ARA(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, momentum=0.1, eps=1e-5, track_running_stats=True, bias=False):
        super(ConvBn_ARA, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=groups)
        self.bn_weight = nn.Parameter(torch.ones(out_channels))
        self.bn_bias = nn.Parameter(torch.zeros(out_channels))
        self.register_buffer('running_mean', torch.zeros(out_channels))
        self.register_buffer('running_var', torch.zeros(out_channels))
        self.momentum = momentum
        self.eps = eps
        self.track_running_stats = track_running_stats
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
    def forward(self, input, ARA):
        return convbn_ara.apply(input, ARA, self.weight, self.bias, self.bn_weight, self.bn_bias, self.running_mean, self.running_var
                                ,self.stride, self.padding, self.groups, self.momentum, self.eps, self.track_running_stats, self.training) 

class convbn_ara(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, ARA, weight, bias, bn_weight, bn_bias, running_mean, running_var,
                stride=1, padding=0, groups=1, momentum=0.1, eps=1e-5, track_running_stats = True, training = True):
        ############# Doing Conv2d ##################
        out = cudnn_convolution.convolution(input, weight, bias, stride, padding, (1, 1), groups, False, False)
        
        ############# Doing batchnorm ##################
        #out, input, mu, weight_div_sqrt, sqrt = batchnorm_forward(out, bn_weight, bn_bias, training, track_running_stats, running_mean, running_var, momentum, eps)    
        out, save_mean, save_var, reservedspace = cudnn_batch_norm.batch_norm(out, bn_weight, bn_bias, running_mean, running_var, training, momentum, eps)
        
        ############# Save for backward ##################
        ctx.stride = stride
        ctx.padding = padding
        ctx.groups = groups
        ctx.eps =eps
        
        ctx.save_for_backward(bn_weight, bn_bias, running_mean, running_var, save_mean, save_var, reservedspace, weight, bias)  
        return out, ARA.clone()
    
    @staticmethod
    def backward(ctx, grad_output, ARA):
        ############# Load saved tensors & recomputation ##################
        stride = ctx.stride
        padding = ctx.padding
        groups = ctx.groups
        eps = ctx.eps
        bn_weight, bn_bias, running_mean, running_var, save_mean, save_var, reservedspace, weight, bias = ctx.saved_tensors
        
        out = cudnn_convolution.convolution(ARA, weight, bias, stride, padding, (1, 1), groups, False, False)
        
        ############# Doing batchnorm ##################
        #grad_input, grad_bn_weight, grad_bn_bias = batchnorm_backward(out, grad_output, mu_div_sqrt, weight_div_sqrt, sqrt)
        grad_input, grad_bn_weight, grad_bn_bias = cudnn_batch_norm.batch_norm_backward(out, grad_output, bn_weight, running_mean, running_var, save_mean, save_var, eps, reservedspace)
    
        ############# Doing Conv2d_AS Backward ##################
        grad_weight = cudnn_convolution.convolution_backward_weight(ARA, weight.shape, grad_input, stride, padding, (1, 1), groups, False, False, False)
        if bias is not None:
            grad_bias = grad_input.sum(dim=[0,2,3])
        grad_input = cudnn_convolution.convolution_backward_input(ARA.shape, weight, grad_input, stride, padding, (1, 1), groups, False, False, False)
        
        return grad_input, ARA, grad_weight, grad_bias, grad_bn_weight, grad_bn_bias, None, None, None, None,None, None, None, None, None
    
#%%
'''
In self-attention layer, q,k, and v is made by one or two input using three weights.
Therefore, we make 

Three_Linear_One_Input, Three_Linear_One_Input_ABA, Three_Linear_One_Input_ASA,
Three_Linear_Two_Input, Three_Linear_Two_Input_ABA, Three_Linear_Two_Input_ASA,

for conserving training memory by only saving one or two input to train three weights.
'''
class  Three_Linear_One_Input(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(Three_Linear_One_Input, self).__init__(in_features, out_features, bias=bias)
        self.in_features = in_features
        self.out_features = out_features
        self.weight1 = nn.Parameter(torch.Tensor(self.weight.size()), requires_grad=True)
        self.weight2 = nn.Parameter(torch.Tensor(self.weight.size()), requires_grad=True)
        nn.init.kaiming_uniform_(self.weight1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight2, a=math.sqrt(5))
        
        if bias:
            self.bias1 = nn.Parameter(torch.Tensor(self.bias.size()), requires_grad=True)
            self.bias2 = nn.Parameter(torch.Tensor(self.bias.size()), requires_grad=True)
            
            fan_in1, _ = nn.init._calculate_fan_in_and_fan_out(self.weight1)
            bound1 = 1 / math.sqrt(fan_in1) if fan_in1 > 0 else 0
            nn.init.uniform_(self.bias1, -bound1, bound1)
            
            fan_in2, _ = nn.init._calculate_fan_in_and_fan_out(self.weight2)
            bound2 = 1 / math.sqrt(fan_in2) if fan_in2 > 0 else 0
            nn.init.uniform_(self.bias2, -bound2, bound2)
        else:
            self.bias = self.bias1 = self.bias2 = None
            
    def forward(self, input):
        return three_linear_one_input.apply(input, self.weight, self.weight1, self.weight2, self.bias, self.bias1, self.bias2)


class three_linear_one_input(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, weight1, weight2, bias, bias1, bias2):
        output = F.linear(input, weight, bias)
        output1 = F.linear(input, weight1, bias1)
        output2 = F.linear(input, weight2, bias2)
        
        ctx.save_for_backward(input, weight, weight1, weight2, bias, bias1, bias2)
        return output, output1, output2

    @staticmethod
    def backward(ctx, grad_output, grad_output1, grad_output2):
        input, weight, weight1, weight2, bias, bias1, bias2 = ctx.saved_tensors
        input_size = input.size()
        grad_output_size = grad_output.size()
        grad_output1_size = grad_output1.size()
        grad_output2_size = grad_output2.size()
        input = input.view(-1, input_size[-1])
        grad_output = grad_output.reshape(-1, grad_output_size[-1])        
        grad_output1 = grad_output1.reshape(-1, grad_output1_size[-1])        
        grad_output2 = grad_output2.reshape(-1, grad_output2_size[-1])        
        
        grad_weight = F.linear(input.t(), grad_output.t()).t()
        grad_weight1 = F.linear(input.t(), grad_output1.t()).t()
        grad_weight2 = F.linear(input.t(), grad_output2.t()).t()
        
        grad_bias = grad_bias1 = grad_bias2 = None
        if bias is not None:
            grad_bias = grad_output.sum(0).squeeze(0)
            grad_bias1 = grad_output1.sum(0).squeeze(0)
            grad_bias2 = grad_output2.sum(0).squeeze(0)
        
        grad_input = F.linear(grad_output, weight.t())
        grad_input1 = F.linear(grad_output1, weight1.t())
        grad_input2 = F.linear(grad_output2, weight2.t())
        return (grad_input + grad_input1 + grad_input2).reshape(input_size), grad_weight, grad_weight1, grad_weight2, grad_bias, grad_bias1, grad_bias2

class  Three_Linear_Two_Input(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(Three_Linear_Two_Input, self).__init__(in_features, out_features, bias=bias)
        self.in_features = in_features
        self.out_features = out_features
        self.weight1 = nn.Parameter(torch.Tensor(self.weight.size()), requires_grad=True)
        self.weight2 = nn.Parameter(torch.Tensor(self.weight.size()), requires_grad=True)
        nn.init.kaiming_uniform_(self.weight1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight2, a=math.sqrt(5))
        
        if bias:
            self.bias1 = nn.Parameter(torch.Tensor(self.bias.size()), requires_grad=True)
            self.bias2 = nn.Parameter(torch.Tensor(self.bias.size()), requires_grad=True)
            
            fan_in1, _ = nn.init._calculate_fan_in_and_fan_out(self.weight1)
            bound1 = 1 / math.sqrt(fan_in1) if fan_in1 > 0 else 0
            nn.init.uniform_(self.bias1, -bound1, bound1)
            
            fan_in2, _ = nn.init._calculate_fan_in_and_fan_out(self.weight2)
            bound2 = 1 / math.sqrt(fan_in2) if fan_in2 > 0 else 0
            nn.init.uniform_(self.bias2, -bound2, bound2)
        else:
            self.bias = self.bias1 = self.bias2 = None
            
        
    def forward(self, input, input12):
        return three_linear_two_input.apply(input, input12, self.weight, self.weight1, self.weight2, self.bias, self.bias1, self.bias2)


class three_linear_two_input(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, input12, weight, weight1, weight2, bias, bias1, bias2):
        output = F.linear(input, weight, bias)
        output1 = F.linear(input12, weight1, bias1)
        output2 = F.linear(input12, weight2, bias2)
        
        ctx.save_for_backward(input, input12, weight, weight1, weight2, bias, bias1, bias2)
        return output, output1, output2

    @staticmethod
    def backward(ctx, grad_output, grad_output1, grad_output2):
        input, input12, weight, weight1, weight2, bias, bias1, bias2 = ctx.saved_tensors
        input_size = input.size()
        input12_size = input12.size()
        grad_output_size = grad_output.size()
        grad_output1_size = grad_output1.size()
        grad_output2_size = grad_output2.size()
        input = input.view(-1, input_size[-1])
        input12 = input12.view(-1, input12_size[-1])
        grad_output = grad_output.reshape(-1, grad_output_size[-1])        
        grad_output1 = grad_output1.reshape(-1, grad_output1_size[-1])        
        grad_output2 = grad_output2.reshape(-1, grad_output2_size[-1])        
        
        grad_weight = F.linear(input.t(), grad_output.t()).t()
        grad_weight1 = F.linear(input12.t(), grad_output1.t()).t()
        grad_weight2 = F.linear(input12.t(), grad_output2.t()).t()
        
        grad_bias = grad_bias1 = grad_bias2 = None
        if bias is not None:
            grad_bias = grad_output.sum(0).squeeze(0)
            grad_bias1 = grad_output1.sum(0).squeeze(0)
            grad_bias2 = grad_output2.sum(0).squeeze(0)
        
        grad_input = F.linear(grad_output, weight.t())
        grad_input1 = F.linear(grad_output1, weight1.t())
        grad_input2 = F.linear(grad_output2, weight2.t())
        
        return grad_input.reshape(input_size), (grad_input1 + grad_input2).reshape(input12_size), grad_weight, grad_weight1, grad_weight2, grad_bias, grad_bias1, grad_bias2

class  Three_Linear_One_Input_ABA(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, get_li=False, lr_expansion = 100):
        super(Three_Linear_One_Input_ABA, self).__init__(in_features, out_features, bias=bias)
        self.in_features = in_features
        self.out_features = out_features
        self.get_li = get_li
        self.weight1 = nn.Parameter(torch.Tensor(self.weight.size()), requires_grad=True)
        self.weight2 = nn.Parameter(torch.Tensor(self.weight.size()), requires_grad=True)
        nn.init.kaiming_uniform_(self.weight1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight2, a=math.sqrt(5))
         
        if bias:
            self.bias1 = nn.Parameter(torch.Tensor(self.bias.size()), requires_grad=True)
            self.bias2 = nn.Parameter(torch.Tensor(self.bias.size()), requires_grad=True)
            
            fan_in1, _ = nn.init._calculate_fan_in_and_fan_out(self.weight1)
            bound1 = 1 / math.sqrt(fan_in1) if fan_in1 > 0 else 0
            nn.init.uniform_(self.bias1, -bound1, bound1)
            
            fan_in2, _ = nn.init._calculate_fan_in_and_fan_out(self.weight2)
            bound2 = 1 / math.sqrt(fan_in2) if fan_in2 > 0 else 0
            nn.init.uniform_(self.bias2, -bound2, bound2)
        else:
            self.bias = self.bias1 = self.bias2 = None
            
        self.ABA = nn.Parameter(torch.ones(in_features), requires_grad=True)
        self.ABA1 = nn.Parameter(torch.ones(in_features), requires_grad=True)
        self.ABA2 = nn.Parameter(torch.ones(in_features), requires_grad=True)
        
        bound_ = 1 / math.sqrt(in_features) if in_features > 0 else 0
        nn.init.uniform_(self.ABA, -bound_, bound_)
        nn.init.uniform_(self.ABA1, -bound_, bound_)
        nn.init.uniform_(self.ABA2, -bound_, bound_)
        
        self.lr_expansion = lr_expansion
        
    def forward(self, input):
        if self.training and self.get_li:
            li = learning_indicator(input+self.ABA, self.ABA)
            li1 = learning_indicator(input+self.ABA1, self.ABA1)
            li2 = learning_indicator(input+self.ABA2, self.ABA2)
            self.li = li+li1+li2
        return three_linear_one_input_aba.apply(input, self.weight, self.weight1, self.weight2, self.bias, self.bias1, self.bias2, self.ABA, self.ABA1, self.ABA2, self.lr_expansion)


class three_linear_one_input_aba(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, weight1, weight2, bias, bias1, bias2, ABA, ABA1, ABA2, lr_expansion):
        output = F.linear(input+ABA, weight, bias)
        output1 = F.linear(input+ABA1, weight1, bias1)
        output2 = F.linear(input+ABA2, weight2, bias2)
        
        ctx.lr_expansion = lr_expansion
        ctx.input_size = input.size()
        
        ctx.save_for_backward(ABA, ABA1, ABA2, weight, weight1, weight2, bias, bias1, bias2)
        return output, output1, output2

    @staticmethod
    def backward(ctx, grad_output, grad_output1, grad_output2):
        ABA, ABA1, ABA2, weight, weight1, weight2, bias, bias1, bias2 = ctx.saved_tensors
        
        grad_output_size = grad_output.size()
        grad_output1_size = grad_output1.size()
        grad_output2_size = grad_output2.size()
        grad_output = grad_output.reshape(-1, grad_output_size[-1])        
        grad_output1 = grad_output1.reshape(-1, grad_output1_size[-1])        
        grad_output2 = grad_output2.reshape(-1, grad_output2_size[-1])        
        
        ABA = ABA.unsqueeze(dim=0).repeat(grad_output.size(0),1)
        ABA1 = ABA1.unsqueeze(dim=0).repeat(grad_output1.size(0),1)
        ABA2 = ABA2.unsqueeze(dim=0).repeat(grad_output2.size(0),1)
        
        
        grad_weight = F.linear(ABA.t(), grad_output.t()).t()
        grad_weight1 = F.linear(ABA1.t(), grad_output1.t()).t()
        grad_weight2 = F.linear(ABA2.t(), grad_output2.t()).t()
        
        grad_weight *= ctx.lr_expansion
        grad_weight1 *= ctx.lr_expansion
        grad_weight2 *= ctx.lr_expansion
        
        grad_bias = grad_bias1 = grad_bias2 = None
        if bias is not None:
            grad_bias = grad_output.sum(0).squeeze(0)
            grad_bias1 = grad_output1.sum(0).squeeze(0)
            grad_bias2 = grad_output2.sum(0).squeeze(0)
        
        grad_input = F.linear(grad_output, weight.t())
        grad_input1 = F.linear(grad_output1, weight1.t())
        grad_input2 = F.linear(grad_output2, weight2.t())
        
        grad_ABA = grad_input.sum(0).squeeze(0)
        grad_ABA1 = grad_input1.sum(0).squeeze(0)
        grad_ABA2 = grad_input2.sum(0).squeeze(0)
        
        return (grad_input + grad_input1 + grad_input2).reshape(ctx.input_size), grad_weight, grad_weight1, grad_weight2, grad_bias, grad_bias1, grad_bias2, grad_ABA, grad_ABA1, grad_ABA2, None
    
#%%
class  Three_Linear_Two_Input_ABA(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, get_li=False, lr_expansion=100):
        super(Three_Linear_Two_Input_ABA, self).__init__(in_features, out_features, bias=bias)
        self.in_features = in_features
        self.out_features = out_features
        self.weight1 = nn.Parameter(torch.Tensor(self.weight.size()), requires_grad=True)
        self.weight2 = nn.Parameter(torch.Tensor(self.weight.size()), requires_grad=True)
        self.get_li = get_li
        self.lr_expansion=lr_expansion
        
        nn.init.kaiming_uniform_(self.weight1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight2, a=math.sqrt(5))
        
        if bias:
            self.bias1 = nn.Parameter(torch.Tensor(self.bias.size()), requires_grad=True)
            self.bias2 = nn.Parameter(torch.Tensor(self.bias.size()), requires_grad=True)
            
            fan_in1, _ = nn.init._calculate_fan_in_and_fan_out(self.weight1)
            bound1 = 1 / math.sqrt(fan_in1) if fan_in1 > 0 else 0
            nn.init.uniform_(self.bias1, -bound1, bound1)
            
            fan_in2, _ = nn.init._calculate_fan_in_and_fan_out(self.weight2)
            bound2 = 1 / math.sqrt(fan_in2) if fan_in2 > 0 else 0
            nn.init.uniform_(self.bias2, -bound2, bound2)
        else:
            self.bias = self.bias1 = self.bias2 = None
            
        self.ABA = nn.Parameter(torch.ones(in_features), requires_grad=True)
        self.ABA1 = nn.Parameter(torch.ones(in_features), requires_grad=True)
        self.ABA2 = nn.Parameter(torch.ones(in_features), requires_grad=True)
        
        bound_ = 1 / math.sqrt(in_features) if in_features > 0 else 0
        nn.init.uniform_(self.ABA, -bound_, bound_)
        nn.init.uniform_(self.ABA1, -bound_, bound_)
        nn.init.uniform_(self.ABA2, -bound2, bound2)
        
    def forward(self, input, input12):
        if self.training and self.get_li:
            li = learning_indicator(input+self.ABA, self.ABA)
            li1 = learning_indicator(input12+self.ABA1, self.ABA1)
            li2 = learning_indicator(input12+self.ABA2, self.ABA2)
            self.li = li+li1+li2
        return three_linear_two_input_aba.apply(input, input12, self.weight, self.weight1, self.weight2, self.bias, self.bias1, self.bias2, self.ABA, self.ABA1, self.ABA2, self.lr_expansion)


class three_linear_two_input_aba(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, input12, weight, weight1, weight2, bias, bias1, bias2, ABA, ABA1, ABA2, lr_expansion):
        output = F.linear(input+ABA, weight, bias)
        output1 = F.linear(input12+ABA1, weight1, bias1)
        output2 = F.linear(input12+ABA2, weight2, bias2)
        
        ctx.lr_expansion = lr_expansion
        ctx.input_size = input.size()
        ctx.input12_size = input12.size()
        ctx.save_for_backward(ABA, ABA1, ABA2, weight, weight1, weight2, bias, bias1, bias2)
        return output, output1, output2

    @staticmethod
    def backward(ctx, grad_output, grad_output1, grad_output2):
        ABA, ABA1, ABA2, weight, weight1, weight2, bias, bias1, bias2 = ctx.saved_tensors
        
        grad_output_size = grad_output.size()
        grad_output1_size = grad_output1.size()
        grad_output2_size = grad_output2.size()
        
        grad_output = grad_output.reshape(-1, grad_output_size[-1])        
        grad_output1 = grad_output1.reshape(-1, grad_output1_size[-1])        
        grad_output2 = grad_output2.reshape(-1, grad_output2_size[-1])        
        
        ABA = ABA.unsqueeze(dim=0).repeat(grad_output.size(0),1)
        ABA1 = ABA1.unsqueeze(dim=0).repeat(grad_output1.size(0),1)
        ABA2 = ABA2.unsqueeze(dim=0).repeat(grad_output2.size(0),1)
        
        grad_weight = F.linear(ABA.t(), grad_output.t()).t() 
        grad_weight1 = F.linear(ABA1.t(), grad_output1.t()).t()
        grad_weight2 = F.linear(ABA2.t(), grad_output2.t()).t()
        
        grad_weight *= ctx.lr_expansion
        grad_weight1 *= ctx.lr_expansion
        grad_weight2 *= ctx.lr_expansion
        
        grad_bias = grad_bias1 = grad_bias2 = None
        if bias is not None:
            grad_bias = grad_output.sum(0).squeeze(0)
            grad_bias1 = grad_output1.sum(0).squeeze(0)
            grad_bias2 = grad_output2.sum(0).squeeze(0)
        
        grad_input = F.linear(grad_output, weight.t())
        grad_input1 = F.linear(grad_output1, weight1.t())
        grad_input2 = F.linear(grad_output2, weight2.t())
        
        grad_ABA = grad_input.sum(0).squeeze(0)
        grad_ABA1 = grad_input1.sum(0).squeeze(0)
        grad_ABA2 = grad_input2.sum(0).squeeze(0)
        
        return grad_input.reshape(ctx.input_size), (grad_input1 + grad_input2).reshape(ctx.input12_size), grad_weight, grad_weight1, grad_weight2, grad_bias, grad_bias1, grad_bias2, grad_ABA, grad_ABA1, grad_ABA2, None

class  Three_Linear_One_Input_ASA(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, epsilon = 0.01, get_li=False, lr_expansion=100):
        super(Three_Linear_One_Input_ASA, self).__init__(in_features, out_features, bias=bias)
        self.in_features = in_features
        self.out_features = out_features
        self.get_li = get_li
        self.weight1 = nn.Parameter(torch.Tensor(self.weight.size()), requires_grad=True)
        self.weight2 = nn.Parameter(torch.Tensor(self.weight.size()), requires_grad=True)
        self.epsilon = epsilon
        self.lr_expansion=lr_expansion
        
        nn.init.kaiming_uniform_(self.weight1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight2, a=math.sqrt(5))
         
        if bias:
            self.bias1 = nn.Parameter(torch.Tensor(self.bias.size()), requires_grad=True)
            self.bias2 = nn.Parameter(torch.Tensor(self.bias.size()), requires_grad=True)
            
            fan_in1, _ = nn.init._calculate_fan_in_and_fan_out(self.weight1)
            bound1 = 1 / math.sqrt(fan_in1) if fan_in1 > 0 else 0
            nn.init.uniform_(self.bias1, -bound1, bound1)
            
            fan_in2, _ = nn.init._calculate_fan_in_and_fan_out(self.weight2)
            bound2 = 1 / math.sqrt(fan_in2) if fan_in2 > 0 else 0
            nn.init.uniform_(self.bias2, -bound2, bound2)
        else:
            self.bias = self.bias1 = self.bias2 = None
            
    def forward(self, input):
        ASA = input.sign().detach().clone()
        if self.training and self.get_li:
            li = learning_indicator(input+self.epsilon*ASA, self.epsilon*ASA)
            self.li = li*3
            
        return three_linear_one_input_asa.apply(input, ASA, self.weight, self.weight1, self.weight2, self.bias, self.bias1, self.bias2, self.epsilon, self.lr_expansion)


class three_linear_one_input_asa(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, ASA, weight, weight1, weight2, bias, bias1, bias2, epsilon, lr_expansion):
        output = F.linear(input+ASA*epsilon, weight, bias)
        output1 = F.linear(input+ASA*epsilon, weight1, bias1)
        output2 = F.linear(input+ASA*epsilon, weight2, bias2)
        
        ctx.mag_ratio = torch.norm(input+ASA*epsilon)/(torch.norm(ASA)+1e-6)
        ctx.epsilon = epsilon
        ctx.input_size = input.size()
        ctx.lr_expansion = lr_expansion
        
        #packbits
        ASA = (ASA>0).to(ASA.device)
        ASA, restore_size = packbits_one_bits(ASA)
        ctx.restore_size = restore_size
        
        ctx.save_for_backward(ASA, weight, weight1, weight2, bias, bias1, bias2)
        return output, output1, output2

    @staticmethod
    def backward(ctx, grad_output, grad_output1, grad_output2):
        ASA, weight, weight1, weight2, bias, bias1, bias2 = ctx.saved_tensors
        
        grad_output_size = grad_output.size()
        grad_output1_size = grad_output1.size()
        grad_output2_size = grad_output2.size()
        grad_output = grad_output.reshape(-1, grad_output_size[-1])        
        grad_output1 = grad_output1.reshape(-1, grad_output1_size[-1])        
        grad_output2 = grad_output2.reshape(-1, grad_output2_size[-1])        
        
        #unpackbit
        ASA = unpackbits_one_bits(ASA, ctx.restore_size)
        ASA = ASA.reshape(-1,ASA.size(2))
        
        grad_weight = F.linear(ASA.t(), grad_output.t()).t()
        grad_weight1 = F.linear(ASA.t(), grad_output1.t()).t()
        grad_weight2 = F.linear(ASA.t(), grad_output2.t()).t()
        
        grad_weight *= 100
        grad_weight1 *= 100        
        grad_weight2 *= 100
        
        grad_bias = grad_bias1 = grad_bias2 = None
        if bias is not None:
            grad_bias = grad_output.sum(0).squeeze(0)
            grad_bias1 = grad_output1.sum(0).squeeze(0)
            grad_bias2 = grad_output2.sum(0).squeeze(0)
        
        grad_input = F.linear(grad_output, weight.t())
        grad_input1 = F.linear(grad_output1, weight1.t())
        grad_input2 = F.linear(grad_output2, weight2.t())
        
        return (grad_input + grad_input1 + grad_input2).reshape(ctx.input_size), None, grad_weight, grad_weight1, grad_weight2, grad_bias, grad_bias1, grad_bias2, None, None
    
#%%
class  Three_Linear_Two_Input_ASA(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, epsilon = 0.01, get_li=False, lr_expansion=100):
        super(Three_Linear_Two_Input_ASA, self).__init__(in_features, out_features, bias=bias)
        self.in_features = in_features
        self.out_features = out_features
        self.weight1 = nn.Parameter(torch.Tensor(self.weight.size()), requires_grad=True)
        self.weight2 = nn.Parameter(torch.Tensor(self.weight.size()), requires_grad=True)
        self.get_li = get_li
        self.epsilon = epsilon
        self.lr_expansion = lr_expansion
        
        nn.init.kaiming_uniform_(self.weight1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight2, a=math.sqrt(5))
        
        if bias:
            self.bias1 = nn.Parameter(torch.Tensor(self.bias.size()), requires_grad=True)
            self.bias2 = nn.Parameter(torch.Tensor(self.bias.size()), requires_grad=True)
            
            fan_in1, _ = nn.init._calculate_fan_in_and_fan_out(self.weight1)
            bound1 = 1 / math.sqrt(fan_in1) if fan_in1 > 0 else 0
            nn.init.uniform_(self.bias1, -bound1, bound1)
            
            fan_in2, _ = nn.init._calculate_fan_in_and_fan_out(self.weight2)
            bound2 = 1 / math.sqrt(fan_in2) if fan_in2 > 0 else 0
            nn.init.uniform_(self.bias2, -bound2, bound2)
        else:
            self.bias = self.bias1 = self.bias2 = None
            
        
    def forward(self, input, input12):
        ASA = input.sign().detach().clone()
        ASA12 = input12.sign().detach().clone()
        if self.training and self.get_li:
            li = learning_indicator(input+ASA*self.epsilon, ASA*self.epsilon)
            li12 = learning_indicator(input12+ASA12*self.epsilon, ASA12*self.epsilon)
            self.li = li + 2*li12
            
        return three_linear_two_input_asa.apply(input, input12, ASA, ASA12, self.weight, self.weight1, self.weight2, self.bias, self.bias1, self.bias2, self.epsilon, self.lr_expansion)


class three_linear_two_input_asa(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, input12, ASA, ASA12, weight, weight1, weight2, bias, bias1, bias2, epsilon, lr_expansion):
        output = F.linear(input+ASA*epsilon, weight, bias)
        output1 = F.linear(input12+ASA12*epsilon, weight1, bias1)
        output2 = F.linear(input12+ASA12*epsilon, weight2, bias2)
        
        ctx.lr_expansion = lr_expansion
        ctx.epsilon = epsilon
        
        #packbits
        ASA = (ASA>0).to(ASA.device)
        ASA, restore_size = packbits_one_bits(ASA)
        ctx.restore_size = restore_size
        
        ASA12 = (ASA12>0).to(ASA12.device)
        ASA12, restore_size12 = packbits_one_bits(ASA12)
        ctx.restore_size12 = restore_size12
        
        ctx.input_size = input.size()
        ctx.input12_size = input12.size()
        ctx.save_for_backward(ASA, ASA12, weight, weight1, weight2, bias, bias1, bias2)
        return output, output1, output2

    @staticmethod
    def backward(ctx, grad_output, grad_output1, grad_output2):
        ASA, ASA12, weight, weight1, weight2, bias, bias1, bias2 = ctx.saved_tensors
        
        #unpackbit
        ASA = unpackbits_one_bits(ASA, ctx.restore_size)
        ASA = ASA.reshape(-1,ASA.size(2))
        ASA12 = unpackbits_one_bits(ASA12, ctx.restore_size12)
        ASA12 = ASA12.reshape(-1,ASA12.size(2))
        
        grad_output_size = grad_output.size()
        grad_output1_size = grad_output1.size()
        grad_output2_size = grad_output2.size()
        
        grad_output = grad_output.reshape(-1, grad_output_size[-1])        
        grad_output1 = grad_output1.reshape(-1, grad_output1_size[-1])        
        grad_output2 = grad_output2.reshape(-1, grad_output2_size[-1])        
        
        grad_weight = F.linear(ASA.t(), grad_output.t()).t()
        grad_weight1 = F.linear(ASA12.t(), grad_output1.t()).t()
        grad_weight2 = F.linear(ASA12.t(), grad_output2.t()).t()
        
        grad_weight *= ctx.lr_expansion
        grad_weight1 *= ctx.lr_expansion
        grad_weight2 *= ctx.lr_expansion
        
        grad_bias = grad_bias1 = grad_bias2 = None
        if bias is not None:
            grad_bias = grad_output.sum(0).squeeze(0)
            grad_bias1 = grad_output1.sum(0).squeeze(0)
            grad_bias2 = grad_output2.sum(0).squeeze(0)
        
        grad_input = F.linear(grad_output, weight.t())
        grad_input1 = F.linear(grad_output1, weight1.t())
        grad_input2 = F.linear(grad_output2, weight2.t())
        
        
        return grad_input.reshape(ctx.input_size), (grad_input1 + grad_input2).reshape(ctx.input12_size), None, None, grad_weight, grad_weight1, grad_weight2, grad_bias, grad_bias1, grad_bias2, None, None
