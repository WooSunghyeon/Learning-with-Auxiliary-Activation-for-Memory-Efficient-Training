import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from Auxiliary_Activation_Learning import Linear_ABA, Linear_ASA
from custom_layer import ReLU, Three_Linear_One_Input, Three_Linear_One_Input_ABA, Three_Linear_One_Input_ASA
from torch.utils.checkpoint import checkpoint


class TransformerEncoder(nn.Module):
    def __init__(self, feats:int, mlp_hidden:int, head:int=8, dropout:float=0., learning_rule='bp', get_li=False, gcp=False):
        super(TransformerEncoder, self).__init__()
        self.la1 = nn.LayerNorm(feats)
        self.msa = MultiHeadSelfAttention(feats, head=head, dropout=dropout, learning_rule=learning_rule, get_li=get_li, gcp=gcp)
        self.la2 = nn.LayerNorm(feats)
        self.mlp = nn.Sequential(
            nn.Linear(feats, mlp_hidden),
            ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, feats),
            ReLU(),
            nn.Dropout(dropout),
        )
        if learning_rule == 'aba1':
            self.mlp = nn.Sequential(
                nn.Linear(feats, mlp_hidden),
                ReLU(),
                nn.Dropout(dropout),
                Linear_ABA(mlp_hidden, feats, get_li=get_li),
                ReLU(),
                nn.Dropout(dropout),
            )
        
        elif learning_rule =='aba2' or learning_rule =='aba3' or learning_rule =='aba4':
            self.mlp = nn.Sequential(
                Linear_ABA(feats, mlp_hidden, get_li=get_li),
                ReLU(),
                nn.Dropout(dropout),
                Linear_ABA(mlp_hidden, feats, get_li=get_li),
                ReLU(),
                nn.Dropout(dropout),
            )
            
        elif learning_rule == 'asa1':
            self.mlp = nn.Sequential(
                nn.Linear(feats, mlp_hidden),
                ReLU(),
                nn.Dropout(dropout),
                Linear_ASA(mlp_hidden, feats, get_li=get_li),
                ReLU(),
                nn.Dropout(dropout),
            )
        
        elif learning_rule =='asa2' or learning_rule =='asa3' or learning_rule =='asa4':
            self.mlp = nn.Sequential(
                Linear_ASA(feats, mlp_hidden, get_li=get_li),
                ReLU(),
                nn.Dropout(dropout),
                Linear_ASA(mlp_hidden, feats, get_li=get_li, relu=True),
                ReLU(),
                nn.Dropout(dropout),
            )
            
    def forward(self, x):
        out = self.msa(self.la1(x)) + x
        out = self.mlp(self.la2(out)) + out
        return out


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, feats:int, head:int=8, dropout:float=0., learning_rule=False, get_li=False, gcp = False):
        super(MultiHeadSelfAttention, self).__init__()
        self.head = head
        self.feats = feats
        self.sqrt_d = self.feats**0.5
        self.gcp = gcp
        
        self.qkv = Three_Linear_One_Input(feats, feats)
        self.self_attention = SelfAttention(self.sqrt_d)
        self.o = nn.Linear(feats, feats)
        if learning_rule == 'aba3':
            self.qkv = Three_Linear_One_Input(feats, feats)
            self.o = Linear_ABA(feats, feats, get_li=get_li)
        elif learning_rule == 'aba4':
            self.qkv = Three_Linear_One_Input_ABA(feats, feats, get_li=get_li)
            self.o = Linear_ABA(feats, feats, get_li=get_li)
        elif learning_rule == 'asa3':
            self.qkv = Three_Linear_One_Input(feats, feats)
            self.o = Linear_ASA(feats, feats, get_li=get_li)
        elif learning_rule == 'asa4':
            self.qkv = Three_Linear_One_Input_ASA(feats, feats, get_li=get_li)
            self.o = Linear_ASA(feats, feats, get_li=get_li)
        
            
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, n, f = x.size()
        q, k, v = self.qkv(x)
        q = q.view(b, n, self.head, self.feats//self.head).transpose(1,2)
        k = k.view(b, n, self.head, self.feats//self.head).transpose(1,2)
        v = v.view(b, n, self.head, self.feats//self.head).transpose(1,2)
        
        #score = F.softmax(torch.einsum("bhif, bhjf->bhij", q, k)/self.sqrt_d, dim=-1) #(b,h,n,n)
        if not(self.gcp):
            attn = self.self_attention(q, k, v)
        else:
            attn = checkpoint(self.self_attention, q, k, v)
        o = self.dropout(self.o(attn.flatten(2)))
        return o

class SelfAttention(nn.Module):
    def __init__(self, sqrt_d):
        super(SelfAttention, self).__init__()
        self.sqrt_d = sqrt_d
        
    def forward(self, q, k, v):
        score = F.softmax(torch.einsum("bhif, bhjf->bhij", q, k)/self.sqrt_d, dim=-1) #(b,h,n,n)
        attn = torch.einsum("bhij, bhjf->bihf", score, v) #(b,n,h,f//h)
        return attn
        



class MultiHeadDepthwiseSelfAttention(nn.Module):
    def __init__(self, feats:int, head:int=8, dropout:float=0):
        super(MultiHeadDepthwiseSelfAttention, self).__init__()
        ...

    def forward(self, x):
        
        ...

if __name__=="__main__":
    b,n,f = 4, 16, 128
    x = torch.randn(b,n,f)
    # net = MultiHeadSelfAttention(f)
    net = TransformerEncoder(f)
    #torchsummary.summary(net, (n,f))
    # out = net(x)
    # print(out.shape)



