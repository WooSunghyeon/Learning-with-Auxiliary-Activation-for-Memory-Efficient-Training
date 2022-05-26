import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
import torchsummary
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from Auxiliary_Activation_Learning import Linear_ABA, Linear_ASA
from custom_layer import ReLU


class MLPMixer(nn.Module):
    def __init__(self,in_channels=3,img_size=32, patch_size=4, hidden_size=512, hidden_s=256, hidden_c=2048, num_layers=8, num_classes=10, drop_p=0., off_act=False, is_cls_token=False, learning_rule = 'bp', get_li = False):
        super(MLPMixer, self).__init__()
        num_patches = img_size // patch_size * img_size // patch_size
        # (b, c, h, w) -> (b, d, h//p, w//p) -> (b, h//p*w//p, d)
        self.is_cls_token = is_cls_token

        self.patch_emb = nn.Sequential(
            nn.Conv2d(in_channels, hidden_size ,kernel_size=patch_size, stride=patch_size),
            Rearrange('b d h w -> b (h w) d')
        )

        if self.is_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))
            num_patches += 1


        self.mixer_layers = nn.Sequential(
            *[
                MixerLayer(num_patches, hidden_size, hidden_s, hidden_c, drop_p, off_act, learning_rule, get_li) 
            for _ in range(num_layers)
            ]
        )
        self.ln = nn.LayerNorm(hidden_size)

        self.clf = nn.Linear(hidden_size, num_classes)
        self.get_li = get_li

    def forward(self, x):
        out = self.patch_emb(x)
        if self.is_cls_token:
            out = torch.cat([self.cls_token.repeat(out.size(0),1,1), out], dim=1)
        out = self.mixer_layers(out)
        out = self.ln(out)
        out = out[:, 0] if self.is_cls_token else out.mean(dim=1)
        out = self.clf(out)
        
        li = []
        
        if self.training and self.get_li:
            for m in self.modules():
              if isinstance(m, Linear_ABA) or isinstance(m, Linear_ASA):  
                  if m.li != None:
                      li += m.li
        return out, li


class MixerLayer(nn.Module):
    def __init__(self, num_patches, hidden_size, hidden_s, hidden_c, drop_p, off_act, learning_rule, get_li):
        super(MixerLayer, self).__init__()
        self.mlp1 = MLP1(num_patches, hidden_s, hidden_size, drop_p, off_act, learning_rule, get_li)
        self.mlp2 = MLP2(hidden_size, hidden_c, drop_p, off_act, learning_rule, get_li)
    def forward(self, x):
        out = self.mlp1(x)
        out = self.mlp2(out)
        return out

class MLP1(nn.Module):
    def __init__(self, num_patches, hidden_s, hidden_size, drop_p, off_act, learning_rule, get_li):
        super(MLP1, self).__init__()
        self.ln = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(num_patches, hidden_s)
        if learning_rule == 'aba4':
            self.fc1 = Linear_ABA(num_patches, hidden_s, get_li=get_li)
        elif learning_rule == 'asa4':
            self.fc1 = Linear_ASA(num_patches, hidden_s, get_li=get_li)
        
        self.do1 = nn.Dropout(p=drop_p)
        self.fc2 = nn.Linear(hidden_s, num_patches)
        if learning_rule == 'aba3' or learning_rule == 'aba4':
            self.fc2 = Linear_ABA(hidden_s, num_patches, get_li=get_li)
        if learning_rule == 'asa3' or learning_rule == 'asa4':
            self.fc2 = Linear_ASA(hidden_s, num_patches, get_li=get_li, relu=True)
        
        self.do2 = nn.Dropout(p=drop_p)
        self.act = ReLU()
    def forward(self, x):
        out = self.do1(self.act(self.fc1(self.ln(x).permute(0,2,1))))
        out = self.do2(self.fc2(out))
        return out.permute(0,2,1)+x
 
class MLP2(nn.Module):
    def __init__(self, hidden_size, hidden_c, drop_p, off_act, learning_rule, get_li):
        super(MLP2, self).__init__()
        self.ln = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_c)
        if learning_rule == 'aba2' or learning_rule == 'aba3' or learning_rule == 'aba4':
            self.fc1 = Linear_ABA(hidden_size, hidden_c, get_li=get_li)
        elif learning_rule == 'asa2' or learning_rule == 'asa3' or learning_rule == 'asa4':
            self.fc1 = Linear_ASA(hidden_size, hidden_c, get_li=get_li)
        
        self.do1 = nn.Dropout(p=drop_p)
        self.fc2 = nn.Linear(hidden_c, hidden_size)
        if learning_rule == 'aba1' or learning_rule == 'aba2' or learning_rule == 'aba3' or learning_rule == 'aba4':
            self.fc2 = Linear_ABA(hidden_c, hidden_size, get_li=get_li)
        elif learning_rule == 'asa1' or learning_rule == 'asa2' or learning_rule == 'asa3' or learning_rule == 'asa4':
            self.fc2 = Linear_ASA(hidden_c, hidden_size, get_li=get_li, relu=True)
            
        self.do2 = nn.Dropout(p=drop_p)
        self.act = ReLU()
    def forward(self, x):
        out = self.do1(self.act(self.fc1(self.ln(x))))
        out = self.do2(self.fc2(out))
        return out+x

if __name__ == '__main__':
    net = MLPMixer(
        in_channels=3,
        img_size=32, 
        patch_size=4, 
        hidden_size=128, 
        hidden_s=512, 
        hidden_c=64, 
        num_layers=8, 
        num_classes=10, 
        drop_p=0.,
        off_act=False,
        is_cls_token=True
        )
    torchsummary.summary(net, (3,32,32))
