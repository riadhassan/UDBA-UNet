from collections import OrderedDict
import torch
import torch.nn as nn
from BAM import BAM
import math
import numpy as np
from torch.distributions.uniform import Uniform
from networks_utils import *
import copy

class Encoder(nn.Module):
    def __init__(self, in_channels,features=[64,128,256,512]):
        super(Encoder,self).__init__()
        self.layers = []
        for i,f in enumerate(features):
            if i ==0:
                self.layers.append(enblock(in_channels,f,name=f"layer-{i}",stride=1))
            else:
                self.layers.append(enblock(in_channels=features[i-1],features=features[i],name=f"layer-{i}",stride=1))
        
        self.layers.append(enblock(in_channels=features[-1],features=features[-1]*2, name="bottleneck",stride=1))
        #self.res_block_1 = multiResBlock(in_channels=features[-1],features=features[-1]*2,kernel=1)
        self.layers = nn.Sequential(*self.layers)
    
    def forward(self,x):
        skips =[]
        for layer in self.layers:
            x = layer(x)
            skips.append(x)
        return x,skips 

class Decoder(nn.Module):
    def __init__(self,out_channels,decoder_type,features =[512,256,128,64]):
        super(Decoder,self).__init__()
        self.layers = []
        self.features = features.copy()
        self.type = decoder_type
        if self.type == "Drop":
            self.feature_drop = FeatureDrop()
        if self.type == "Noise":
            self.feature_noise = FeatureNoise()
        
        self.layers.append(decblock(self.features[0]*2,self.features[0],name=f"bottleneck",stride=1))
        
        for i,f in enumerate(features):
            self.layers.append(decblock(self.features[i]*2,self.features[i]//2,name=f"layer-{i}",stride=1))
        
        self.layers.append(nn.Conv2d(
            in_channels=self.features[-1]//2, out_channels=out_channels, kernel_size=1))
        self.layers = nn.Sequential(*self.layers)
    
    def dec_forward(self,x,skips):
        skips.reverse()
        for i,layer in enumerate(self.layers):
                x = layer(x)
                if i<len(skips):
                    x = torch.cat([x,skips[i]],dim=1)
        return x
        
    def forward(self,x,skips):
        if self.type=="main":
            return self.dec_forward(x,skips[:4])
        if self.type == "Drop":
            x = self.feature_drop(x)
            
        if self.type == "Noise":
            x = self.feature_noise(x)
        
        return self.dec_forward(x,skips[:4])

class UNet(nn.Module):

    def __init__(self, in_channels=1, out_channels=5,features=[64,128,256,512]):
        super(UNet, self).__init__()
        self.encoder = Encoder(in_channels,features=features)
        self.main_decoder = Decoder(out_channels,"main")
        self.drop_decoder = Decoder(out_channels,"Noise")
        #### Uncertainty modules ###
        self.attenion = U_Attention(features[0] * 16) 
        
    def forward(self, x,weights=None):
        #enc0 = self.encoder0(x)
        self.enc,self.skips = self.encoder(x)
        
        if isinstance(weights,torch.Tensor):
            self.enc = self.attenion(weights,self.enc)
    
        seg = self.main_decoder(self.enc,self.skips)
        aux_seg = self.drop_decoder(self.enc,self.skips)
        
        return seg, aux_seg
    
class FeatureDrop(nn.Module):
    def __init__(self):
        super(FeatureDrop, self).__init__()
        #self.upsample = upsample(conv_in_ch, num_classes, upscale=upscale)

    def feature_dropout(self, x):
        attention = torch.mean(x, dim=1, keepdim=True)
        max_val, _ = torch.max(attention.view(x.size(0), -1), dim=1, keepdim=True)
        threshold = max_val * np.random.uniform(0.7, 0.9)
        threshold = threshold.view(x.size(0), 1, 1, 1).expand_as(attention)
        drop_mask = (attention < threshold).float()
        return x.mul(drop_mask)

    def forward(self, x):
        x = self.feature_dropout(x)
        #x = self.upsample(x)
        return x


class FeatureNoise(nn.Module):
    def __init__(self, uniform_range=0.3):
        super(FeatureNoise, self).__init__()
        #self.upsample = upsample(conv_in_ch, num_classes, upscale=upscale)
        self.uni_dist = Uniform(-uniform_range, uniform_range)

    def feature_based_noise(self, x):
        noise_vector = self.uni_dist.sample(x.shape[1:]).to(x.device).unsqueeze(0)
        x_noise = x.mul(noise_vector) + x
        return x_noise

    def forward(self, x):
        x = self.feature_based_noise(x)
        #x = self.upsample(x)
        return x

class U_Attention(nn.Module):
    def __init__(self,bottleneck_dim,reduction_ratio=16,dilation_num=2,dilation_val=4):
        super(U_Attention,self).__init__()
        self.attention = nn.Sequential()
        self.features = bottleneck_dim
        self.shape = (1,bottleneck_dim,256,256)
        self.attention.add_module("attSoftmax",nn.Softmax2d())
        self.attention.add_module("attDown",nn.Conv2d(self.features,self.features,kernel_size=2,stride=32))
        '''
        self.attention.add_module("attConv1",nn.Conv2d(in_channels=1,out_channels=self.features,kernel_size=1))
        self.attention.add_module( 'attBN',	nn.BatchNorm2d(self.features//reduction_ratio) )
        self.attention.add_module( 'attRL',nn.ReLU() )
        self.attention.add_module("attConv3_di",nn.Conv2d(in_channels=self.features,
        out_channels=self.features, kernel_size=3,dilation=dilation_val))
        '''
        self.attention.add_module("Spatial_BAM",BAM(self.features))
    
    def forward(self,weights,bottleneck):
        weight_list = []
        for i in range(self.features):
            weight_list.append(weights)
        weights = torch.stack(weight_list,dim=1)
        weights = torch.reshape(weights,self.shape)
        weights = weights.to("mps")
        attention = self.attention(weights)
        new_bottleneck = attention * bottleneck
        return new_bottleneck


