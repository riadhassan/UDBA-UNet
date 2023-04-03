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
        
        self.layers.append(enblock(in_channels=features[-1],features=features[-1]*2, name="bottleneck",stride=1,
        pool_k=2,pool_stride=2))
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
        
        self.layers.append(decblock(self.features[0]*2,self.features[0],name=f"bottleneck",stride=1,
        pool_k=2,pool_stride=2))
        
        for i,f in enumerate(features):
            self.layers.append(decblock(self.features[i]*2,self.features[i]//2,name=f"layer-{i}",stride=1))
        
        self.layers.append(nn.Conv2d(
            in_channels=self.features[-1]//2, out_channels=out_channels, kernel_size=1))
        self.layers = nn.Sequential(*self.layers)
    
    def dec_forward(self,x,skips):
        skips.reverse()
        #import pdb
        #pdb.set_trace()
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
        features.reverse()
        self.main_decoder = Decoder(out_channels,"main",features=features)
        self.drop_decoder = Decoder(out_channels,"Noise",features=features)
            #### Uncertainty modules ###
        #features.reverse()
        #self.attenion = U_Attention(bottleneck_dim=features[0] * 16)
           
    def forward(self, x,weights=None):
        #enc0 = self.encoder0(x)
        self.enc,self.skips = self.encoder(x)
        #aux_seg = self.drop_decoder(self.enc,self.skips)
        if isinstance(weights,torch.Tensor):
            features = [x.shape[1] for x in self.skips]
            dims = [x.shape[2] for x in self.skips]
            self.attenion = U_AttentionDense(dims,features)
            self.skips = self.attenion(weights,self.skips)
        aux_seg = self.drop_decoder(self.enc,self.skips)
        seg = self.main_decoder(self.enc,self.skips)
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

class U_AttentionDense(nn.Module):
    def __init__(self,dims,Features):
        super(U_AttentionDense,self).__init__()
        self.features = Features
        self.shape = []
        for c,w in zip(Features,dims):
            self.shape.append((1,c,w,w))
        self.downsamples = OrderedDict()
        for i,x in enumerate(dims): 
            self.downsamples['down'+str(i)] = nn.Upsample(size=(x,x),mode='bilinear')
    
    def create_weight_list(self,weights):
        self.weight_list = []
        for i,feat in enumerate(self.features):
            weights_new = self.downsamples['down'+str(i)](weights)
            weight_stake = []
            for j in range(feat):
                weight_stake.append(weights_new)
            weights_new = torch.stack(weight_stake,dim=1)
            weights_new = torch.reshape(weights_new,self.shape[i])
            weights_new = weights_new.to("cuda")
            self.weight_list.append(weights_new)


    def forward(self,weights,skips):
        self.create_weight_list(weights)
        self.skip_list = []
        
        for sk,we in zip(skips,self.weight_list):
            self.skip_list.append(sk*we)
        return self.skip_list

class U_Attention(nn.Module):
    def __init__(self,bottleneck_dim,reduction_ratio=16,dilation_num=2,dilation_val=4):
        super(U_Attention,self).__init__()
        self.attention = nn.Sequential()
        self.features = bottleneck_dim
        self.shape = (1,bottleneck_dim,8,8)

        self.downsample =nn.Upsample(size=(8,8),mode='bilinear')
        self.attention.add_module("attSoftmax",nn.Softmax2d())
        #self.attention.add_module("attDown",nn.Conv2d(self.features,self.features,kernel_size=2,stride=16))
        self.attention.add_module("Spatial_BAM",BAM(self.features))
    
    def forward(self,weights,bottleneck):
        weights = self.downsample(weights)
        weight_list = []
        for i in range(self.features):
            weight_list.append(weights)
        weights = torch.stack(weight_list,dim=1)
        weights = torch.reshape(weights,self.shape)
        
        weights = weights.to("cuda")
        #attention = self.attention(weights)
        new_bottleneck = weights * bottleneck
        return new_bottleneck

## Unet++

class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out

class NestedUNet(nn.Module):
    def __init__(self, num_classes, input_channels=1, deep_supervision=False, **kwargs):
        super().__init__()

        nb_filter = [32,64, 128, 256, 512]

        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        self.attenion = U_Attention(bottleneck_dim=nb_filter[0] * 8)   #Riad use 8 but in unet it is 16

        self.feature_drop = FeatureDrop()

        self.feature_noise = FeatureNoise()

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input, weights=None):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))
        ''''
        x3_0_aux_seg = self.feature_drop(x3_0_enc)
        x4_0_aux_seg = self.conv4_0(self.pool(x3_0_aux_seg))
        x3_1_aux_seg = self.conv3_1(torch.cat([x3_0_aux_seg, self.up(x4_0_aux_seg)], 1))
        x2_2_aux_seg = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1_aux_seg)], 1))
        x1_3_aux_seg = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2_aux_seg)], 1))
        x0_4_aux_seg = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3_aux_seg)], 1))

        if isinstance(weights,torch.Tensor):
            x3_0_enc = self.attenion(weights,x3_0_enc)

        #x3_0_seg = self.feature_drop(x3_0)
        x4_0_seg = self.conv4_0(self.pool(x3_0_enc))
        x3_1_seg = self.conv3_1(torch.cat([x3_0_enc, self.up(x4_0_seg)], 1))
        x2_2_seg = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1_seg)], 1))
        x1_3_seg = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2_seg)], 1))
        x0_4_seg = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3_seg)], 1))
        

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output_seg = self.final(x0_4_seg)
            output_aux_seg = self.final(x0_4_aux_seg)
            return output_seg, output_aux_seg
        '''
        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            return output,output

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x
class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi

class AttU_Net(nn.Module):
    def __init__(self,input_channels=1,num_classes=1):
        super(AttU_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=input_channels,ch_out=64)
        self.Conv2 = conv_block(ch_in=64,ch_out=128)
        self.Conv3 = conv_block(ch_in=128,ch_out=256)
        self.Conv4 = conv_block(ch_in=256,ch_out=512)
        self.Conv5 = conv_block(ch_in=512,ch_out=1024)

        self.Up5 = up_conv(ch_in=1024,ch_out=512)
        self.Att5 = Attention_block(F_g=512,F_l=512,F_int=256)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Att4 = Attention_block(F_g=256,F_l=256,F_int=128)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Att3 = Attention_block(F_g=128,F_l=128,F_int=64)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Att2 = Attention_block(F_g=64,F_l=64,F_int=32)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64,num_classes,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5,x=x4)
        d5 = torch.cat((x4,d5),dim=1)        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1,d1
