import os
import torch
import torch.nn as nn
import torch.nn.functional as F
#import torch.backends.cudnn as cudnn
from collections import OrderedDict
#import _init_paths
from fvcore.common.file_io import PathManager
from segmentation.config import config, update_config
from segmentation.utils.logger import setup_logger
from segmentation.model import build_segmentation_model_from_cfg
from segmentation.utils.utils import get_module
import pickle
from resnet_settings import parse_opts 
from resnet_model import generate_model
class auxnet(nn.Module):
    def __init__(self,conf):
        super(auxnet, self).__init__()
        self.conf=conf
        sets = parse_opts()
        sets.target_type = "normal"
        sets.phase = 'test'
        checkpoint = torch.load(sets.pretrain_path,map_location=torch.device('mps'))
        self.model = generate_model(sets)
        self.model.load_state_dict(checkpoint['state_dict'],strict=False)
       
        #self.input = nn.Conv3d(3,256,kernel_size=5,stride=1,padding=2)
        #self.output = nn.ConvTranspose3d(conf.output_channels,conf.output_channels,
        #kernel_size=(1,2,2),stride=(1,4,4),dilation=(1,4,4))
        #nn.functional.interpolate(size=(1,conf.output_channels,2,256,256),mode='bilinear')
    
    def forward(self,input):
         #x = self.input(input)
         x = torch.stack([input for i in range(5)],dim=1)
         x = torch.reshape(x,(1,1,5,256,256))
         x = self.model(x)
         y =  nn.functional.interpolate(input=x[0],size=(256,256),mode='nearest-exact')
         y = y[:,0,:,:]
         return y[None,:,:,:]
'''         
class auxnet(nn.Module):
    def __init__(self,conf):
        super(auxnet, self).__init__()
        
        if conf.aux_file == "res_50":
            aux_file_loc = os.path.join(os.getcwd(),'configs','panoptic_deeplab_R50_os32_cityscapes.yaml')
            update_config(config,aux_file_loc)
        elif conf.aux_file == "res_101":
            aux_file_loc = os.path.join(os.getcwd(),'configs','panoptic_deeplab_R101_os32_cityscapes.yaml')
            update_config(config,aux_file_loc)

        self.model = build_segmentation_model_from_cfg(config)
        self.path = os.path.join('/Users/Segthor/panoptic_weights','model_final_coco.pkl')

        with open(self.path, 'rb') as f:
            obj = f.read()
        weights = {key:arr  for key, arr in pickle.loads(obj).items()}
        self.model= get_module(self.model, False)
        self.model.load_state_dict(weights, strict=False)
        #model.load_state_dict(torch.load(path))
        self.model = self.model.decoder.semantic_head
        for module in self.model.modules():
            if isinstance(module, torch.nn.modules.container.Sequential):
                self.model = module
                break
        self.input = nn.Conv2d(1,256,kernel_size=5,stride=1,padding=2)
        self.output = nn.Conv2d(19,conf.output_channels,kernel_size=1,stride=1)
    
    def forward(self,input):
         x = self.input(input)
         x = self.model(x)
         x = self.output(x)
         return x
'''



