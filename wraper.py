import torch
import torch.optim as optim
from networks import *
from auxillary_net import auxnet
from loss import Loss, Distill 
import numpy as np
import torchvision
from torch.autograd import Variable
from utils import*

class ModelWraper:
    def __init__(self,conf):
        self.device = torch.device(conf.device)
        self.seg_model = UNet(in_channels=conf.input_channels,out_channels= conf.output_channels)
        self.seg_model.to(self.device)
        self.conf=conf
        self.weights = None
        self.optimizer1 = optim.Adam(self.seg_model.parameters(), lr=conf.lr)
        self.loss_function = Loss(conf)
        
        if conf.isuncertainty:
            #self.a_model = Decoder(conf.output_channels,"Noise")
            #self.a_model = self.a_model.to(device=self.device)
            #y = self.a_model(torch.randn((1,1,2,256,256)))
            #self.optimizer1 = optim.SGD(self.seg_model.parameters(), lr=conf.lr,momentum=0.09)
            #self.optimizer2 = optim.SGD(self.a_model.parameters(), lr=conf.lr,momentum=0.09)
            self.distill_loss = Distill(self.conf.device)
            self.distill_loss= self.distill_loss.to(self.conf.device)
        
    def set_mood(self,Train=True):
        if Train:
            self.seg_model.train()
            #if self.conf.isuncertainty:
                #self.a_model.train()
        else:
            self.seg_model.eval()
            #if self.conf.isuncertainty:
                #self.a_model.eval()
    
    def get_uncertainty_weights(self):
        subs = torch.zeros((self.conf.imsize, self.conf.imsize))
        pred = torch.squeeze(torch.argmax(self.conf.pred,dim=1))
        pred_a = torch.squeeze(torch.argmax(self.conf.pred_a,dim=1))
        for i in range(1,self.conf.output_channels):
            mask_pred = torch.zeros((self.conf.imsize, self.conf.imsize),dtype=torch.long)
            mask_pred_a = torch.zeros((self.conf.imsize, self.conf.imsize),dtype=torch.long)
            mask_pred[pred==i] = 1
            mask_pred_a[pred_a==i] =1
            mask_union = torch.bitwise_or(mask_pred,mask_pred_a)
            mask_inter = torch.bitwise_and(mask_pred,mask_pred_a)
            mask_sub = mask_union - mask_inter
            subs[mask_sub==1] = i
        #subs = 1.0 - subs/subs.max()
        weights = torch.softmax(subs,dim=-1)
        return weights[None,None,:,:],subs[None,None,:,:]

    def update_models(self,input,iter):
        self.conf.iter = iter
        self.conf = self.loss_function.setup_input(input)
        self.optimizer1.zero_grad()
        if self.weights is None:
            self.conf.pred,self.conf.pred_a = self.seg_model(self.conf.input)
        else:
            self.conf.pred,self.conf.pred_a = self.seg_model(self.conf.input,self.weights)
        
        main_loss = self.loss_function(self.conf,isAux=False)
        
        if self.conf.curr_epoch>2:
            aux_loss = Variable(self.distill_loss(self.conf.pred_a,self.conf.pred.detach()),requires_grad=True)
            if self.conf.isuncertainty:
                self.weights,subs = self.get_uncertainty_weights()
                visualize(subs,'subs',self.conf.debug_path,self.conf.iter)      
        else:
            aux_loss = self.loss_function(self.conf,isAux=True)
        
        loss = main_loss + aux_loss  
        loss.backward(retain_graph=True)
        self.optimizer1.step()
        if self.conf.iter % 100==0:
            visualize(self.conf.pred,'seg',self.conf.debug_path,self.conf.iter)
            visualize(self.conf.pred_a,'aux',self.conf.debug_path,self.conf.iter)
        return main_loss,aux_loss
    
    
            



        
        

        
        

    

        
        
