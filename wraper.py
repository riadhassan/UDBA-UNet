import torch
import torch.optim as optim
from networks import UNet,NestedUNet,AttU_Net
from R2Unet import R2U_Net
from auxillary_net import auxnet
from loss import Loss, Distill 
import numpy as np
import torchvision
from torch.autograd import Variable
from utils import*

class ModelWraper:
    def __init__(self,conf):
        self.conf=conf
        self.device = torch.device(conf.device)
        if self.conf.network_type == "Unet":
            self.seg_model = UNet(in_channels=conf.input_channels,out_channels= conf.output_channels)
        elif self.conf.network_type == "Unet++":
            self.seg_model = NestedUNet(input_channels=conf.input_channels,num_classes=conf.output_channels)
        elif self.conf.network_type == "Att_Unet":
            self.seg_model = AttU_Net(input_channels=conf.input_channels,num_classes=conf.output_channels)
        elif self.conf.network_type == "R2Unet":
            self.seg_model = R2U_Net(img_ch=conf.input_channels,output_ch=conf.output_channels,t=2)

        self.seg_model.to(self.device)
        self.weights = None
        self.optimizer1 = optim.Adam(self.seg_model.parameters(), lr=conf.lr)
        self.loss_function = Loss(conf)
       
        if conf.isprob =='yes':
            self.distill_loss = Distill(self.conf.device)
            self.distill_loss= self.distill_loss.to(self.conf.device)
        
    def set_mood(self,Train=True):
        if Train:
            self.seg_model.train()
        else:
            self.seg_model.eval()
    
    def get_uncertainty_weights(self):
        subs = torch.zeros((self.conf.imsize, self.conf.imsize))
        weights = torch.zeros((self.conf.imsize, self.conf.imsize))
        pred = torch.squeeze(torch.argmax(self.conf.pred,dim=1))
        pred_a = torch.squeeze(torch.argmax(self.conf.pred_a,dim=1))
        for i in range(1,self.conf.output_channels):
            mask_pred = torch.zeros((self.conf.imsize, self.conf.imsize),dtype=torch.long)
            mask_pred_a = torch.zeros((self.conf.imsize, self.conf.imsize),dtype=torch.long)
            mask_pred[pred==i] = 1
            mask_pred_a[pred_a==i] =1
            mask_union = torch.bitwise_or(mask_pred,mask_pred_a)
            mask_inter = torch.bitwise_and(mask_pred,mask_pred_a)
            #mask_sub = mask_union + mask_inter
            #mask_sub = mask_sub/mask_sub.max()
            subs[mask_inter==1] = i
            subs[mask_union==1] = i

        ### Probability over channels
        pw = torch.softmax(self.conf.pred,dim=1).detach().cpu()
        pw_a = torch.softmax(self.conf.pred_a,dim=1).detach().cpu()
     
        #import pdb
        #pdb.set_trace()
        pw = torch.max(pw,dim=1).values
        pw_a = torch.max(pw_a,dim=1).values
        
        if self.conf.iter % 100==0:
            #visualize_mask(pw[None,:,:,:],'pw',self.conf.debug_path,self.conf.iter)
            #visualize_mask(pw_a[None,:,:,:],'pw_a',self.conf.debug_path,self.conf.iter)
            visualize(mask_union[None,None,:,:],'union',self.conf.debug_path,self.conf.iter)
            visualize(mask_inter[None,None,:,:],'inter',self.conf.debug_path,self.conf.iter)

        weights = (pw + pw_a)
        weights = weights/weights.max()
        #all_max = torch.max(torch.concat([pw,pw_a],dim=0),dim=0).values
        #idx = torch.where(subs>0)
        #weights[idx[0],idx[1]] = all_max[idx[0],idx[1]] 
        return weights[None,:,:,:],subs[None,None,:,:]

    def update_models(self,input,iter):
        self.conf.iter = iter
        self.conf = self.loss_function.setup_input(input)
        self.optimizer1.zero_grad()
        
        if self.weights is None:
            self.conf.pred,self.conf.pred_a = self.seg_model(self.conf.input)
        else:
            self.conf.pred,self.conf.pred_a = self.seg_model(self.conf.input,self.weights)
        
        main_loss = self.loss_function(self.conf,isAux=False)
        
        if self.conf.isprob == 'yes': #and self.conf.curr_epoch>0:
            aux_loss = self.loss_function(self.conf,isAux=True)
            aux_loss2 = Variable(self.distill_loss(self.conf.pred_a,self.conf.pred.detach()),requires_grad=True)
            self.weights,subs = self.get_uncertainty_weights()
            loss = main_loss +aux_loss +aux_loss2
        else:
            loss = main_loss
            aux_loss = torch.tensor(0.0)

        loss.backward(retain_graph=True)
        self.optimizer1.step()
        if self.conf.iter % 100==0:
            visualize(self.conf.pred,'seg',self.conf.debug_path,self.conf.iter)
            visualize(self.conf.pred_a,'aux',self.conf.debug_path,self.conf.iter)
            if self.conf.isprob =='yes':
                visualize_mask(self.weights,'weight',self.conf.debug_path,self.conf.iter)
                visualize(subs,'subs',self.conf.debug_path,self.conf.iter)
        
        return main_loss,aux_loss
    
    
            



        
        

        
        

    

        
        
