import torch
import torch.optim as optim
from networks import UNet
from auxillary_net import auxnet
from loss import Loss, Distill 
import numpy as np
import torchvision

class ModelWraper:
    def __init__(self,conf):
        self.device = torch.device(conf.device)
        self.seg_model = UNet(in_channels= conf.input_channels,out_channels= conf.output_channels)
        self.seg_model.to(self.device)
        self.conf=conf
        if conf.isuncertainty:
            self.a_model = auxnet(conf=conf)
            self.a_model = self.a_model.to(device=self.device)
            #y = a_model(torch.randn((1,1,256,256)))
            self.optimizer1 = optim.Adam(self.seg_model.parameters(), lr=conf.lr)
            self.optimizer2 = optim.Adam(self.a_model.parameters(), lr=conf.lr)
            self.distill_loss = Distill(self.conf.device)
            self.distill_loss= self.distill_loss.to(self.conf.device)
        else:
            self.optimizer1 = optim.Adam(self.seg_model.parameters(), lr=conf.lr)
        
        self.loss_function = Loss(conf)

    def set_mood(self,Train=True):
        if Train:
            self.seg_model.train()
            if self.conf.isuncertainty:
                self.a_model.train()
        else:
            self.seg_model.eval()
            if self.conf.isuncertainty:
                self.a_model.eval()
    
    def get_uncertainty_weights(self):
        union = torch.zeros((1, self.conf.output_channels, self.conf.imsize, self.conf.imsize))
        ints = torch.zeros((1, self.conf.output_channels, self.conf.imsize, self.conf.imsize))
        
        for i in range(self.conf.output_channels):
            union[:,i,:,:] = self.conf.pred[:,i,:,:].logical_and(self.conf.a_pred[:,i,:,:])
            ints[:,i,:,:] = self.conf.pred[:,i,:,:].logical_or(self.conf.a_pred[:,i,:,:])
        subs = union - ints
        return subs

    def update_models(self, input,iter):
        self.conf.iter = iter
        self.conf = self.loss_function.setup_input(input)
        self.optimizer1.zero_grad()
        self.conf.pred = self.seg_model(self.conf.input)
        pred = self.conf.pred
        main_loss = self.loss_function(self.conf)  
        main_loss.backward(retain_graph=True)
        self.optimizer1.step()
        self.visualize(self.conf.pred,'seg')
        
        if self.conf.isuncertainty:
            self.optimizer2.zero_grad()
            self.conf.pred_a = self.a_model(self.conf.input)
            dist_loss = self.distill_loss(self.conf.pred_a,pred.detach(),self.conf.gt)
            dist_loss.backward()
            self.optimizer2.step()
            self.visualize(self.conf.pred_a,'aux')
            #weights = self.get_uncertainty_weights()
        return main_loss,dist_loss
    
    def visualize(self, out, model_type):
        out = torch.argmax(out,dim=1).cpu()
        class_to_color = [torch.tensor([0, 0, 0]),torch.tensor([10, 133, 1]), torch.tensor([14, 1, 133]),  torch.tensor([33, 255, 1]), torch.tensor([243, 5, 247])]
        output = torch.zeros(1, 3, out.size(-2), out.size(-1), dtype=torch.float)
        for class_idx, color in enumerate(class_to_color):
            mask = out == class_idx
            mask = mask.unsqueeze(1) # should have shape 1, 1, 100, 100
            curr_color = color.reshape(1, 3, 1, 1)
            segment = mask*curr_color # should have shape 1, 3, 100, 100
            output += segment
        if self.conf.iter % 100==0:
            torchvision.utils.save_image(output, f"{self.conf.debug_path}/pred_{model_type}_{self.conf.iter}.png")
            



        
        

        
        

    

        
        
