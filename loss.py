
import torch
import torch.nn as nn
import torch.nn.functional as F
#from torchmetrics import JaccardIndex
#from .one_hot import one_hot
class Loss():
    def __init__(self,conf):
        self.conf=conf
        self.loss_pool = {"Dice":DiceLoss(num_organ=conf.output_channels-1,
                               imsize=self.conf.imsize,device=self.conf.device),

                           "SoftDice":CustomSoftDiceLoss(n_classes=self.conf.output_channels,
                           class_ids=[range(0,self.conf.output_channels)]),
                           
                           "Wreg":WassersteinCT(self.conf.output_channels,self.conf.imsize,
                           device=self.conf.device),
                           "CrossEntropy":nn.CrossEntropyLoss(),
                           "Wreg_mat":WassersteinCT_mat(self.conf.output_channels,self.conf.imsize,
                           device=self.conf.device),
                           "Distill": Distill(self.conf.device),
                           "DiceCT":DiceCT(num_organ=conf.output_channels-1,
                               imsize=self.conf.imsize,device=self.conf.device)
                           }
                           #"IoU":JaccardIndex(num_classes=self.conf.output_channels)}
    
    def setup_input(self,data):
        x, y_true, patient_id , slice_id = data
        self.conf.input, self.conf.gt = x.to(self.conf.device), y_true.to(self.conf.device)
        return self.conf
        
    def __call__(self,conf,isAux):
        self.conf = conf
        if isAux:
            pred = self.conf.pred_a
        else:
            pred = self.conf.pred
        gt = self.conf.gt

        if self.conf.loss == "Dice":
            loss_fn = self.loss_pool["Dice"]
            loss_fn.to(self.conf.device)
            loss = loss_fn(pred,gt)
            return loss
        elif self.conf.loss == "SoftDice":
            loss_fn = self.loss_pool["Dice"]
            loss_fn.to(self.conf.device)
            loss = loss_fn(pred,gt)
            return loss
        elif self.conf.loss == "Wreg":
            loss_fn1 = self.loss_pool["Dice"]
            loss_fn1.to(self.conf.device)
            loss_fn2 = self.loss_pool["Wreg"]
            loss_fn2.to(self.conf.device)
            loss = loss_fn1(pred,gt) 
            + loss_fn2(pred,gt,self.conf.input)
            return loss
        elif self.conf.loss == "Wreg_mat":
            loss_fn1 = self.loss_pool["Dice"]
            loss_fn1.to(self.conf.device)
            loss_fn2 = self.loss_pool["Wreg_mat"]
            loss_fn2.to(self.conf.device)
            loss = loss_fn1(pred,gt) 
            + loss_fn2(pred,gt,self.conf.input)
            return loss
        elif self.conf.loss == "CrossEntropy":
            loss_fn = self.loss_pool["CrossEntropy"]
            loss_fn.to(self.conf.device)
            loss = loss_fn(pred, torch.squeeze(gt,dim=1))
            return loss
        elif self.conf.loss == "DiceCT":
            loss_fn = self.loss_pool["DiceCT"]
            loss_fn.to(self.conf.device)
            loss = loss_fn(pred,gt,self.conf.input)
            return loss
        '''    
        elif self.conf.isuncertainty and isinstance(self.conf.loss,str):
            loss_fn = self.loss_pool["Distill"]
            loss_fn.to(self.conf.device)
            loss = loss_fn(self.conf.pred, torch.squeeze(self.conf.gt,dim=1))
            return loss
        '''

class DiceVal(nn.Module):

    def __init__(self):
        super(DiceVal, self).__init__()
        self.smooth = 1.0

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        y_pred = y_pred.contiguous().view(-1)
        y_true = y_true.contiguous().view(-1)
        intersection = (y_pred * y_true).sum()
        dsc = (2. * intersection + self.smooth) / (y_pred.sum() + y_true.sum() + self.smooth)
        return dsc

class One_Hot(nn.Module):
    def __init__(self, depth):
        super(One_Hot, self).__init__()
        self.depth = depth
        self.ones = torch.eye(depth)

    def forward(self, X_in):
        n_dim = X_in.dim()
        output_size = X_in.size() + torch.Size([self.depth])
        num_element = X_in.numel()
        X_in = X_in.data.long().view(num_element)
        out = torch.Variable(self.ones.index_select(0, X_in)).view(output_size)
        return out.permute(0, -1, *range(1, n_dim)).squeeze(dim=2).float()

    def __repr__(self):
        return self.__class__.__name__ + "({})".format(self.depth)

class CustomSoftDiceLoss(nn.Module):
    def __init__(self, n_classes, class_ids):
        super(CustomSoftDiceLoss, self).__init__()
        self.one_hot_encoder = One_Hot(n_classes).forward
        self.n_classes = n_classes
        self.class_ids = class_ids

    def forward(self, input, target):
        smooth = 0.01
        batch_size = input.size(0)

        input = F.softmax(input[:,self.class_ids], dim=1).view(batch_size, len(self.class_ids), -1)
        target = self.one_hot_encoder(target).contiguous().view(batch_size, self.n_classes, -1)
        target = target[:, self.class_ids, :]

        inter = torch.sum(input * target, 2) + smooth
        union = torch.sum(input, 2) + torch.sum(target, 2) + smooth

        score = torch.sum(2.0 * inter / union)
        score = 1.0 - score / (float(batch_size) * float(self.n_classes))

        return score

class DiceLoss(nn.Module):
    def __init__(self,num_organ,imsize,device):
        super(DiceLoss, self).__init__()
        self.num_organ=num_organ
        self.imsize = imsize
        self.device = device

    def forward(self, pred_stage1, target):
        pred_stage1 = F.softmax(pred_stage1, dim=1)
        
        organ_target = torch.zeros((target.size(0), self.num_organ, self.imsize, self.imsize))
        for organ_index in range(1, self.num_organ+1):
            temp_target = torch.zeros(target.size())
            temp_target[target == organ_index] = 1
            organ_target[:, organ_index-1, :, :] = torch.squeeze(temp_target)
        # loss
        dice_stage1 = 0.0

        organ_target = organ_target.to(self.device)
        for organ_index in range(1, self.num_organ+1):
            dice_stage1 += 2 * (pred_stage1[:, organ_index, :, :] * organ_target[:, organ_index-1 , :, :]).sum(dim=1).sum(
                dim=1) / (pred_stage1[:, organ_index, :, :].pow(2).sum(dim=1).sum(dim=1) +
                          organ_target[:, organ_index-1, :, :].pow(2).sum(dim=1).sum(dim=1) + 1e-5)

        dice_stage1 /= self.num_organ
        dice = dice_stage1.mean() 
        return (1.0-dice) #(1 - dice).mean()

class WassersteinCT(nn.Module):
    def __init__(self,num_organ,imsize,device):
        super(WassersteinCT, self).__init__()
        self.num_organ = num_organ
        self.imsize = imsize
        self.device = device
     
    def forward(self, pred_stage1, target,ct):
        pred_stage1 = torch.argmax(pred_stage1, dim=1)
        pred_stage1 = pred_stage1[None,:,:,:]
        organ_mask = torch.zeros((target.size(0),self.num_organ, self.imsize, self.imsize))
        seg_mask = torch.zeros((target.size(0),self.num_organ, self.imsize, self.imsize))
        
        for label in range(1,self.num_organ):
            temp_target = torch.zeros(target.size())
            temp_target[target == label] = 1
            organ_mask[:,label, :, :] = torch.squeeze(temp_target)
            temp_seg = torch.zeros(target.size())
            temp_seg[pred_stage1==label] =1
            seg_mask[:,label,:,:] = torch.squeeze(temp_seg)
            
        # loss
        dice_stage1 = 0.0
        organ_mask = organ_mask.to(self.device)
        seg_mask = seg_mask.to(self.device)
        ct_tile = torch.tile(ct,(1,5,1,1))
        loss = torch.mean(torch.abs(organ_mask*ct_tile - seg_mask*ct_tile))
        return loss

class WassersteinCT_mat(nn.Module):
    def __init__(self,num_organ,imsize,device):
        super(WassersteinCT_mat, self).__init__()
        self.num_organ = num_organ
        self.imsize = imsize
        self.device = device
     
    def forward(self, pred_stage1, target,ct):
        pred_stage1 = torch.argmax(pred_stage1, dim=1)
        pred_stage1 = pred_stage1[None,:,:,:]
        organ_mask = torch.zeros((target.size(0),self.num_organ, self.imsize, self.imsize))
        seg_mask = torch.zeros((target.size(0),self.num_organ, self.imsize, self.imsize))
        
        for label in range(0,self.num_organ):
            temp_target = torch.zeros(target.size())
            temp_target[target == label] = 1
            organ_mask[:,label, :, :] = torch.squeeze(temp_target)
            temp_seg = torch.zeros(target.size())
            temp_seg[pred_stage1==label] =1
            seg_mask[:,label,:,:] = torch.squeeze(temp_seg)
            
        # inter class texture difference matrix
        matrix = torch.zeros((self.num_organ,self.num_organ))
        matrix = matrix.to(self.device)
        organ_mask = organ_mask.to(self.device)
        seg_mask = seg_mask.to(self.device)

        for c in range(0,self.num_organ):
            for l in range(0,self.num_organ):
                matrix[c,l] = torch.mean(torch.abs(organ_mask[:,l,:,:]*ct - seg_mask[:,c,:,:]*ct))
        
        return torch.mean(matrix)

class Distill(nn.Module):
    def __init__(self,device, temperature = 1):
        super(Distill, self).__init__()
        self.device = device
        self.ce_fn = nn.MSELoss()
        self.T = temperature

    def forward(self,scores, targets):
        soft_pred = nn.functional.softmax(scores,1)
        soft_targets = nn.functional.softmax(targets,1)
        loss = nn.functional.mse_loss(soft_pred, soft_targets,reduction='mean')
        return loss

class DiceCT(nn.Module):
    def __init__(self,num_organ,imsize,device):
        super(DiceCT, self).__init__()
        self.num_organ=num_organ
        self.imsize = imsize
        self.device = device

    def forward(self, pred_stage1, target,ct):
        pred_stage1 = F.softmax(pred_stage1, dim=1)
        
        organ_target = torch.zeros((target.size(0), self.num_organ, self.imsize, self.imsize))
        for organ_index in range(1, self.num_organ+1):
            temp_target = torch.zeros(target.size())
            temp_target[target == organ_index] = 1
            organ_target[:, organ_index-1, :, :] = torch.squeeze(temp_target)
        # loss
        dice_stage1 = 0.0
        ct_weights = 0.0
        organ_target = organ_target.to(self.device)
        for organ_index in range(1, self.num_organ+1):
            dice_stage1 += 2 * (pred_stage1[:, organ_index, :, :] * organ_target[:, organ_index-1 , :, :]).sum(dim=1).sum(
                dim=1) / (pred_stage1[:, organ_index, :, :].pow(2).sum(dim=1).sum(dim=1) +
                          organ_target[:, organ_index-1, :, :].pow(2).sum(dim=1).sum(dim=1) + 1e-5)
            ct_weights += torch.mean(torch.abs(pred_stage1[:, organ_index, :, :] *ct - 
            organ_target[:, organ_index-1 , :, :]*ct))

        dice_stage1 = (dice_stage1*ct_weights) / self.num_organ
        dice = dice_stage1.mean() 
        return (1.0-dice) #(1 - dice).mean()