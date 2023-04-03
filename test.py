import numpy as np
import torch
import argparse
import torch.nn as nn
from data_loader import*
from tqdm import tqdm
import glob
from torch.utils.tensorboard import SummaryWriter
from scipy.io import loadmat
import evaluate
import nibabel as nib
import pandas as pd
import os
from wraper import ModelWraper
from train import run_on_slices, save_validation_nifti

def conf():
    args = argparse.ArgumentParser()
    args.add_argument("--data_root",type=str,default="/home/nazib/Medical/Data/Dataset_LCTSC")
    args.add_argument("--model_path",type=str,default="/home/nazib/Medical/train_logs/LCTSC_R2Unet_Dice/LCTSC_R2Unet_Dice_epoch_0.pth")
    args.add_argument("--input_channels",type=int,default=1)
    args.add_argument("--output_channels",type=int,default=6)
    args.add_argument("--lr",type=float,default=0.01)
    args.add_argument("--batch_size",type=int,default=1)
    #args.add_argument("--save_dir",type=str,default="/home/nazib/Medical/train_logs")
    #args.add_argument("--model_name",type=str,default="Test")
    #args.add_argument("--printfq",type=int,default=10)
    #args.add_argument("--writerfq",type=int,default=10)
    #args.add_argument("--model_save_fq",type=bool,default=True)
    #args.add_argument("--debug_type",type=str,default="nifti",help="Two options: 1) nifti. 2)jpg")
    #args.add_argument("--num_epoch",type=int,default=100)
    #args.add_argument("--done_epoch",type=int,default=0)
    args.add_argument("--device",type=str,default="cuda")
    #args.add_argument("--loss",type=str,default="Dice")
    args.add_argument("--imsize",type=int,default=256)
    args.add_argument("--isprob",type=str,default='yes',help="Will calculate uncertainty")
    #args.add_argument("--aux_file",type=str,default="res_50")
    args.add_argument("--network_type",type=str,default="R2Unet")
    args = args.parse_args()
    
    return args

def test_model(conf):
  save_dir = os.path.join(os.path.dirname(conf.model_path),"Results")

  if not os.path.exists(save_dir):
    os.mkdir(save_dir)

  wraper = ModelWraper(conf)
  if conf.device == "cpu":
    wraper.seg_model.load_state_dict(torch.load(conf.model_path,map_location=torch.device('cpu'))['model_state_dict'])
  else:   
    wraper.seg_model.load_state_dict(torch.load(conf.model_path)['model_state_dict'])
  _, val_loader = data_loaders(conf.data_root)
  model_parameters = filter(lambda p: p.requires_grad, wraper.seg_model.parameters())
  params = sum([np.prod(p.size()) for p in model_parameters])
  print(f"Total Trainable Parameters: {params}")

  all_dice = []
  all_hd =[]
  all_iou = []
  with torch.no_grad():
      for i, data in enumerate(val_loader):
        vdata,patient = data
        img_vol,gt,seg,affine_mat = run_on_slices(wraper.seg_model,vdata,conf)
        #import pdb
        #pdb.set_trace()
        if "LCTSC" in conf.data_root:
           dice,hd,iou = evaluate.evaluate_case(seg,gt,evaluate.get_LCTSC_regions())
        else:
          dice,hd,iou = evaluate.evaluate_case(seg,gt,evaluate.get_Organ_regions())
        all_dice.append(dice)
        all_hd.append(hd)
        all_iou.append(iou)
        save_validation_nifti(img_vol,gt,seg,save_dir,patient,affine_mat)
    
      pd.DataFrame.from_dict(all_dice).to_csv(os.path.join(save_dir,"test_dice.csv"))
      pd.DataFrame.from_dict(all_hd).to_csv(os.path.join(save_dir,"test_hd.csv"))
      pd.DataFrame.from_dict(all_iou).to_csv(os.path.join(save_dir,"test_iou.csv"))

if __name__=="__main__":
  test_model(conf())