import numpy as np
import torch
import argparse
from networks import UNet
import torch.nn as nn
import torch.optim as optim
from data_loader import*
from loss import Loss 
from tqdm import tqdm
import glob
from torch.utils.tensorboard import SummaryWriter
from scipy.io import loadmat
import evaluate
import nibabel as nib
import pandas as pd
import os


def save_validation_nifti(img,gt,seg,path,patient,affine):
  new_img = nib.Nifti1Image(img,affine)
  nib.save(new_img,path+f"/{patient[0]}.nii.gz")
  new_img = nib.Nifti1Image(seg,affine)
  nib.save(new_img,path+f"/{patient[0]}_mask.nii.gz")
  new_img = nib.Nifti1Image(gt,affine)
  nib.save(new_img,path+f"/{patient[0]}_GT.nii.gz")


def get_prev_traning_data():
  last_epoch = 0
  checkPoints = sorted(glob.glob("/content/drive/MyDrive/dataset/Output/checkpoint/*.pt"))
  for checkPoint in checkPoints:
    checkEpoch = int(checkPoint.split('/')[-1].split('_')[1])
    if checkEpoch > last_epoch:
      last_epoch = checkEpoch
      last_checkpoint = checkPoint

  if len(checkPoints) != 0:
    return torch.load(last_checkpoint)
  else:
    return 0

def conf():
    args = argparse.ArgumentParser()
    args.add_argument("--data_root",type=str,default="/home/nazib/Medical/Data")
    args.add_argument("--input_channels",type=int,default=1)
    args.add_argument("--output_channels",type=int,default=5)
    args.add_argument("--lr",type=float,default=0.001)
    args.add_argument("--batch_size",type=int,default=1)
    args.add_argument("--save_dir",type=str,default="/home/nazib/Medical/train_logs")
    args.add_argument("--model_name",type=str,default="test")
    args.add_argument("--printfq",type=int,default=10)
    args.add_argument("--writerfq",type=int,default=10)
    args.add_argument("--model_save_fq",type=bool,default=True)
    args.add_argument("--debug_type",type=str,default="nifti",help="Two options: 1) nifti. 2)jpg")
    args.add_argument("--num_epoch",type=int,default=100)
    args.add_argument("--done_epoch",type=int,default=0)
    args.add_argument("--device",type=str,default="cuda")
    args.add_argument("--loss",type=str,default="Wreg")
    args.add_argument("--imsize",type=int,default=256)
    args = args.parse_args()
    return args
  
def create_dirs(conf):
  if not os.path.exists(os.path.join(conf.save_dir,conf.model_name)):
    os.mkdir(os.path.join(conf.save_dir,conf.model_name))
  model_path = os.path.join(conf.save_dir,conf.model_name)
  if not os.path.exists(os.path.join(conf.save_dir,conf.model_name,"curves")):
    os.mkdir(os.path.join(conf.save_dir,conf.model_name,"curves"))
    os.mkdir(os.path.join(conf.save_dir,conf.model_name,"debug"))
  
  curve_path = os.path.join(conf.save_dir,conf.model_name,"curves")
  debug_path = os.path.join(conf.save_dir,conf.model_name,"debug")
  return model_path,curve_path,debug_path

def run_on_slices(model,data,conf):
    seg_mask = []
    gt_mask = []
    img_vol = []
    for sl in data:
      mat = loadmat(sl[0])
      if 'affine' in mat.keys():
        affine = mat['affine']
        header = mat['header']
        
      mask = mat['mask']
      image = mat['img']
      mask = mask[None,:,:]
      image = (image - np.std(image))/np.mean(image)
      image = torch.from_numpy(image[None,None,:,:].astype(np.float32))
      image = image.to(conf.device)
      pred = model(image)
      pred = torch.argmax(pred,axis=1)
      gt_mask.append(mask)
      seg_mask.append(pred.detach().cpu().numpy())
      img_vol.append(image.detach().cpu().numpy())
  
    img_vol = np.transpose(np.squeeze(np.asarray(img_vol)).astype(np.float32),(1,2,0))
    gt_mask = np.transpose(np.squeeze(np.asarray(gt_mask)).astype(np.uint8),(1,2,0))
    seg_mask = np.transpose(np.squeeze(np.asarray(seg_mask)).astype(np.uint8),(1,2,0))
   
    return img_vol,gt_mask,seg_mask,affine
  
def main(conf):
    #device = torch.device("cpu" if not torch.cuda.is_available() else "mps")
    device = torch.device(conf.device)
    model = UNet(in_channels= conf.input_channels,out_channels= conf.output_channels)
    model.to(device)
    train_loader, val_loader = data_loaders(conf.data_root)
    print(model)

    loaders = {"train": train_loader, "valid": val_loader}
    model_path, log_path, debug_path =create_dirs(conf)
    writer = SummaryWriter(log_path)

    #loss_function = DiceLoss(num_organ=conf.output_channels-1,imsize=256,device=conf.device_type)
    #loss_function = WassersteinCT(conf.output_channels,imsize=256)
    loss_function = Loss(conf)
   
    optimizer = optim.Adam(model.parameters(), lr=conf.lr)

    prev_traning_data = get_prev_traning_data()
    if prev_traning_data:
      print("hello")
      done_epoch = prev_traning_data['epoch']
      loss = prev_traning_data['loss']
      model.load_state_dict(prev_traning_data['model_state_dict'])
      optimizer.load_state_dict(prev_traning_data['optimizer_state_dict'])
     
    all_dice_dict = []
    all_hd_dict =[]

    ###### Training #######
    total_iter = 0
    for epoch in tqdm(range(conf.done_epoch, conf.num_epoch+1)):
      print("Training...")
      #### Training Loop ###
      model.train()
      
      for i, data in enumerate(train_loader):
        conf = loss_function.setup_input(data)
        optimizer.zero_grad()
        conf.pred = model(conf.input)
        loss = loss_function(conf)  
        loss.backward()
        optimizer.step()

        if i % conf.printfq==0:
          print(f"Epoch:{epoch} Iter:{i} Loss:{loss:0.4f}")
        if i % conf.writerfq == 0:
          writer.add_scalar("train_loss",float(loss.item()),total_iter)
        total_iter+=1

      print(f"Enad of epoch: {epoch}. Now validating.....")
      model.eval()
      all_dice = []
      all_hd =[]
      with torch.no_grad():
        for i, data in enumerate(val_loader):
          vdata,patient = data
          img_vol,gt,seg,affine_mat = run_on_slices(model,vdata,conf)
          dice,hd = evaluate.evaluate_case(seg,gt,evaluate.get_Organ_regions())
          all_dice.append(dice)
          all_hd.append(hd)
          if conf.debug_type == "nifti":
            save_validation_nifti(img_vol,gt,seg,debug_path,patient,affine_mat)
      
      organ_dice = np.mean(all_dice,0) 
      organ_hd = np.mean(all_hd,0)
      dice_dict = {"Esophegus Dice":organ_dice[0],"Heart Dice":organ_dice[1],"Trachea Dice":organ_dice[2],"Aorta Dice":organ_dice[3]}
      hd_dict = {"Esophegus HD":organ_hd[0],"Heart HD":organ_hd[1],"Trachea HD":organ_hd[2],"Aorta HD":organ_hd[3]}
      print(dice_dict)
      print(hd_dict)
      all_dice_dict.append(dice_dict)
      all_hd_dict.append(hd_dict)
      pd.DataFrame.from_dict(all_dice_dict).to_csv(os.path.join(model_path,"Validation_dice.csv"))
      pd.DataFrame.from_dict(all_hd_dict).to_csv(os.path.join(model_path,"Validation_hd.csv"))
      
      ### Saving the best model ###
      if epoch<1:
        prev_mean_dice = 0
      
      model_name = f"{conf.model_name}_epoch_{epoch}.pth"
      curr_mean_dice = np.mean(organ_dice)
      if curr_mean_dice > prev_mean_dice:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            },os.path.join(model_path,model_name))
        prev_mean_dice = curr_mean_dice
      else:
        continue

if __name__=="__main__":
    main(conf())

