from copy import deepcopy
from multiprocessing.pool import Pool

from medpy import metric
import numpy as np

### code is taken from 
#https://github.com/himashi92/VT-UNet/blob/main/VTUNet/vtunet/evaluation/region_based_evaluation.py

def get_Organ_regions():
    regions = {
        "esophagus": 1,
        "heart": 2,
        "trachea":3,
        "aorta":4
    }
    return regions


def create_region_from_mask(mask, join_labels: tuple):
    mask_new = np.zeros_like(mask, dtype=np.uint8)
    mask_new[mask == join_labels]=1
    '''
    for l in join_labels:
        mask_new[mask == l] = 1
    '''
    return mask_new

def evaluate_case(image_gt,image_pred, regions):
    #results = {}
    dice_values =[]
    hd_values = []
    for r in regions:
        mask_pred = create_region_from_mask(image_pred, regions[r])
        mask_gt = create_region_from_mask(image_gt, regions[r])
        dc = np.nan if np.sum(mask_gt) == 0 and np.sum(mask_pred) == 0 else metric.dc(mask_pred, mask_gt)
        hd = np.nan if np.sum(mask_gt) == 0 or np.sum(mask_pred) == 0 else metric.hd(mask_pred, mask_gt)
        dice_values.append(dc)
        hd_values.append(hd)
    
    #results["Dice"] = dice_values
    #results["HD"] = hd_values
    return dice_values,hd_values