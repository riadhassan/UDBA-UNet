import numpy as np
import nibabel as nib
import os
import pandas as pd
from matplotlib import pyplot as plt

data_path = "/Users/lyra/Medical/train_logs/"
models = ['LCTSC_CrossEntropy_attn_raw', 'LCTSC_AttUNet']
sample_name = 'Patient_002'
images = []
masks = []
gts = []
label =1
for model in models:
    path = os.path.join(data_path,model,"Results")
    img = nib.load(os.path.join(path,sample_name+".nii.gz")).get_fdata()
    msk = nib.load(os.path.join(path,sample_name+"_mask.nii.gz")).get_fdata()
    gt = nib.load(os.path.join(path,sample_name+"_GT.nii.gz")).get_fdata()
    #img = (img.astype(np.uint8)
    pred = msk*0
    GT = gt*0 
    pred[msk==1]=1
    GT[gt==1]=1
    ct_in_msk = (img * pred).flatten()
    #ct_in_msk = ct_in_msk.reshape(ct_in_msk.shape[0],1)
    ct_in_gt = (img *GT).flatten()
    #ct_in_gt = ct_in_gt.reshape(ct_in_gt.shape[0],1)
    '''
    data = np.concatenate([ct_in_msk,ct_in_gt],axis=1)
    dist = pd.DataFrame(data,columns=['pred','gt'])
    fig, ax = plt.subplots()
    #dist.plot.kde(ax=ax, legend=False, title='Histogram: Pred vs. GT')
    dist.plot.hist(ax=ax,bins=200,rwidth=0.9,title='Histogram: Pred vs. GT')
    ax.set_ylabel('Probability')
    ax.grid(axis='y')
    ax.set_facecolor('#d8dcd6')
    plt.show()
    '''
    import random
    #x = [random.gauss(3,1) for _ in range(400)]
    #y = [random.gauss(4,2) for _ in range(400)]
    #bins = np.linspace(-127,127, 256)
    hist_p, bin_p = np.histogram(ct_in_msk, bins=256, range=(0, 1))
    hist_gt, bin_gt = np.histogram(ct_in_gt, bins=256, range=(0, 1))
    plt.hist(hist_p,bin_p, alpha=0.5, label='Pred')
    plt.hist(hist_gt,bin_gt, alpha=0.5, label='GT')
    plt.legend(loc='upper right')
    plt.show()

    

    




