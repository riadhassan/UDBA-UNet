import numpy as np
import nibabel as nib
import os
import glob
from PIL import Image
from skimage.measure import label, regionprops
from skimage.morphology import dilation, binary_opening
import scipy.io as sio

def intesity_svaling(vol,lavel,window):
    max_value = window/2+lavel
    min_value = window/2-lavel
    
    vol[vol>max_value] = max_value
    vol[vol<-min_value] = -min_value
    return vol

def find_body(regions):
    max_area = regions[0].area
    indx = 0
    for i,x in enumerate(regions):
        if x.area >max_area:
            max_area=x.area
            indx =i
        else:
            continue
    return  regions[indx]


data_root = r"/Users/Segthor/Data/" 
dest_dir = r"/Users/Segthor/Data/train"
files = os.listdir(data_root)
files = [x for x in files if 'Patient_' in x]
files.sort()
lavel = 30
window = 400
imsize = 256

if not os.path.exists(dest_dir):
    os.mkdir(dest_dir)

for d in files[:34]:
    patient = d.split(os.sep)[-1]
    img = nib.load(os.path.join(data_root,d,patient+".nii.gz"))
    img_vol = img.get_fdata()
    gt = nib.load(os.path.join(data_root,d,"GT.nii.gz"))
    gt_vol = gt.get_fdata()

    ## intensity scaling 
    img_vol = intesity_svaling(img_vol,lavel=lavel,window=window)
    ##img_vol = img_vol/img_vol.max()
    bin_img = img_vol*0
    bin_img[img_vol>img_vol.min()]=1
    bin_img = binary_opening(image=bin_img,footprint=np.ones((5,5,5)))
    regions = regionprops(label(bin_img, connectivity=2))
    body = find_body(regions)
    box = body.bbox
    ## Extracting ROI with minimum axis among x and y
    indx = np.argmin(np.abs(np.array(box[0:2])-np.array(box[3:-1])))
    start = box[indx]
    end = box[indx+3] +20
    new_vol = img_vol[start:end,start:end,box[2]:box[5]]
    #new_vol = img_vol[box[0]:box[3],box[1]:box[4],:]
    new_vol = new_vol+np.abs(new_vol.min())
    new_vol = (new_vol/new_vol.max())*255
    new_vol = new_vol.astype(np.uint8)

    new_mask = gt_vol[start:end,start:end,box[2]:box[5]]
    #new_mask = gt_vol[box[0]:box[3],box[1]:box[4],box[2]:box[5]]
    '''
    new_img = nib.Nifti1Image(new_vol,img.affine,img.header)
    nib.save(new_img,"label.nii.gz")
    new_img = nib.Nifti1Image(new_mask,gt.affine,gt.header)
    nib.save(new_img,"mask.nii.gz")
    '''
    flag = True
    for i in range(new_mask.shape[2]):
        sl = new_mask[:,:,i]
        if sl.max()>0:
            img_sl = Image.fromarray(new_vol[:,:,i],'L').resize((imsize,imsize),Image.BILINEAR).transpose(Image.TRANSPOSE)
            img_sl.save(os.path.join(r"/Users/Segthor/RGB",f"{patient}_S_{i}.jpg"))
            mask_sl = np.asarray(Image.fromarray(new_mask[:,:,i]).resize((imsize,imsize),Image.NEAREST).transpose(Image.TRANSPOSE))
            if flag:
                sio.savemat(os.path.join(dest_dir,f"{patient}_S_{i}.mat"),{'img':np.asarray(img_sl),'mask':mask_sl,\
                'affine':img.affine,'header':img.header})
                flag=False
            else:
                sio.savemat(os.path.join(dest_dir,f"{patient}_S_{i}.mat"),{'img':np.asarray(img_sl),'mask':mask_sl})
        else:
            continue
    print(f"Procesed {patient}")

print("Done!!!")
#print(regions)

    


    

    





