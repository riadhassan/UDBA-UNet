import torch
import numpy as np
from PIL import Image
import os
def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.
    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)

def visualize(out, model_type,path,iter):
        if model_type !='subs':
            out = torch.argmax(out,dim=1).cpu()
        
        class_to_color = [torch.tensor([0, 0, 0]),torch.tensor([10, 133, 1]), torch.tensor([14, 1, 133]),  torch.tensor([33, 255, 1]), torch.tensor([243, 5, 247])]
        output = torch.zeros(1, 3, out.size(-2), out.size(-1), dtype=torch.float)
        for class_idx, color in enumerate(class_to_color):
            mask = out == class_idx
            if model_type!='subs':
                mask = mask.unsqueeze(1) # should have shape 1, 1, 100, 100
            curr_color = color.reshape(1, 3, 1, 1)
            segment = mask*curr_color # should have shape 1, 3, 100, 100
            output += segment
        output = tensor2im(output)
        Image.fromarray(output).save(f"{path}/pred_{model_type}_{iter}.png")
        #torchvision.utils.save_image(output, f"{self.conf.debug_path}/pred_{model_type}_{self.conf.iter}.png")