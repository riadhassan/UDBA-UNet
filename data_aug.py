import sys, os
import numpy as np
#from matplotlib import pyplot as plt
#from scipy.io import loadmat
#from skimage import io
import keras
from keras_preprocessing.image import transform_matrix_offset_center,Iterator,random_channel_shift,flip_axis
from keras.preprocessing.image import ImageDataGenerator

#from scipy.ndimage.interpolation import map_coordinates
#from scipy.ndimage.filters import gaussian_filter
#import scipy
#from scipy.ndimage import rotate, map_coordinates, gaussian_filter

_dir = os.path.join(os.path.realpath(os.path.dirname(__file__)), '')
data_path = os.path.join(_dir, '../')
aug_data_path = os.path.join(_dir, 'aug_data')

def random_zoom(x, y, zoom_range, row_index=1, col_index=2, channel_index=0,
                fill_mode='nearest', cval=0.):
    if len(zoom_range) != 2:
        raise Exception('zoom_range should be a tuple or list of two floats. '
                        'Received arg: ', zoom_range)

    if zoom_range[0] == 1 and zoom_range[1] == 1:
        zx, zy = 1, 1
    else:
        zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)
    zoom_matrix = np.array([[zx, 0, 0],
                            [0, zy, 0],
                            [0, 0, 1]])

    h, w = x.shape[row_index], x.shape[col_index]
    transform_matrix = transform_matrix_offset_center(zoom_matrix, h, w)
    x = apply_transform(x, transform_matrix, channel_index, fill_mode, cval)
    y = apply_transform(y, transform_matrix, channel_index, fill_mode, cval)
    return x, y


def random_rotation(x, y, rg, row_index=1, col_index=2, channel_index=0,
                    fill_mode='nearest', cval=0.):
    theta = np.pi / 180 * np.random.uniform(-rg, rg)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])

    h, w = x.shape[row_index], x.shape[col_index]
    transform_matrix = transform_matrix_offset_center(rotation_matrix, h, w)
    x = apply_transform(x, transform_matrix, channel_index, fill_mode, cval)
    y = apply_transform(y, transform_matrix, channel_index, fill_mode, cval)
    return x, y


def random_shear(x, y, intensity, row_index=1, col_index=2, channel_index=0,
                 fill_mode='constant', cval=0.):
    shear = np.random.uniform(-intensity, intensity)
    shear_matrix = np.array([[1, -np.sin(shear), 0],
                             [0, np.cos(shear), 0],
                             [0, 0, 1]])

    h, w = x.shape[row_index], x.shape[col_index]
    transform_matrix = transform_matrix_offset_center(shear_matrix, h, w)
    x = apply_transform(x, transform_matrix, channel_index, fill_mode, cval)
    y = apply_transform(y, transform_matrix, channel_index, fill_mode, cval)
    return x, y


def elastic_transform(image, mask, alpha, sigma, alpha_affine=None, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.
     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))


    res_x = map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)
    res_y = map_coordinates(mask, indices, order=1, mode='reflect').reshape(shape)
    return res_x, res_y


def elastic_transform1(image, mask, maskC, alpha, sigma, alpha_affine=None, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.
     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))


    res_x = map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)
    res_y = map_coordinates(mask, indices, order=1, mode='reflect').reshape(shape)
    res_z = map_coordinates(maskC, indices, order=1, mode='reflect').reshape(shape)
    return res_x, res_y, res_z


def augmentation_pytorch_only_image(x,imsize=256,trans_threshold=0.0, horizontal_flip=None,rotation_range=None,height_shift_range=None,width_shift_range=None,shear_range=None,zoom_range=None,elastic=None,add_noise=None): # 2D image 

        x=np.reshape(x,(1,imsize,imsize))
        
        h=imsize
        w=imsize
        row_index=1
        col_index=2
        
        if horizontal_flip:
            if np.random.random() < trans_threshold :
                x = flip_axis(x, 2)
        
        tep2=np.random.random() 
        if tep2 < trans_threshold :
            if rotation_range:
                theta = np.pi / 180.0 * np.random.uniform(-rotation_range, rotation_range)
            else:
                theta = 0
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                        [np.sin(theta), np.cos(theta), 0],
                                        [0, 0, 1]])
            if height_shift_range:
                tx = np.random.uniform(-height_shift_range, height_shift_range) * x.shape[row_index]
            else:
                tx = 0

            if width_shift_range:
                ty = np.random.uniform(-width_shift_range, width_shift_range) * x.shape[col_index]
            else:
                ty = 0

            translation_matrix = np.array([[1, 0, tx],
                                        [0, 1, ty],
                                        [0, 0, 1]])
            if shear_range:
                shear = np.random.uniform(-shear_range, shear_range)
            else:
                shear = 0
            shear_matrix = np.array([[1, -np.sin(shear), 0],
                                    [0, np.cos(shear), 0],
                                    [0, 0, 1]])

            if zoom_range[0] == 1 and zoom_range[1] == 1:
                zx, zy = 1, 1
            else:
                zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)
            zoom_matrix = np.array([[zx, 0, 0],
                                    [0, zy, 0],
                                    [0, 0, 1]])

            transform_matrix = np.dot(np.dot(np.dot(rotation_matrix, translation_matrix), shear_matrix), zoom_matrix)
 
            transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
        
            x = apply_transform(x, transform_matrix, 0,
                            fill_mode='nearest')

        if elastic is not None:
            if np.random.random() < trans_threshold :
                x, x = elastic_transform(x.reshape(imsize,imsize), x.reshape(imsize,imsize), *elastic)
                x = x.reshape(1, imsize, imsize)#, y.reshape(1, 256, 256)
        tep3=np.random.random()
        if add_noise>0:
            if  tep3< trans_threshold :
                x = x + 0.15 * x.std() * np.random.random(x.shape)
        return x

def augment_image_label_class(x,y,z, imsize=256, trans_threshold=0.0, horizontal_flip=None, rotation_range=None, height_shift_range=None, width_shift_range=None, shear_range=None, zoom_range = None, elastic=None, add_noise=None):
        x=np.reshape(x,(1,imsize,imsize))
        y=np.reshape(y,(1,imsize,imsize))  #force to reshape 
        z=np.reshape(z,(1,imsize,imsize))  #force to reshape 
        h=imsize
        w=imsize
        row_index=1
        col_index=2

        if horizontal_flip is not None:
            if np.random.random() < trans_threshold :
                x = flip_axis(x, 2)
                y = flip_axis(y, 2)
                z = flip_axis(z, 2)
        temp2=np.random.random()
        if temp2 < trans_threshold :

            if rotation_range is not None:
                #theta = np.pi / 180 * np.random.uniform(-rotation_range, rotation_range)
                theta = np.pi / 180 * np.random.uniform(-rotation_range, rotation_range)
            else:
                theta = 0.0
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                        [np.sin(theta), np.cos(theta), 0],
                                        [0, 0, 1]])
            if height_shift_range is not None:
                tx = np.random.uniform(-height_shift_range, height_shift_range) * x.shape[row_index]
            else:
                tx = 0

            if width_shift_range:
                ty = np.random.uniform(-width_shift_range, width_shift_range) * x.shape[col_index]
            else:
                ty = 0

            translation_matrix = np.array([[1, 0, tx],
                                        [0, 1, ty],
                                        [0, 0, 1]])
            if shear_range is not None:
                shear = np.random.uniform(-shear_range, shear_range)
            else:
                shear = 0
            shear_matrix = np.array([[1, -np.sin(shear), 0],
                                    [0, np.cos(shear), 0],
                                    [0, 0, 1]])

            if zoom_range[0] == 1 and zoom_range[1] == 1:
                zx, zy = 1, 1
            else:
                zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)
            zoom_matrix = np.array([[zx, 0, 0],
                                    [0, zy, 0],
                                    [0, 0, 1]])

            transform_matrix = np.dot(np.dot(np.dot(rotation_matrix, translation_matrix), shear_matrix), zoom_matrix)
            transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)

            img_gen = ImageDataGenerator()
            x = apply_transform(x, transform_matrix, 0)#,fill_mode='nearest')

            y = apply_transform(y, transform_matrix, 0)#,
            z = apply_transform(z, transform_matrix, 0)#,

            if elastic is not None:
               if np.random.random() < trans_threshold :
                  x, y, z = elastic_transform1(x.reshape(imsize,imsize), y.reshape(imsize,imsize), z.reshape(imsize, imsize), *elastic)
                  x, y, z = x.reshape(1, imsize, imsize), y.reshape(1, imsize, imsize), z.reshape(1, imsize, imsize)
            temp3=np.random.random()
            if add_noise is not None:
               if  temp3< trans_threshold :
                   x = x + 0.15 * x.std() * np.random.random(x.shape)
            
        return x.reshape(1,imsize,imsize), y.reshape(1,imsize,imsize), z.reshape(1, imsize,imsize)



def augment_image_label(x, y, imsize=256, trans_threshold=0.0, horizontal_flip=None,rotation_range=None,height_shift_range=None,width_shift_range=None,shear_range=None,zoom_range=None,elastic=None,add_noise=None): # 2D image 

        x=np.reshape(x,(1,imsize,imsize))
        y=np.reshape(y,(1,imsize,imsize))  #force to reshape 
        h=imsize
        w=imsize
        row_index=1
        col_index=2
        
        if horizontal_flip is not None:
            if np.random.random() < trans_threshold :
                x = flip_axis(x, 2)
                y = flip_axis(y, 2)
        tep2=np.random.random() 
        if tep2 < trans_threshold :

            if rotation_range is not None:
                theta = np.pi / 180.0 * np.random.uniform(-rotation_range, rotation_range)
            else:
                theta = 0
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                        [np.sin(theta), np.cos(theta), 0],
                                        [0, 0, 1]])
            if height_shift_range is not None:
                tx = np.random.uniform(-height_shift_range, height_shift_range) * x.shape[row_index]
            else:
                tx = 0

            if width_shift_range is not None:
                ty = np.random.uniform(-width_shift_range, width_shift_range) * x.shape[col_index]
            else:
                ty = 0

            translation_matrix = np.array([[1, 0, tx],
                                        [0, 1, ty],
                                        [0, 0, 1]])
            if shear_range is not None:
                shear = np.random.uniform(-shear_range, shear_range)
            else:
                shear = 0
            shear_matrix = np.array([[1, -np.sin(shear), 0],
                                    [0, np.cos(shear), 0],
                                    [0, 0, 1]])

            if zoom_range[0] == 1 and zoom_range[1] == 1:
                zx, zy = 1, 1
            else:
                zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)
            zoom_matrix = np.array([[zx, 0, 0],
                                    [0, zy, 0],
                                    [0, 0, 1]])

            transform_matrix = np.dot(np.dot(np.dot(rotation_matrix, translation_matrix), shear_matrix), zoom_matrix)
            transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
            transform_parameters = {'theta':theta,'tx':height_shift_range,
                                    'ty':width_shift_range, 'zx':zoom_range[0],
                                    'zy':zoom_range[1], 'flip_horizontal':horizontal_flip}

            img_gen = ImageDataGenerator()
            x = img_gen.apply_transform(x, transform_parameters)#,fill_mode='nearest')

            y = img_gen.apply_transform(y, transform_parameters)#,
                           # fill_mode='nearest')
        
        if elastic is not None:
            if np.random.random() < trans_threshold :
                x, y = elastic_transform(x.reshape(imsize,imsize), y.reshape(imsize,imsize), *elastic)
                x, y = x.reshape(1, imsize, imsize), y.reshape(1, imsize, imsize)
        tep3=np.random.random()
        if add_noise is not None:
            if  tep3< trans_threshold :
                x = x + 0.15 * x.std() * np.random.random(x.shape)
        #return x.reshape(1,1,imsize,imsize), y.reshape(1,1,imsize,imsize)           
        return x.reshape(1,imsize,imsize), y.reshape(1,imsize,imsize)           


