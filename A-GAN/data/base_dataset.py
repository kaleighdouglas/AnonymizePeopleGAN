"""This module implements an abstract base class (ABC) 'BaseDataset' for datasets.

It also includes common transformation functions (e.g., get_transform, __scale_width), which can be later used in subclasses.
"""
import random
import numpy as np
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
from abc import ABC, abstractmethod
import torch

class BaseDataset(data.Dataset, ABC):
    """This class is an abstract base class (ABC) for datasets.

    To create a subclass, you need to implement the following four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.
    """

    def __init__(self, opt):
        """Initialize the class; save the options in the class

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.opt = opt
        self.root = opt.dataroot

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    @abstractmethod
    def __len__(self):
        """Return the total number of images in the dataset."""
        return 0

    @abstractmethod
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns:
            a dictionary of data with their names. It ususally contains the data itself and its metadata information.
        """
        pass


def get_params(opt, size, i=0):
    w, h = size
    new_h = h
    new_w = w
    if 'resize' in opt.preprocess and 'crop' in opt.preprocess:
        new_h = new_w = opt.load_size
    elif 'scale_width' in opt.preprocess and 'crop' in opt.preprocess:
        new_w = opt.load_size
        new_h = opt.load_size * h // w

    ## Get Crop Position
    if i == 0: #random x and y
        x = random.randint(0, np.maximum(0, new_w - opt.crop_size))
        y = random.randint(0, np.maximum(0, new_h - opt.crop_size))
    elif i == 1: #left
        x = 0 
        y = random.randint(0, np.maximum(0, new_h - opt.crop_size))
    elif i == 2: #right
        x = np.maximum(0, new_w - opt.crop_size - 1) 
        y = random.randint(0, np.maximum(0, new_h - opt.crop_size))
    elif i == 3: #top
        x = random.randint(0, np.maximum(0, new_w - opt.crop_size))
        y = 0
    elif i == 4: #bottom
        x = random.randint(0, np.maximum(0, new_w - opt.crop_size))
        y = np.maximum(0, new_h - opt.crop_size - 1)
    else: #random x and y
        x = random.randint(0, np.maximum(0, new_w - opt.crop_size))
        y = random.randint(0, np.maximum(0, new_h - opt.crop_size))

    flip = random.random() > 0.5

    return {'crop_pos': (x, y), 'flip': flip}


def get_transform(opt, params=None, grayscale=False, method=Image.BICUBIC, convert=True, mask=False):    #### CHECK what is method=Image.BICUBIC  #### ADDED bbox_mask arg
    # print('phase', opt.phase)
    # print('params', params)
    transform_list = []
    if mask:
        transform_list.append(transforms.ToPILImage()) #Converts mask to PIL before transforms
    if grayscale:
        transform_list.append(transforms.Grayscale(1))

    if 'resize' in opt.preprocess:
        osize = [opt.load_size, opt.load_size]
        transform_list.append(transforms.Resize(osize, method))
    elif 'scale_width' in opt.preprocess:
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.load_size, opt.crop_size, method)))

    if 'crop' in opt.preprocess:
        if params is None:
            transform_list.append(transforms.RandomCrop(opt.crop_size))
        else:
            transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt.crop_size)))

    if 'color' in opt.preprocess and opt.phase=='train' and not mask:  ## Do not alter color in mask, validation, and test images
        # transform_list.append(transforms.ColorJitter())
        color_jitter = transforms.ColorJitter(brightness=(0.9, 1.2)) #brightness=(0.9, 1.2), contrast=(0.9,1.2) contrast=0.1, saturation=0.1
        color_transform = transforms.ColorJitter.get_params(color_jitter.brightness, color_jitter.contrast, color_jitter.saturation, color_jitter.hue)
        transform_list.append(color_transform)
        # transform_list.append(transforms.Lambda(lambda img: __force_min_max(img)))

    if opt.preprocess == 'none':
        transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base=4, method=method)))

    if not opt.no_flip:
        if params is None:
            transform_list.append(transforms.RandomHorizontalFlip())                                   #### CHECK
        elif params['flip']:
            # transform_list.append(transforms.RandomHorizontalFlip(1.0))   
            transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))
    
    if mask:
        transform_list.append(transforms.ToTensor())  #Converts mask back to Tensor after transformations

    if convert:
        transform_list += [transforms.ToTensor()] #Converts a PIL Image or numpy.ndarray in the range [0, 255] to a torch.FloatTensor in the range [0.0, 1.0] (Scales data)
        if grayscale:
            transform_list += [transforms.Normalize((0.5,), (0.5,))]
        else:
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))] #Normalizes img tensor: output[channel] = (input[channel] - mean[channel]) / std[channel] --> output [-1,1]
    # print('transform_list', transform_list)
    return transforms.Compose(transform_list)


def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if h == oh and w == ow:
        return img

    __print_size_warning(ow, oh, w, h)
    return img.resize((w, h), method)


def __scale_width(img, target_size, crop_size, method=Image.BICUBIC):
    ow, oh = img.size
    if ow == target_size and oh >= crop_size:
        return img
    w = target_size
    h = int(max(target_size * oh / ow, crop_size))
    return img.resize((w, h), method)


def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img


def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img

def __force_min_max(img):
    # print('img max',torch.max(img))
    # print('img min',torch.min(img))
    print('min_max', img.getextrema())
    return img


def __print_size_warning(ow, oh, w, h):
    """Print warning information about image size(only print once)"""
    if not hasattr(__print_size_warning, 'has_printed'):
        print("The image size needs to be a multiple of 4. "
              "The loaded image size was (%d, %d), so it was adjusted to "
              "(%d, %d). This adjustment will be done to all images "
              "whose sizes are not multiples of 4" % (ow, oh, w, h))
        __print_size_warning.has_printed = True


# def get_bbox_transform(bbox, width, opt, params=None, method=Image.BICUBIC):    #### CHECK what is method=Image.BICUBIC  CHANGE add bbox transforms here/another function
#     # print('bbox transform params:', params)
#     # print(opt)

#     if 'resize' in opt.preprocess:
#         raise NotImplementedError('bbox resize function not implemented')
#     elif 'scale_width' in opt.preprocess:
#         raise NotImplementedError('bbox scale_width function not implemented')
#     if 'crop' in opt.preprocess:
#         raise NotImplementedError('bbox crop function not implemented')

#     # if opt.preprocess == 'none':
#     #     pass

#     if not opt.no_flip:
#         if params is None:
#             raise NotImplementedError('bbox random flip function not implemented')
#         elif params['flip']:
#             # print('original bbox:', bbox)
#             bbox_flipped = [width - bbox[2], bbox[1], width - bbox[0], bbox[3]]  # x1, y1, x2, y2
#             if bbox_flipped[0] < 0:
#                 bbox_flipped[0] = 0
#                 print('WARNING --- flipped bbox x1 < 0 ---', bbox, bbox_flipped)
#             if bbox_flipped[2] > width:
#                 bbox_flipped[2] = width
#                 print('WARNING --- flipped bbox x2 > width ---', bbox, bbox_flipped)

#             bbox = bbox_flipped
#             # print('flipped bbox:', bbox)

#     return bbox
