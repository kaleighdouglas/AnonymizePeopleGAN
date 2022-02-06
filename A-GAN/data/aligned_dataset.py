import os
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset, make_dataset_pix2pix
from PIL import Image
import json
import numpy as np
import torch
import torchvision.transforms as transforms

class AlignedDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)

        self.model = opt.model
        # self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory    #### CHANGE readme to specify 'images' dir
        
        if opt.model == 'pix2pix' and False:   #### NO-BBOX VERSION -- 
            self.dir_AB = os.path.join(opt.dataroot, 'images', opt.phase)            # get the image directory
            self.AB_paths = sorted(make_dataset_pix2pix(self.dir_AB, opt.max_dataset_size))  # get image paths
        else:
            self.dir_AB = os.path.join(opt.dataroot, 'images', opt.phase)  # get the image directory
            self.dir_bbox = os.path.join(opt.dataroot, 'bbox', opt.phase)  # get the bbox directory
            self.AB_paths, self.bbox_paths = make_dataset(self.dir_AB, self.dir_bbox)
            self.AB_paths = sorted(self.AB_paths)
            self.bbox_paths = sorted(self.bbox_paths)
            self.netD_person = opt.netD_person

        # assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc


    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')

        # split AB image into A and B
        w, h = AB.size
        w2 = int(w / 2)
        A = AB.crop((0, 0, w2, h))
        B = AB.crop((w2, 0, w, h))


        #### Get bbox data
        bbox_path = self.bbox_paths[index]
        bbox = json.load(open(bbox_path))
        # bbox = [bbox['x'], bbox['y'], bbox['w'], bbox['h']]     ##### OLD VERSION
        bbox = [bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']]
        # print('bbox width orig', bbox[2]-bbox[0])
        # print('bbox height orig', bbox[3]-bbox[1])

        #### Create mask from bbox
        img_mask = torch.zeros((w2, h))   #torch.Size([256, 256])
        img_mask[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 1
        img_mask = img_mask.unsqueeze(0)
        # print('img_mask', img_mask.size())


        #### Set minimum bbox size depending on person discriminator
        if self.netD_person == 'spp' and self.opt.crop_size == 256:
            min_bbox_size = 32
        # elif self.netD_person == 'spp_128':
        #     min_bbox_size = 17
        else:
            min_bbox_size = 32 // (256 / self.opt.crop_size)
            # min_bbox_size = 10 #4

        #### Transform images until minimum bbox size is met (error raised after 5 attempts)
        bbox_width = -1
        bbox_height = -1
        i = 0
        while bbox_width < min_bbox_size or bbox_height < min_bbox_size:
            # if i > 0:
            #     print('---- loop', i)
            #     print('bbox_width',bbox_width)
            #     print('bbox_height',bbox_height)
            #     print('A size', A.size)
            if i == 5:
                print('size error with bbox transformation')
                print('bbox_width',bbox_width)
                print('bbox_height',bbox_height)
                raise ValueError('size h/w error with bbox transformation:', bbox_height, bbox_width, AB_path, min_bbox_size)
    
            ## Get transform params for A, B and mask
            transform_params = get_params(self.opt, A.size, i)
            # print('transform_params:', transform_params)

            ## Transform mask
            BBOX_transform = get_transform(self.opt, transform_params, grayscale=False, convert=False, mask=True)
            mask = BBOX_transform(img_mask).squeeze(0)
            # print('transformed mask', mask.size())

            ## Get new bbox coords from transformed mask
            nonzero_row, nonzero_col = torch.nonzero(mask, as_tuple=True)  # Lists indices of nonzero elements in tensor
            x1 = torch.min(nonzero_col).item()
            x2 = torch.max(nonzero_col).item() + 1  ## max is always 1 more than edge
            y1 = torch.min(nonzero_row).item()
            y2 = torch.max(nonzero_row).item() + 1  ## max is always 1 more than edge
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            i += 1 

        #### BBOX coordinates after transformation
        bbox = [x1, y1, x2, y2]
        # print('bbox transformed:', bbox)
        # print('bbox_width',bbox_width)
        # print('bbox_height',bbox_height)

        #### Apply the same transform to both A and B
        img_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        A = img_transform(A)
        B = img_transform(B)

        #### Change bbox noise
        if self.opt.bbox_noise == 'random':
            ## Create noise patch in Black & White & Grey
            randnoise = torch.Tensor(np.random.choice([-1,0,1], (bbox[3]-bbox[1], bbox[2]-bbox[0])))
            ## Stack Noise in 3 channels
            randnoise = torch.stack((randnoise,randnoise,randnoise))
            ## Add noise patch to image B
            B[:, bbox[1]:bbox[3], bbox[0]:bbox[2]] = randnoise
        elif self.opt.bbox_noise == 'none':
            ## Create noise patch in Solid Grey
            randnoise = torch.zeros((bbox[3]-bbox[1], bbox[2]-bbox[0]))
            ## Stack Noise in 3 channels
            randnoise = torch.stack((randnoise,randnoise,randnoise))
            ## Add noise patch to image B
            B[:, bbox[1]:bbox[3], bbox[0]:bbox[2]] = randnoise
        
        # print('A', A.size())
        # print('B', B.size())
        # print('bbox', bbox)
        if self.opt.model == 'progan':
            g2_img_size = A.shape[-1]
            g1_img_size = int(self.opt.netG.split('_')[-1])
            assert(g2_img_size >= g1_img_size)
            assert(g2_img_size % g1_img_size == 0)
            scale = g2_img_size // g1_img_size
            # print('scale', scale)
            A_small = transforms.functional.resize(A, g1_img_size, Image.BICUBIC)
            B_small = transforms.functional.resize(B, g1_img_size, Image.BICUBIC)
            bbox_small = [int(x//scale) for x in bbox]  # Scale bbox
            return {'A_big': A, 'B_big': B, 'bbox_big': bbox, 'A_paths': AB_path, 'B_paths': AB_path, 'bbox_small':bbox_small, 'A_small':A_small, 'B_small':B_small}
        
        return {'A': A, 'B': B, 'bbox': bbox, 'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)
