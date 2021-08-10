import argparse
import torch
import pickle
import os
from glob import glob
import json

from PIL import Image
from torchvision import transforms as T
# from itertools import combinations
# import operator
# import random


def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--image_dir", type=str, required=True,
                        help="path directory of images")
    parser.add_argument("--bbox_dir", type=str, required=False,
                        help="path directory of images (if different from image_dir)")
    parser.add_argument("--ignore_img_suffix", type=str, default='', required=False,
                        help="part of image name to ignore (for example, '_fake_B')")
    return parser


# def crop_person(img, bbox):
#     crop = img[:, :, bbox['y']:bbox['h'], bbox['x']:bbox['w']]
#     return crop

# def crop_person(img, bbox_center):

#     center_x, center_y = bbox_center
#     crop = img[:, :, :, center_x-64:center_x+64]
#     # crop = img[:, :, center_y-128:center_y+128, center_x-64:center_x+64]
#     return crop

def crop_person(img, bbox_center, bbox):
    bbox_width = bbox['w']-bbox['x']
    bbox_height = bbox['h']-bbox['y']
    # print()
    # print('bbox_width', bbox_width)
    # print('bbox_height', bbox_height)

    if bbox_width <= (bbox_height/2):
        # print('case 1')
        if bbox_height % 2 == 0:
            height = bbox_height
        else:
            height = bbox_height + 1
        width = bbox_height // 2
    else:
        # print('case 2')
        if bbox_width % 2 == 0:
            width = bbox_width
        else:
            width = bbox_width + 1
        height = width * 2

    w_2 = width // 2
    h_2 = height // 2

    center_x, center_y = bbox_center

    y1 = center_y-h_2
    y2 = center_y+h_2
    x1 = center_x-w_2
    x2 = center_x+w_2
    # print('y1', y1)
    # print('y2', y2)
    # print('x1', x1)
    # print('x2', x2)

    if y1 < 0:
        y2 += abs(y1)
        y1 = 0
    elif y2 > 256:
        y1 = y1 - (y2 - 256)
        y2 = 256
    elif x1 < 0:
        x2 += abs(x1)
        x1 = 0
    elif x2 > 256:
        x1 = x1 - (x2 - 256)
        x2 = 256


    crop = img[:, :, y1:y2, x1:x2]
    # print('crop size', crop.size())
    # print('y1', y1)
    # print('y2', y2)
    # print('x1', x1)
    # print('x2', x2)

    return crop


def calc_center_bboxes(bbox_files):
    bbox_centers = {}
    bbox_data = {}
    for bbox_path in bbox_files:
        # name of file 
        img_name = os.path.basename(bbox_path).split('.')[0]

        # load bbox file
        bbox = json.load(open(bbox_path))
        
        # print('bbox width',bbox['w']-bbox['x'])
        # print('bbox height',bbox['h']-bbox['y'])

        bbox_data[img_name] = bbox
        # print(bbox)

        # calc center of bbox
        center_x = bbox['x'] + (bbox['w'] - bbox['x']) // 2
        center_y = bbox['y'] + (bbox['h'] - bbox['y']) // 2

        bbox_centers[img_name] = (center_x, center_y)
    return bbox_centers, bbox_data


# def get_bbox_data(bbox_files, scale=1):
#     bbox_data = {}
#     for bbox_path in bbox_files:
#         # name of file 
#         img_name = os.path.basename(bbox_path).split('.')[0]

#         # load bbox file
#         bbox = json.load(open(bbox_path))

#         # scale bbox
#         if scale != 1:
#             print('bbox scale:', scale)
#             bbox['x'] = bbox['x'] // scale
#             bbox['y'] = bbox['y'] // scale
#             bbox['w'] = bbox['w'] // scale
#             bbox['h'] = bbox['h'] // scale
#         # print('width',bbox['w']-bbox['x'])
#         # print('height',bbox['h']-bbox['y'])

#         bbox_data[img_name] = bbox
#         # print(bbox)
#     return bbox_data


def main():
    opts = get_argparser().parse_args()

    # remove opts.ignore_img_suffix (for example, '_fake_B') from image filename so that it matches with bbox filename
    if opts.ignore_img_suffix:
        for file in os.listdir(opts.image_dir):
            if file.endswith('.png'):
                # print(file)
                os.rename(opts.image_dir+'/'+file, opts.image_dir+'/'+file.replace('_fake_B', ''))


    # path = os.path.join(opts.image_dir, 'cropped_')
    # os.makedirs(path, exist_ok=True)
    # raise

    # try:
    #     original_umask = os.umask(0)
    #     path = os.path.join(opts.image_dir, 'cropped')
    #     os.makedirs(path, mode=777, exist_ok=True)
    # finally:
    #     os.umask(original_umask)

    

    # Setup dataloader
    image_files = []
    if os.path.isdir(opts.image_dir):
        for ext in ['png']:  #['png', 'jpeg', 'jpg', 'JPEG']
            files = glob(os.path.join(opts.image_dir, '*.png'), recursive=False)
            if len(files)>0:
                image_files.extend(files)
    elif os.path.isfile(opts.image_dir):
        image_files.append(opts.image_dir)
    # print('image_files', image_files)
    


    # select_images = ['hanover_000000_026356_9',
    #     'zurich_000062_000019_5',
    #     'munster_000049_000019_8',
    #     'darmstadt_000043_000019_2',
    #     'strasbourg_000000_017283_1',
    #     'erfurt_000068_000019_12',
    #     'monchengladbach_000000_018294_5',
    #     'hamburg_000000_103367_13',
    #     'aachen_000046_000019_1',
    #     'hamburg_000000_054850_8',
    #     'cologne_000148_000019_3',
    #     'frankfurt_000000_011461_1',
    #     'munster_000026_000019_0']
    select_images = []


    # Image Scale
    img_size = Image.open(image_files[0]).size
    # print('img_size',img_size)    # img_size (512, 256)
    scale = 256//img_size[1]

    # Indicate if images are in AB format -- need to separate along width
    if img_size[0] != img_size[1]:
        SPLIT_WIDTH = img_size[0]//2 * scale
    else:
        SPLIT_WIDTH = False

    # BBOX data
    bbox_files = []
    if opts.bbox_dir:
        bbox_path = opts.bbox_dir
    else: 
        bbox_path = opts.image_dir
    if os.path.isdir(bbox_path):
        for ext in ['json']:
            files = glob(os.path.join(bbox_path, '*.json'), recursive=False)
            if len(files)>0:
                bbox_files.extend(files)
    elif os.path.isfile(bbox_path):
        bbox_files.append(bbox_path)
    # print('bbox_files', bbox_files)
    bbox_centers, bbox_data = calc_center_bboxes(bbox_files)


    ## Image Transformations
    if scale != 1:
        transform = T.Compose([
                    T.Resize(256),
                    T.ToTensor(),
                ])
    else:
        transform = T.Compose([
                    T.ToTensor(),
                ])

    resize_crop = T.Resize((256, 128))

    transform_back = T.ToPILImage()


    ## ITERATE THROUGH IMAGES
    for img_path in image_files:
        img_name = os.path.basename(img_path).split('.')[0]
        if not select_images or img_name in select_images:

            img = Image.open(img_path).convert('RGB')
            # print('original', img.size)
            
            img = transform(img).unsqueeze(0)        ## To tensor of NCHW (256 x 256)
            # print('transformed', img.size())

            if SPLIT_WIDTH:
                img = img[:,:,:,0:SPLIT_WIDTH]       ## Split imgage width-wise if in AB format
                # print('new img size',img.size())

            try:
                img_cropped = crop_person(img, bbox_centers[img_name], bbox_data[img_name])  ## Crop image around person
                # print('cropped', img_cropped.size())

                img_cropped = resize_crop(img_cropped)    ## Resize cropped image to 256 x 128
                # print('resized cropped image', img_cropped.size())  #[1, 3, 256, 128]

                assert img_cropped.size()[2] == 256 
                assert img_cropped.size()[3] == 128

                img_cropped = transform_back(img_cropped.squeeze(0))   ## Transform tensor back to PIL
                # print('final', img_cropped.size)

            except:
                print('CROP ERROR:', img_name)
                print('cropped', img_cropped.size())
            

            ## SAVE CROPPED IMAGES
            try:
                img_cropped.save(os.path.join(opts.image_dir, 'cropped', img_name+'.png'))
            except:
                print('error saving file to cropped directory:', img_name)



    
if __name__ == '__main__':
    main()