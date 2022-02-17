import argparse
import os
from glob import glob

from PIL import Image
from torchvision import transforms as T


def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--image_dir", type=str, required=True,
                        help="path directory of images")
    parser.add_argument("--ignore_img_suffix", type=str, default='', required=False,
                        help="part of image name to ignore (for example, '_fake_B')")
    parser.add_argument("--save_dir", type=str, default='fake_B_selected', required=False,
                        help="path directory of images")
    return parser



def main():
    opts = get_argparser().parse_args()

    # remove opts.ignore_img_suffix (for example, '_fake_B') from image filename so that it matches with bbox filename
    if opts.ignore_img_suffix:
        for file in os.listdir(opts.image_dir):
            if file.endswith('.png'):
                # print(file)
                os.rename(opts.image_dir+'/'+file, opts.image_dir+'/'+file.replace('_fake_B', ''))


    path = os.path.join(opts.image_dir, opts.save_dir)
    print('path', path)
    os.makedirs(path, exist_ok=True)
    # raise

    

    # Setup dataloader
    image_files = []
    if os.path.isdir(opts.image_dir):
        files = glob(os.path.join(opts.image_dir, '*.png'), recursive=False)
        if len(files)>0:
            image_files.extend(files)
    elif os.path.isfile(opts.image_dir):
        image_files.append(opts.image_dir)
    print('image_files', len(image_files))

    

    # ### SELECT VALIDATION IMAGES
    # select_images = ['aachen_000018_000019_10',
    #                 'bremen_000028_000019_2',
    #                 'bremen_000176_000019_9',
    #                 'erfurt_000047_000019_3',
    #                 'erfurt_000070_000019_4',
    #                 'ulm_000077_000019_1',
    #                 'zurich_000076_000019_0',
    #                 'aachen_000154_000019_0',
    #                 'aachen_000154_000019_1',
    #                 'zurich_000072_000019_27',
    #                 'aachen_000024_000019_2',
    #                 'bochum_000000_006746_1',
    #                 'bremen_000235_000019_0',
    #                 'dusseldorf_000113_000019_7',
    #                 'frankfurt_000001_073911_0',
    #                 'zurich_000072_000019_24',
    #                 'zurich_000087_000019_7',
    #                 'bochum_000000_006484_0',
    #                 'erfurt_000047_000019_5',
    #                 'hamburg_000000_056508_12',
    #                 'hanover_000000_039021_21',
    #                 'strasbourg_000001_000710_5',
    #                 'strasbourg_000001_017469_4',
    #                 'zurich_000106_000019_3']

    ## List of select images to copy

    ## PIX2PIX
    # select_images = ['strasbourg_000001_051574_28', 'strasbourg_000000_029915_4', 'weimar_000016_000019_0', 'frankfurt_000001_064130_3', 'zurich_000065_000019_41', 'frankfurt_000001_014406_14', 'hamburg_000000_000042_6', 'cologne_000015_000019_13', 'hamburg_000000_084746_6', 'tubingen_000078_000019_2']
    # select_images = ['strasbourg_000001_051574_28']
    ## PSGAN
    select_images = ['strasbourg_000001_051574_28', 'zurich_000006_000019_9', 'jena_000084_000019_5', 'strasbourg_000000_015506_17', 'hamburg_000000_038915_18', 'frankfurt_000001_011835_14', 'hanover_000000_027561_20', 'cologne_000053_000019_7']
    # select_images = ['strasbourg_000001_051574_28']
    ## SINGLE STAGE
    # select_images = ['strasbourg_000001_051574_28', 'frankfurt_000001_064130_3', 'zurich_000065_000019_41', 'tubingen_000085_000019_6', 'erfurt_000098_000019_1', 'strasbourg_000000_029915_4', 'weimar_000139_000019_0', 'zurich_000006_000019_9', 'cologne_000023_000019_23', 'zurich_000090_000019_0', 'hamburg_000000_039264_0']
    # select_images = ['strasbourg_000001_051574_28', 'frankfurt_000001_064130_3', 'hanover_000000_023881_11']
    ## TWO STAGE
    # select_images = ['strasbourg_000001_051574_28', 'strasbourg_000000_029915_4', 'cologne_000104_000019_1', 'zurich_000065_000019_31', 'krefeld_000000_017489_0', 'strasbourg_000000_029915_12', 'stuttgart_000088_000019_6', 'cologne_000135_000019_0', 'hamburg_000000_067338_4', 'zurich_000006_000019_9', 'aachen_000016_000019_27', 'frankfurt_000001_062793_4', 'strasbourg_000000_013223_7', 'weimar_000016_000019_0', 'cologne_000083_000019_11']
    # select_images = ['strasbourg_000001_051574_28', 'strasbourg_000000_029915_4']


    ## List of tuples of select image pairs to copy
    ## PIX2PIX
    # select_pairs = [('strasbourg_000001_051448_22', 'frankfurt_000000_020215_5'), ('jena_000111_000019_0', 'tubingen_000031_000019_0'), ('cologne_000023_000019_19', 'hamburg_000000_086636_7'), ('bremen_000098_000019_4', 'hamburg_000000_088939_9'), ('tubingen_000057_000019_3', 'strasbourg_000000_033062_11'), ('strasbourg_000001_051448_22', 'hanover_000000_028202_1'), ('jena_000046_000019_5', 'frankfurt_000000_020215_5'), ('bochum_000000_003245_0', 'frankfurt_000000_020215_5'), ('erfurt_000039_000019_1', 'strasbourg_000000_018358_1'), ('tubingen_000057_000019_3', 'stuttgart_000101_000019_0')]
    # select_pairs = [('zurich_000006_000019_1', 'zurich_000006_000019_5'), ('monchengladbach_000000_033683_0', 'hanover_000000_014319_4'), ('aachen_000047_000019_3', 'hamburg_000000_078407_1'), ('erfurt_000104_000019_0', 'munster_000145_000019_18'), ('strasbourg_000001_061285_6', 'hamburg_000000_048750_5'), ('frankfurt_000001_066438_4', 'strasbourg_000001_030725_4'), ('stuttgart_000179_000019_30', 'frankfurt_000001_055172_11'), ('hanover_000000_014319_4', 'bochum_000000_016591_0'), ('strasbourg_000000_029051_20', 'strasbourg_000001_034494_5'), ('dusseldorf_000093_000019_2', 'dusseldorf_000125_000019_0')]
    
    ## PSGAN
    # select_pairs = [('munster_000140_000019_16', 'frankfurt_000000_020215_5'), ('strasbourg_000000_029915_10', 'tubingen_000032_000019_2'), ('munster_000140_000019_16', 'tubingen_000032_000019_2'), ('hanover_000000_012675_3', 'tubingen_000050_000019_0'), ('strasbourg_000001_002519_6', 'tubingen_000032_000019_2'), ('munster_000140_000019_16', 'zurich_000078_000019_4'), ('strasbourg_000000_029915_10', 'hanover_000000_028202_1'), ('frankfurt_000001_011835_2', 'stuttgart_000077_000019_1'), ('strasbourg_000000_029915_10', 'hamburg_000000_100300_9'), ('strasbourg_000000_029915_10', 'strasbourg_000000_018358_1')]
    # select_pairs = [('strasbourg_000000_013223_7', 'krefeld_000000_017489_0'), ('dusseldorf_000057_000019_5', 'hanover_000000_023881_12'), ('hamburg_000000_027857_8', 'weimar_000139_000019_0'), ('strasbourg_000001_055698_5', 'stuttgart_000130_000019_3'), ('krefeld_000000_017489_0', 'frankfurt_000001_066438_3'), ('hanover_000000_014319_4', 'tubingen_000085_000019_6'), ('cologne_000023_000019_12', 'erfurt_000035_000019_15'), ('stuttgart_000088_000019_3', 'hanover_000000_023881_12'), ('strasbourg_000001_051448_29', 'strasbourg_000001_052297_23'), ('dusseldorf_000057_000019_2', 'bremen_000070_000019_4')]
    
    ## SINGLE STAGE
    # select_pairs = [('jena_000079_000019_6', 'tubingen_000069_000019_5'), ('bremen_000058_000019_1', 'aachen_000071_000019_16'), ('jena_000084_000019_9', 'hanover_000000_026804_4'), ('bremen_000098_000019_8', 'erfurt_000108_000019_0'), ('bremen_000014_000019_0', 'aachen_000033_000019_4'), ('stuttgart_000088_000019_5', 'aachen_000033_000019_4'), ('aachen_000114_000019_0', 'aachen_000071_000019_16'), ('erfurt_000098_000019_0', 'aachen_000048_000019_3'), ('strasbourg_000000_029915_7', 'frankfurt_000001_021825_3'), ('strasbourg_000000_029051_0', 'frankfurt_000001_079206_0')]
    # select_pairs = [('krefeld_000000_017489_0', 'krefeld_000000_017489_1'), ('erfurt_000098_000019_0', 'aachen_000016_000019_26'), ('aachen_000173_000019_1', 'aachen_000098_000019_6'), ('frankfurt_000000_002196_2', 'hamburg_000000_039264_2'), ('frankfurt_000001_044787_0', 'jena_000084_000019_7'), ('hanover_000000_040793_7', 'strasbourg_000001_052297_22'), ('aachen_000173_000019_1', 'cologne_000104_000019_0'), ('aachen_000087_000019_1', 'dusseldorf_000057_000019_8'), ('aachen_000098_000019_6', 'cologne_000104_000019_0'), ('aachen_000087_000019_1', 'erfurt_000074_000019_2')]
    
    ## TWO STAGE
    # select_pairs = [('cologne_000125_000019_1', 'munster_000147_000019_17'), ('cologne_000125_000019_8', 'weimar_000028_000019_9'), ('erfurt_000072_000019_4', 'stuttgart_000101_000019_0'), ('stuttgart_000134_000019_3', 'weimar_000099_000019_11'), ('strasbourg_000001_034494_5', 'munster_000147_000019_17'), ('bremen_000045_000019_18', 'strasbourg_000000_018358_1'), ('strasbourg_000000_015506_0', 'hamburg_000000_029676_8'), ('strasbourg_000001_052297_32', 'frankfurt_000001_007857_1'), ('munster_000140_000019_13', 'hamburg_000000_029676_8'), ('bremen_000045_000019_18', 'hanover_000000_028202_1')]
    # select_pairs = [('hanover_000000_012675_3', 'aachen_000047_000019_3'), ('strasbourg_000000_015506_0', 'frankfurt_000001_011835_2'), ('hamburg_000000_096063_9', 'hamburg_000000_096063_10'), ('cologne_000082_000019_22', 'aachen_000047_000019_3'), ('cologne_000082_000019_22', 'zurich_000042_000019_22'), ('cologne_000082_000019_22', 'hanover_000000_012675_3'), ('cologne_000082_000019_22', 'hamburg_000000_096063_9'), ('hamburg_000000_077642_0', 'strasbourg_000000_029051_14'), ('hamburg_000000_047157_32', 'frankfurt_000001_014406_15'), ('bremen_000056_000019_14', 'hamburg_000000_028439_11')]
    select_pairs = []

    if select_pairs:
        select_images = list(set([i for x in select_pairs for i in x]))
    print(select_images)
    print()
    print('num selected images:', len(select_images))


    

    # Image Scale
    img_size = Image.open(image_files[0]).size
    # print('img_size',img_size)    # img_size (512, 256)
    scale = 256//img_size[1]

    # Indicate if images are in AB format -- need to separate along width
    if img_size[0] > img_size[1]:
        SPLIT_WIDTH = img_size[0]//2 * scale
    else:
        SPLIT_WIDTH = False
    print('SPLIT_WIDTH:', SPLIT_WIDTH)



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

    # resize_crop = T.Resize((256, 128))

    transform_back = T.ToPILImage()


    ## ITERATE THROUGH IMAGES
    for img_path in image_files:
        img_name = os.path.basename(img_path).split('.')[0]
        if img_name in select_images:
            img = Image.open(img_path).convert('RGB')
                # print('original', img.size)
            
            if SPLIT_WIDTH:

                img = transform(img).unsqueeze(0)        ## To tensor of NCHW (256 x 256)
                # print('transformed', img.size())
            
                img = img[:,:,:,0:SPLIT_WIDTH]       ## Split imgage width-wise if in AB format
                # print('new img size',img.size())

                img = transform_back(img.squeeze(0))   ## Transform tensor back to PIL
                # print('final', img.size)

            img.save(os.path.join(opts.image_dir, opts.save_dir, img_name+'.png'))

            ## SAVE CROPPED IMAGES
            try:
                img.save(os.path.join(opts.image_dir, opts.save_dir, img_name+'.png'))
                print('saving', img_name)
            except:
                print('error saving file to cropped directory:', img_name)



    
if __name__ == '__main__':
    main()