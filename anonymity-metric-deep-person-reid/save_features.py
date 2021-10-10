from torchreid.utils import FeatureExtractor
from torchreid import metrics
import argparse
import os
from glob import glob
import pickle


def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--image_dir", type=str, required=True,
                        help="path directory of images")
    return parser



def main():
    opts = get_argparser().parse_args()


    #### LOAD IMAGES ####
    image_files = []
    if os.path.isdir(opts.image_dir):
        files = glob(os.path.join(opts.image_dir, '*.png'), recursive=False)
        if len(files)>0:
            image_files.extend(files)
    elif os.path.isfile(opts.image_dir):
        image_files.append(opts.image_dir)
    print('image_files', len(image_files))


    #### EXTRACT FEATURES ####
    extractor = FeatureExtractor(
        model_name='osnet_ain_x1_0',
        model_path='weights/osnet_ain_x1_0_dukemtmcreid_256x128_amsgrad_ep90_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth',
        # model_path='weights/osnet_ain_x1_0_imagenet.pth',
        device='cpu'
    )

    # image_list = [
    #     'a/b/c/image001.jpg',
    #     'a/b/c/image002.jpg',
    #     'a/b/c/image003.jpg',
    #     'a/b/c/image004.jpg',
    #     'a/b/c/image005.jpg'
    # ]

    features = extractor(image_files)
    print()
    print('features', features.shape) # output (5, 512)


    image_files = [name.split('.')[0].split('/')[-1] for name in image_files]

    saved_features = {}
    for i, img in enumerate(image_files):
        saved_features[img] = features[i]
    # print('saved_features', saved_features.keys())

    #### SAVE RESULTS - single target image ####
    try:
        filename = 'saved_features_torchreid__' + opts.image_dir.split('.')[0].replace('/', '_') + '.pkl'
    except:
        filename = 'saved_features_torchreid.pkl'
    try:
        with open(filename, 'wb') as f:
            pickle.dump(saved_features, f)
            print('SAVED FEATURES - ', filename)
    except:
        print('error saving results - file already exists', filename)



if __name__ == '__main__':
    main()