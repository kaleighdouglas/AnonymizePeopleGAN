from torchreid.utils import FeatureExtractor
from torchreid import metrics
import argparse
import os
from glob import glob
import pickle
import torch


def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--image_dir", type=str, required=True,
                        help="path directory of dir containing cropped images or pickle file of image features")
    parser.add_argument("--image_dir_targets", type=str, default='', required=False,
                        help="path directory of target images, if provided, compares each target image with all images in image_dir")
    return parser



def main():
    opts = get_argparser().parse_args()

    #### EXTRACTOR ####
    extractor = FeatureExtractor(
        model_name='osnet_ain_x1_0',
        model_path='weights/osnet_ain_x1_0_dukemtmcreid_256x128_amsgrad_ep90_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth',
        # model_path='weights/osnet_ain_x1_0_imagenet.pth',
        device='cpu'
    )


    #### LOAD IMAGES ####
    if opts.image_dir[-3:] == 'pkl':
        USE_SAVED_FEATURES = True
        with open(opts.image_dir, 'rb') as f:
            saved_image_features = pickle.load(f)
        image_files = list(saved_image_features.keys())

    else:
        USE_SAVED_FEATURES = False
        image_files = []
        if os.path.isdir(opts.image_dir):
            files = glob(os.path.join(opts.image_dir, '*.png'), recursive=False)
            if len(files)>0:
                image_files.extend(files)
        elif os.path.isfile(opts.image_dir):
            image_files.append(opts.image_dir)
    # print('image_files', image_files)

    #### LOAD TARGET IMAGES ####
    if opts.image_dir_targets[-3:] == 'pkl':
        USE_SAVED_TARGET_FEATURES = True
        with open(opts.image_dir_targets, 'rb') as f:
            saved_target_features = pickle.load(f)
        target_image_files = list(saved_target_features.keys())

    else:
        USE_SAVED_TARGET_FEATURES = False
        target_image_files = []
        if os.path.isdir(opts.image_dir_targets):
            files = glob(os.path.join(opts.image_dir_targets, '*.png'), recursive=False)
            if len(files)>0:
                target_image_files.extend(files)
        elif os.path.isfile(opts.image_dir_targets):
            target_image_files.append(opts.image_dir_targets)
        print('first five target_image_files', target_image_files[:5])

    
    # image_list = [
    #     'a/b/c/image001.jpg',
    #     'a/b/c/image002.jpg',
    #     'a/b/c/image003.jpg',
    #     'a/b/c/image004.jpg',
    #     'a/b/c/image005.jpg'
    # ]

    #### EXTRACT FEATURES ####
    if USE_SAVED_FEATURES:
        features = torch.stack(list(saved_image_features.values()))
        print('features', features.shape)
    else:
        features = extractor(image_files)
        print()
        print('features', features.shape) # output (5, 512)
        image_files = [name.split('.')[0].split('/')[-1] for name in image_files]

    #### EXTRACT TARGET FEATURES ####
    if target_image_files:
        if USE_SAVED_TARGET_FEATURES:
            target_features = torch.stack(list(saved_target_features.values()))
            print('target_features', target_features.shape)
        else:
            target_features = extractor(target_image_files)
            print('target_features', target_features.shape)
            target_image_files = [name.split('.')[0].split('/')[-1] for name in target_image_files]

    
    #### To Use Specific Target Images From Saved Training Images ####
    # target_image_files = ['aachen_000017_000019_13', 'zurich_000025_000019_0']
    # target_image_files = ['strasbourg_000000_029339_15']
    # target_features = []
    # for img in target_image_files:
    #     target_features.append(saved_image_features[img])
    # target_features = torch.stack(target_features)
    # print('target_features', target_features.shape)
    
    
    cd = {}
    # ed = {}
    #### COMPUTE COSINE DISTANCE ####
    if target_image_files:
        for i, target_img in enumerate(target_image_files):
            target_cd = {}
            # target_ed = {}
            target_feat = target_features[i,:].unsqueeze(0)

            for j, img in enumerate(image_files):
                image_feat = features[j,:].unsqueeze(0)

                # print(target_img, img)
                cos_dist = metrics.compute_distance_matrix(target_feat, image_feat, metric='cosine')
                cd[(target_img, img)] = round(cos_dist.item(), 4)
                target_cd[img] = round(cos_dist.item(), 4)
                # print('Cosine Distance:', round(cos_dist.item(), 4))

                # euc_dist = metrics.compute_distance_matrix(target_feat, image_feat)
                # ed[(target_img, img)] = round(euc_dist.item(), 4)
                # target_ed[img] = round(euc_dist.item(), 4)
                # print('Euclidean Distance:', round(euc_dist.item(), 4))

            #     if target_img == img:
            #         print()
            #         print()
            #         print(target_img, img)
            #         print('Cosine Distance:', round(cos_dist.item(), 4))
            #         # print('Euclidean Distance:', round(euc_dist.item(), 4))

            # #### PRINT MOST / LEAST SIMILAR ####
            # target_results_cd = sorted(target_cd.items(), key=lambda x: x[1], reverse=False)
            # print()
            # print('TARGET:', target_img)
            # # print()
            # print('Image pairs with lowest cosine distance')
            # print(target_results_cd[:6])
            # # print()
            # # print('Image pairs with highest cosine distance')
            # # print(target_results_cd[-5:])
            # # print()

            # # target_results_ed = sorted(target_ed.items(), key=lambda x: x[1], reverse=False)
            # # print()
            # # print('Image pairs with lowest euclidean distance')
            # # print(target_results_ed[:6])
            # # print()
            # # print('Image pairs with highest euclidean distance')
            # # print(target_results_ed[-5:])
            # # print()

    else:
        for i, target_img in enumerate(image_files):
            #### Target Features ####
            target_feat = features[i,:].unsqueeze(0)

            for j in range(i+1, len(image_files)):
                image_feat = features[j,:].unsqueeze(0)
                img = image_files[j]

                # print(target_img, img)
                cos_dist = metrics.compute_distance_matrix(target_feat, image_feat, metric='cosine')
                cd[(target_img, img)] = round(cos_dist.item(), 4)
                # print('Cosine Distance:', round(cos_dist.item(), 4))

                # euc_dist = metrics.compute_distance_matrix(target_feat, image_feat)
                # ed[(target_img, img)] = round(euc_dist.item(), 4)
                # # print('Euclidean Distance:', round(euc_dist.item(), 4))

    results_cd = sorted(cd.items(), key=lambda x: x[1], reverse=False)
    # results_ed = sorted(ed.items(), key=lambda x: x[1], reverse=False)


    #### PRINT MOST / LEAST SIMILAR ####
    if not target_image_files or True:
        print()
        print('------------------------------')
        print('LOWEST/HIGHEST COSINE DISTANCE')
        print('------------------------------')
        print()
        print('Image pairs with lowest cosine distance')
        for pair in results_cd[:10]:
            print(pair)
        print()
        print('Image pairs with highest cosine distance')
        for pair in results_cd[-10:]:
            print(pair)
        print()

        
        
        # print()
        # print('Image pairs with lowest euclidean distance')
        # print(results_ed[:10])
        # print('Image pairs with highest euclidean distance')
        # print(results_ed[-10:])
        # print()



    #### SAVE RESULTS - single target image ####
    try:
        filename_cd = 'torchreid_cosine_distance__' + opts.image_dir.split('.')[0].replace('/', '_') + '__' + opts.image_dir_targets.split('.')[0].replace('/', '_') + '.pkl'
        # filename_ed = 'torchreid_cosine_distance__' + opts.image_dir.split('.')[0].replace('/', '_') + '__' + opts.image_dir_targets.split('.')[0].replace('/', '_') + '.pkl'
    except:
        filename_cd = 'compare_all_features_torchreid_cosine_distance.pkl'
        # filename_ed = 'compare_all_features_torchreid_euclidean_distance.pkl'
    try:
        with open(filename_cd, 'wb') as f:
            pickle.dump(results_cd, f)
            print('SAVED DISTANCES - ', filename_cd)
    except:
        print('error saving results - file already exists', filename_cd)
    # try:
    #     with open(filename_ed, 'wb') as f:
    #         pickle.dump(results_ed, f)
    # except:
    #     print('error saving results - file already exists', filename_ed)




if __name__ == '__main__':
    main()