# Anonymizing People in Images Using Generative Adversarial Networks

MSc AI Thesis: Anonymizing People in Images Using Generative Adversarial Networks



<img width="600" alt="example_original_anonymized_image" src="https://user-images.githubusercontent.com/8717892/152705101-c2e5487b-061c-49a6-94ff-5101ccade3c7.png">

---

## Project Folder Structure


1) [`A-GAN`](./A-GAN): Folder with code for generating images (adapted from pix2pix code)
1) [`data_preprocessing`](./data_preprocessing): Folder with code for pre-processing the CityScapes dataset into the required format
1) [`evaluation`](./evaluation): Folder with code for evaluating fidelity, diversity, and anonymity of generated anonymized images
1) [`evaluation/deep-person-reid`](./evaluation/deep-person-reid): Folder with torchreid library for person re-identification


---


## Usage


To train the single-stage model:

```
$ python train.py --model='psgan' --dataroot='DATASETS'  --checkpoints_dir='CHECKPOINTS'
    --batch_size=1 --n_epochs=100 --n_epochs_decay=100 --lr=0.0002 --clip_value=100 --preprocess='resize_crop_contrast'
    --netG='unet_256' --netG_mask_input --netG_noise='decoder' 
    --pool_size_image=50 --pool_size_person=50 --name='EXP_SINGLE_STAGE'
```

To test the single-stage model:

```
$ python test.py --model='psgan' --dataroot='DATASETS'  --checkpoints_dir='CHECKPOINTS' --results_dir='RESULTS'
    --phase='test' --epoch='best' --num_test=500 --preprocess='resize' --load_size=256 --crop_size=256 
    --netG='unet_256' --netG_mask_input --netG_noise='decoder' --name='EXP_SINGLE_STAGE'
```

To train the two-stage model:

```
python train.py --model='progan' --dataroot='DATASETS' --checkpoints_dir='CHECKPOINTS'
    --batch_size=1 --n_epochs=150 --n_epochs_decay=50 --stage_1_epochs=100 --pretrain_iters_stage_2 1000
    --lr=0.00012 --clip_value=10 --preprocess='resize_crop_contrast' --load_size=286 --crop_size=256
    --netG='unet_128' --netG2='unet_256' --netG_mask_input --netG2_mask_input --netG_noise='decoder' --unet_diff_map
    --netD_image='n_layers' --n_layers_D=2 --netD2_image='basic' --netD_person='spp_128' --netD2_person='spp' 
    --lambda_L1=80 --lambda_L1_stage_2=80 --name='EXP_TWO_STAGE'
```

To test the two-stage model:

```
python test.py --model='progan' --dataroot='DATASETS' --checkpoints_dir='CHECKPOINTS' --results_dir='RESULTS'
    --num_test=500 --preprocess='resize' --load_size=256 --crop_size=256 --netD_person='spp_128' --netD2_person='spp'
    --netG='unet_128' --netG2='unet_256' --netG_mask_input --netG2_mask_input --netG_noise='decoder' --unet_diff_map
    --phase='test' --epoch='best' --name='EXP_TWO_STAGE'
```


---
## Acknowledgements


pix2pix github repo: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix  
Corresponding paper:  Image-to-Image Translation with Conditional Adversarial Networks

PS-GAN github repo: https://github.com/yueruchen/Pedestrian-Synthesis-GAN  
Corresponding paper: Pedestrian-Synthesis-GAN: Generating Pedestrian Data in Real Scene and Beyond  

torchreid github repo: https://github.com/KaiyangZhou/deep-person-reid  
