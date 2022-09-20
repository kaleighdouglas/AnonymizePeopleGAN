# Anonymizing People in Images Using Generative Adversarial Networks

MSc AI Thesis: Anonymizing People in Images Using Generative Adversarial Networks

Image datasets captured from public spaces are used in many applications and are especially crucial for computer vision tasks requiring real-world data. However, these datasets pose an inherent risk to the people appearing in the images and are often subject to strict privacy regulations that dictate their use and distribution. Through image anonymization, which aims to remove the identifiable aspects of people from images, we can mitigate the privacy issues associated with image datasets, allowing them to be freely shared for collaboration, future research, and peer review.

In this work, we present our research on methods of generating and evaluating realistic anonymized image datasets that can be used in a wide range of applications. We use conditional Generative Adversarial Networks to develop models for generating anonymized people in place of the identifiable people who appear in the original images. Furthermore, in the absence of an industry-standard evaluation method for person anonymization, we also propose anonymity and diversity metrics as part of a comprehensive method for evaluating the anonymity and realism of generated anonymized image datasets.



<img width="600" alt="example_original_anonymized_image" src="https://user-images.githubusercontent.com/8717892/152705101-c2e5487b-061c-49a6-94ff-5101ccade3c7.png">

---

## Project Folder Structure


1) [`A-GAN`](./A-GAN): Folder with code for generating images (adapted from pix2pix code)
1) [`data-preprocessing`](./data_preprocessing): Folder with code for pre-processing the CityScapes dataset into the required format
1) [`evaluation`](./evaluation): Folder with code for evaluating fidelity, diversity, and anonymity of generated anonymized images
1) [`evaluation/deep-person-reid`](./evaluation/deep-person-reid): Folder with torchreid library for person re-identification


---


## Usage


To train the single-stage model:

```
$ python train.py --model='psgan' --name='EXP_SINGLE_STAGE'
    --dataroot='datasets' --preprocess='resize_crop_contrast'
    --batch_size=1 --n_epochs=100 --n_epochs_decay=100 --lr=0.0002 --clip_value=100
    --netG='unet_256' --netG_mask_input --netG_noise='decoder' 
    --pool_size_image=50 --pool_size_person=50 
```

To test the single-stage model:

```
$ python test.py --model='psgan' --name='EXP_SINGLE_STAGE' 
    --phase='test' --epoch='best' --num_test=500
    --dataroot='datasets' --preprocess='resize' --load_size=256 --crop_size=256 
    --netG='unet_256' --netG_mask_input --netG_noise='decoder'
```

To train the two-stage model:

```
python train.py --model='progan' --name='EXP_TWO_STAGE'
    --dataroot='datasets' --preprocess='resize_crop_contrast' --load_size=286 --crop_size=256
    --batch_size=1 --n_epochs=150 --n_epochs_decay=50 --stage_1_epochs=100 --pretrain_iters_stage_2=1000
    --lr=0.00012 --clip_value=10 --lambda_L1=80 --lambda_L1_stage_2=80
    --netG='unet_128' --netG2='unet_256' --netG_mask_input --netG2_mask_input --netG_noise='decoder' --unet_diff_map
    --netD_image='n_layers' --n_layers_D=2 --netD2_image='basic' --netD_person='spp_128' --netD2_person='spp' 
    
```

To test the two-stage model:

```
python test.py --model='progan' --name='EXP_TWO_STAGE' 
    --phase='test' --epoch='best' --num_test=500
    --dataroot='datasets' --preprocess='resize' --load_size=256 --crop_size=256
    --netG='unet_128' --netG2='unet_256' --netG_mask_input --netG2_mask_input --netG_noise='decoder' --unet_diff_map
    --netD_person='spp_128' --netD2_person='spp' 
```


---
## Acknowledgements


pix2pix github repo: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix  
Corresponding paper:  Image-to-Image Translation with Conditional Adversarial Networks

PS-GAN github repo: https://github.com/yueruchen/Pedestrian-Synthesis-GAN  
Corresponding paper: Pedestrian-Synthesis-GAN: Generating Pedestrian Data in Real Scene and Beyond  

torchreid github repo: https://github.com/KaiyangZhou/deep-person-reid  
