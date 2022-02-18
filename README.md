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
    --netG='unet_256' --netG_mask_input --netG_noise='decoder' --gan_mode_image='lsgan' --pool_size_image=50
    --gan_mode_person='vanilla' --pool_size_person=50 --name='EXP_SINGLE_STAGE'
```

To test the single-stage model:

```
$ python test.py --model='psgan' --dataroot='DATASETS'  --checkpoints_dir='CHECKPOINTS' --results_dir='RESULTS'
    --phase='test' --epoch='best' --num_test=500 --preprocess='resize' --load_size=256 --crop_size=256 
    --netG='unet_256' --netG_mask_input --netG_noise='decoder' --name='EXP_SINGLE_STAGE'
```


---
## Acknowledgements


pix2pix github repo: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix  
Corresponding paper:  Image-to-Image Translation with Conditional Adversarial Networks

PS-GAN github repo: https://github.com/yueruchen/Pedestrian-Synthesis-GAN  
This code corresponds to the paper: Pedestrian-Synthesis-GAN: Generating Pedestrian Data in Real Scene and Beyond  

torchreid github repo: https://github.com/KaiyangZhou/deep-person-reid  
