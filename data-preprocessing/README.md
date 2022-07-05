## Data Preprocessing for PS-GAN

### CityScapes Dataset - https://www.cityscapes-dataset.com/ (leftImg8bit_trainvaltest)
### CityPersons Annotations - https://github.com/cvgroup-njust/CityPersons (gtBbox_cityPersons_trainval)

In order to run the PS-GAN code, we need to create data that looks like the provided files in the 'original_data' folder.  This directory contains code used to preprocess the cityscapes dataset into the correct format for PS-GAN.

### File Setup
1) Place train and val image data from CityScapes into the 'images' folder (images can remain in city specific folders)
2) Place train and val bounding box (bbox) annotation data into the 'bboxes' folder (files can remain in city specific folders)

### Split Cityscapes Dataset.ipynb
This code combines the filenames from all cities in the 'images' directory and splits the dataset into train, validation, and test sets.  
The train set contains 2495 images, the validation set contains 500 images, and the test set contains 500 images.
Each set contains images from all 21 cities present in the original CityScapes training and validation sets.
The code saves a list of filenames for each of the train/val/test sets: train_images.pkl, val_images.pkl, and test_images.pkl

### data_preprocessing_cityscapes.ipynb
This code takes the images from the 'images' directory, the annotations from the 'bboxes' directory and creates cropped 256 x 256 'original' and 'noisy' images of each person annotated in the CityPersons bboxes data.  
#### Image Output
The 'noisy' cropped images contain a rectangular region of black, white, and gray noise covering a single person in the image. The noisy region is in the shape of the visible bounding box (bboxVis) from the image's corresponding bboxes annotation file.  
The 'original' cropped images are simply the identical cropped images without the addition of the noisy region.  
If specified, the output images are separated into the train/val/test sets as designated by the train_images.pkl, val_images.pkl, and test_images.pkl files.
The 'original' cropped images are saved in the 'A' directory.  The noisy cropped images are saved in the 'B' directory.
#### BBOX Output
For each noisy cropped image, the code also outputs a json file contain the x and y coordinates of the upper left and lower right corners of the noisy bounding box. These json files are saved in the 'bboxes_AB' directory.

### combine_A_and_B.py
This file was taken from pix2pix - https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/datasets/combine_A_and_B.py
This code takes the original cropped image from directory 'A' and the noisy cropped image from directory 'B' and combines the images into a single side-by-side image format that is required of the PS-GAN code.  The combined image is saved in the 'AB' directory.

### Final Dataset Format
The image pairs saved in the 'AB' directory and the corresponding bbox coordinates saved in the 'bboxes_AB' directory are in the correct format to be used in the PS-GAN code.

The final dataset has the following number of cropped image pairs:
- training data: 5667
- validation data: 1063
- test data: 1235




