"""General-purpose training script for image-to-image translation.

This script works for --model='pix2pix' and with --dataset_mode='aligned'.
You need to specify the dataset ('--dataroot') and experiment name ('--name').

It first creates model, dataset, and visualizer given the options.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer, ValidationVisualizer, save_images, print_fidelity_metrics
import os
import random
import numpy as np
import torch
from util import html
import torch_fidelity
# import pytorch_lightning as pl


if __name__ == '__main__':
    ## Training options
    opt = TrainOptions().parse()

    ## Validation options
    opt_val = TrainOptions().parse('val')
    # opt_val.phase = 'val'
    # opt_val.num_threads = 0 
    # opt_val.batch_size = 1 
    # opt_val.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    # opt_val.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    # opt_val.max_dataset_size = 200
    # opt_val.load_size = opt_val.crop_size  # to avoid cropping for validation images, set load_size to equal crop_size
    
    ## Validation Visualization Set
    opt_val_viz = TrainOptions().parse('val5')
    # opt_val_viz.phase = 'val5'
    # opt_val_viz.num_threads = 0 
    # opt_val_viz.batch_size = 1 
    # opt_val_viz.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    # opt_val_viz.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    # opt_val_viz.max_dataset_size = 5
    # opt_val_viz.load_size = opt_val_viz.crop_size  # to avoid cropping for validation images, set load_size to equal crop_size
    
    ## Set the seed
    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    set_seed(opt.seed)

    # Setting the seed
    # pl.seed_everything(42)

    ## Ensure all operations are deterministic on GPU
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False

    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    opt.dataset_size = dataset_size
    print('The number of training images = %d' % dataset_size)
    validation_dataset = create_dataset(opt_val)  # create a validation dataset given opt.dataset_mode and other options  #### ADDED
    print('The number of validation images = %d' % len(validation_dataset))
    validation_viz_dataset = create_dataset(opt_val_viz)  # create a visualization validation dataset given opt.dataset_mode and other options  #### ADDED
    print('The number of visualization validation images = %d' % len(validation_viz_dataset))

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    val_model = create_model(opt_val)                                                               ####### !!!!! ADDED
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    validation_visualizer = ValidationVisualizer(opt_val_viz)   # create a visualizer that displays/saves validation images   #### ADDED
    # if opt.continue_train:
    total_iters = (opt.epoch_count - 1) * (dataset_size // opt.batch_size)  # the total number of training iterations
    print('starting total iters:', total_iters)
    # else:
    #     total_iters = 0                
    lr = opt.lr  #### ADDED
    best_val_fid = 10000 #### ADDED
    
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_images = 0                  # the number of training images processed in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        validation_visualizer.reset()   # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        # model.update_learning_rate()    # update learning rates in the beginning of every epoch.   #### CHANGED DUE TO WARNING MESSAGE - MOVED TO END #### CHECK
        print('total iters at start of epoch', total_iters)
        
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += 1 #opt.batch_size
            epoch_images += opt.batch_size

            if total_iters % opt.print_freq == 0:
                save_iter_data = True
            else:
                save_iter_data = False

            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters(total_iters, save_iter_data)   # calculate loss functions, get gradients, update network weights  #### CHANGED -- added total_iters

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                accuracies = model.get_current_accuracies() #### ADDED
                if opt.save_grads:
                    grads = model.get_current_grads() #### ADDED
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                # print('train losses', losses)
                # Plot Losses & Accuracies
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_images) / dataset_size, losses)
                    visualizer.plot_current_accuracies(epoch, float(epoch_images) / dataset_size, accuracies) #### ADDED
                    if opt.save_grads:
                        visualizer.plot_current_grads(epoch, float(epoch_images) / dataset_size, grads) #### ADDED
                # Print Losses & Accuracies
                losses.update(accuracies)  #### ADDED
                if opt.save_grads:
                    losses.update(grads)  #### ADDED
                visualizer.print_current_losses(epoch, total_iters, losses, t_comp, t_data, lr)   #### ADDED lr  #### CHANGED epoch_images(epoch_iter) to total_iters

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()

        if epoch % opt.save_epoch_freq == 0 or epoch % opt.save_val_freq == 0 or epoch == (opt.n_epochs + opt.n_epochs_decay):              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            # model.save_networks(epoch)

        # if opt.model == 'progan' and epoch == opt.stage_1_epochs:
        #     print('saving the model at the end of stage one - epoch %d, iters %d' % (epoch, total_iters))
        #     model.save_networks('stage_one')

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))

       
        ##### VALIDATION IMAGES ########
        if epoch % opt_val.save_val_freq == 0: #or epoch == (opt.n_epochs + opt.n_epochs_decay):
            val_model.load_networks('latest')  ##works

            # create a website
            web_dir = os.path.join(opt.checkpoints_dir, opt.name, opt_val.phase)  # define the website directory
            webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt_val.phase, opt.epoch))

            for i, data in enumerate(validation_dataset):
                val_model.set_input(data, False)  # unpack data from data loader
                val_model.test()           # run inference
                visuals = val_model.get_current_visuals()  # get image results
                img_path = val_model.get_image_paths()     # get image paths
                save_images(webpage, visuals, img_path, use_label_dirs=True, create_label_dirs=(epoch==opt_val.save_val_freq and i == 0))
            webpage.save()  # save the HTML

            try:
                metrics_dict = torch_fidelity.calculate_metrics(
                    input1 = os.path.join(web_dir, 'images', 'fake_B'),
                    input2 = os.path.join(web_dir, 'images', 'real_B'),
                    cuda = opt.gpu_ids,
                    isc = epoch == (opt.n_epochs + opt.n_epochs_decay),
                    fid = True,
                    verbose = False,
                )
            except:
                metrics_dict = {'frechet_inception_distance': 0}
                print('unable to compute validation set fid score')
            # print('metrics dict', metrics_dict)
            print_fidelity_metrics(epoch, total_iters, metrics_dict, opt_val)

            val_fid = metrics_dict['frechet_inception_distance']
            if val_fid > 0 and val_fid < best_val_fid:
                best_val_fid = val_fid
                print('saving new best model at the end of epoch %d, iters %d' % (epoch, total_iters))
                model.save_networks('best')

                if opt.model == 'progan' and epoch <= opt.stage_1_epochs:
                    model.save_networks('stage_one_best')


        ##### VALIDATION VISUALIZATION IMAGES ########
        if epoch % 1 == 0:   # visualize validation images
            # print()
            # print()
            # print('------ VAL DATASET ----')
            val_img_paths = []
            model.freeze_running_stats()
            for i, data in enumerate(validation_viz_dataset):  #### CHANGE - set to model.eval for validation images???
                
                with torch.no_grad():
                    # model.eval()
                    model.set_input(data, False)
                    model.forward()

                    if opt.model!='pix2pix':
                        model.backward_D_image()
                        model.backward_D_person()

                        losses = model.get_current_losses()
                        losses.pop('G_image')
                        losses.pop('G_person')
                        losses.pop('G_L1')
                        # print('val losses', losses)
                        accuracies = model.get_current_accuracies() #### ADDED
                        # # Plot Losses & Accuracies
                        if opt.display_id > 0:
                            validation_visualizer.plot_current_losses(epoch, i/len(validation_viz_dataset), losses)
                            validation_visualizer.plot_current_accuracies(epoch, i/len(validation_viz_dataset), accuracies) #### ADDED
                        # Print Losses & Accuracies
                        losses.update(accuracies)  #### ADDED
                        validation_visualizer.print_current_losses(epoch, total_iters, losses, lr, i)   #### ADDED lr, i  #### CHANGED epoch_images(epoch_iter) to total_iters

                    # model.compute_visuals()
                    save_val_result = True
                    val_img_paths.extend(model.get_image_paths())
                    validation_visualizer.display_current_results(model.get_current_visuals(), model.get_image_paths(), val_img_paths, epoch, save_val_result)
                    # validation_visualizer.display_current_results(model.get_current_visuals(), model.get_image_paths(), epoch, save_val_result)
                    # model.train()
            model.unfreeze_running_stats()

        print('End of Validation %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))

        lr = model.update_learning_rate()    # update learning rates at the end of every epoch.   #### ADDED lr return var

