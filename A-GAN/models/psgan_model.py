"""
PS-GAN version uses ImagePool
"""
import torch
from .base_model import BaseModel
from . import networks
from util.image_pool import ImagePool
import torchvision.transforms as transforms

class PSGANModel(BaseModel):
    """ This class implements the PS-GAN model, which is based on the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD_image basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            # parser.set_defaults(pool_size=0, gan_mode_image='lsgan', gan_mode_person='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--use_L1_mask', action='store_true', help='remove bbox region with mask from L1 loss')
            parser.add_argument('--lambda_person', type=float, default=1.0, help='weight for generator person loss')
            parser.add_argument('--lambda_image', type=float, default=1.0, help='weight for generator image loss')
            parser.add_argument('--EL_person', action='store_true', help='use Embedding Loss for person in generator update')
            parser.add_argument('--EL_image', action='store_true', help='use Embedding Loss for image in generator update')
            parser.add_argument('--save_grads', action='store_true', help='save generator gradient magnitudes for plotting')
            parser.add_argument('--use_resnet_mask', action='store_true', help='use gradient mask for resnet image discriminator')
        parser.add_argument('--fake_B_display', action='store_true', help='use display version of fake_B')
        parser.add_argument('--use_padding', action='store_true', help='pad batches of cropped people for person discriminator')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)

        #### ADDED - Specify whether to use the display version of fake_B
        self.use_fake_B_display = opt.fake_B_display
        self.use_padding = opt.use_padding
        self.batch_size = opt.batch_size

        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_image', 'G_person', 'G_L1', 'D_image_real', 'D_image_fake', 'D_person_real', 'D_person_fake']
        # specify the accuracy names to print out.
        self.acc_names = ['acc_D_image_real', 'acc_D_image_fake', 'acc_D_person_real', 'acc_D_person_fake']
        self.grad_names = ['grad_G_outer_image', 'grad_G_outer_person', 'grad_G_outer_L1', 'grad_G_outer', 'grad_G_outer_clip',
                         'grad_G_inner_image', 'grad_G_inner_person', 'grad_G_inner_L1', 'grad_G_inner', 'grad_G_inner_clip',
                         'grad_G_mid_image', 'grad_G_mid_person', 'grad_G_mid_L1', 'grad_G_mid', 'grad_G_mid_clip']
        # self.grad_names = ['grad_G_image', 'grad_G_person', 'grad_G_L1','grad_G', 'grad_G_clip']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        if self.use_fake_B_display:
            self.visual_names = ['real_A', 'fake_B_display', 'real_B'] #person_crop_real                                                              ############################# CHANGE to add cropped people
        else:
            self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D_image', 'D_person']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        print('netG')
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        
        if self.isTrain and opt.save_grads:
            if self.gpu_ids:
                self.netG_outer_layer = self.netG.module.model.model[3]
                if opt.netG == 'unet_256':
                    self.netG_inner_layer = self.netG.module.model.model[1].model[3].model[3].model[3].model[3].model[3].model[3].model[5]
                    self.netG_mid_layer = self.netG.module.model.model[1].model[3].model[3].model[3].model[7]
                elif opt.netG == 'unet_128':
                    self.netG_inner_layer = self.netG.module.model.model[1].model[3].model[3].model[3].model[3].model[3].model[5]
                    self.netG_mid_layer = self.netG.module.model.model[1].model[3].model[3].model[7]
                elif opt.netG == 'unet_64':
                        self.netG_inner_layer = self.netG.module.model.model[1].model[3].model[3].model[3].model[3].model[5]
                        self.netG_mid_layer = self.netG.module.model.model[1].model[3].model[3].model[7]
            else:
                self.netG_outer_layer = self.netG.model.model[3]
                if opt.netG == 'unet_256':
                    self.netG_inner_layer = self.netG.model.model[1].model[3].model[3].model[3].model[3].model[3].model[3].model[5]
                    self.netG_mid_layer = self.netG.model.model[1].model[3].model[3].model[3].model[7]
                elif opt.netG == 'unet_128':
                    self.netG_inner_layer = self.netG.model.model[1].model[3].model[3].model[3].model[3].model[3].model[5]
                    self.netG_mid_layer = self.netG.model.model[1].model[3].model[3].model[7]
                elif opt.netG == 'unet_64':
                    self.netG_inner_layer = self.netG.model.model[1].model[3].model[3].model[3].model[3].model[5]
                    self.netG_mid_layer = self.netG.model.model[1].model[3].model[3].model[7]
                    

        if self.isTrain:  
            # define image discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            print('netD_image')
            self.netD_image = networks.define_image_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD_image, 
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

            # define person discriminator
            print('netD_person')
            self.netD_person = networks.define_person_D(opt.input_nc, opt.ndf, opt.netD_person, 
                                          opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)


            # define loss functions
            self.criterionGAN_image = networks.GANLoss(opt.gan_mode_image, self.device, label_noise=opt.disc_label_noise).to(self.device)
            self.criterionGAN_person = networks.GANLoss(opt.gan_mode_person, self.device, label_noise=opt.disc_label_noise).to(self.device)
            self.criterionL1 = torch.nn.L1Loss(reduction='mean')
            self.criterionL2 = torch.nn.MSELoss(reduction='mean')

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_image = torch.optim.Adam(self.netD_image.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_person = torch.optim.Adam(self.netD_person.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D_image)
            self.optimizers.append(self.optimizer_D_person)

            # specify gradient clipping
            self.clip_value = opt.clip_value

            # Image Pooling -- used in psgan code, not in pix2pix code  #### CHECK
            self.fake_AB_pool = ImagePool(opt.pool_size)
            self.fake_person_pool = ImagePool(opt.pool_size)

            # specify number of generator steps per iteration
            self.generator_steps = opt.generator_steps
            # specify number of person discriminator steps per iteration
            self.person_disc_steps = opt.person_disc_steps
            # specify whether to remove bbox region from L1 Loss with mask
            self.use_L1_mask = opt.use_L1_mask

            # list of image discriminators that accept a single image as input
            self.single_input_netD = ['resnet18', 'densenet']


    def set_input(self, input, training=True):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)  ## real image with noise bbox
        self.real_B = input['B' if AtoB else 'A'].to(self.device)  ## real image
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        self.bbox = input['bbox']  #[x,y,w,h]

        self.training = training

    def crop_person(self):
        """Create cropped real and fake images around person using bbox coordinates"""
        img_shape = self.real_A.shape

        ## ORIGINAL VERSION -- Only works with batch size = 1
        if self.batch_size == 1 or not self.training:
            x1,y1,x2,y2 = self.bbox
            scale = 256 // img_shape[-1]
            
            self.person_crop_real = self.real_B[:,:,y1[0]:y2[0],x1[0]:x2[0]]   #### !!!! Only takes first values in bbox list, so can only use with batch size = 1
            self.person_crop_fake = self.fake_B[:,:,y1[0]:y2[0],x1[0]:x2[0]]

            self.fake_B_display = self.real_B.clone().detach()      #### ADDED fake_B_display
            self.fake_B_display[:,:,y1[0]:y2[0],x1[0]:x2[0]] = self.person_crop_fake

            if scale > 1 and self.opt.netD_person == 'spp':
                h_resized = (y2[0] - y1[0]) * scale
                w_resized = (x2[0] - x1[0]) * scale

                self.person_crop_real = transforms.functional.resize(self.person_crop_real, (h_resized, w_resized))   #### !!!! Only takes first values in bbox list, so can only use with batch size = 1
                self.person_crop_fake = transforms.functional.resize(self.person_crop_fake, (h_resized, w_resized))

        ## PADDED VERSION - Add padding to person crop real/fake images
        elif self.use_padding:
            x1,y1,x2,y2 = self.bbox
            max_width = torch.max(x2-x1).item()
            max_height = torch.max(y2-y1).item()
            self.fake_B_display = self.real_B.clone().detach()      #### ADDED fake_B_display
            self.person_crop_real = torch.empty(self.real_B.size()[0], self.real_B.size()[1], max_height, max_width)
            self.person_crop_fake = torch.empty(self.real_B.size()[0], self.real_B.size()[1], max_height, max_width)
            # print('self.person_crop_real.size',self.person_crop_real.size())
            # print()

            for i in range(img_shape[0]):
                person_crop_real = self.real_B[i,:,y1[i]:y2[i],x1[i]:x2[i]]
                person_crop_fake = self.fake_B[i,:,y1[i]:y2[i],x1[i]:x2[i]]
                self.fake_B_display[i,:,y1[i]:y2[i],x1[i]:x2[i]] = person_crop_fake
                # print('person_crop_real.size',person_crop_real.size())  ###

                # print('person_crop_real.size',person_crop_real.size())
                pad_height = max_height - person_crop_real.size()[1]
                pad_width = max_width - person_crop_real.size()[2]
                # print('pad_height',pad_height, ', pad_width',pad_width)
                pad_left = pad_width//2
                pad_right = pad_width//2
                pad_top = pad_height//2
                pad_bottom = pad_height//2
                if pad_height % 2 != 0:
                    pad_bottom += 1
                if pad_width % 2 != 0:
                    pad_right += 1
                # print()
                # print('pad', (pad_left, pad_right, pad_top, pad_bottom))  ###
                                
                pad = torch.nn.ZeroPad2d((pad_left, pad_right, pad_top, pad_bottom))
                # pad = torch.nn.ReplicationPad2d((pad_left, pad_right, pad_top, pad_bottom))
                
                self.person_crop_real[i,:,:,:] = pad(torch.unsqueeze(person_crop_real,0))
                self.person_crop_fake[i,:,:,:] = pad(torch.unsqueeze(person_crop_fake,0))
                # print('self.person_crop_real', self.person_crop_real[i,:,:,:])
            # print('person_crop_real result size',self.person_crop_real.size())

        ## NON-PADDED BATCH VERSION - List of person crop real/fake images
        else:
            x1,y1,x2,y2 = self.bbox
            self.fake_B_display = self.real_B.clone().detach()      #### ADDED fake_B_display
            self.person_crop_real_batch = []
            self.person_crop_fake_batch= []

            for i in range(img_shape[0]):
                person_crop_real = self.real_B[i,:,y1[i]:y2[i],x1[i]:x2[i]]
                person_crop_fake = self.fake_B[i,:,y1[i]:y2[i],x1[i]:x2[i]]
                self.fake_B_display[i,:,y1[i]:y2[i],x1[i]:x2[i]] = person_crop_fake
                self.person_crop_real_batch.append(torch.unsqueeze(person_crop_real, 0))
                self.person_crop_fake_batch.append(torch.unsqueeze(person_crop_fake, 0))


    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        #### MASKS ####
        ## Add mask to noisy image before sending to generator                #### ADDED MASK
        img_shape = self.real_A.shape
        gen_mask = torch.ones((img_shape[0], 1, img_shape[2], img_shape[3])).to(self.device)
        self.mask = torch.zeros((img_shape[0], 1, img_shape[2], img_shape[3])).to(self.device)
        for i in range(img_shape[0]):
            gen_mask[i, :, self.bbox[1][i]:self.bbox[3][i], self.bbox[0][i]:self.bbox[2][i]] = -1
            self.mask[i, :, self.bbox[1][i]:self.bbox[3][i], self.bbox[0][i]:self.bbox[2][i]] = 1
        self.inv_mask = 1 - self.mask
        masked_real_A = torch.cat((self.real_A, gen_mask), 1)

        #### FORWARD PASS THROUGH GENERATOR ####
        self.fake_B = self.netG(masked_real_A)  # G(A) ## Masked Version
        # self.fake_B = self.netG(self.real_A)  # G(A)  ## Original Non-Masked Version

        #### PERSON CROPPED IMAGES ####
        self.crop_person()
        


    # def backward_D_image(self):
    #     """Calculate GAN loss for the image discriminator"""
    #     #### FAKE #### 
    #     # stop backprop to the generator by detaching fake_B
    #     # we use conditional GANs; we need to feed both input and output to the discriminator
    #     #### ADDED ImagePool used in psgan code, not in pix2pix code
    #     if not self.training:  # Do not use ImagePool for validation images
    #         fake_AB = torch.cat((self.real_A, self.fake_B), 1)
    #     elif self.use_fake_B_display:
    #         fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A, self.fake_B_display), 1))
    #         # fake_AB = torch.cat((self.real_A, self.fake_B_display), 1)
    #     else:
    #         fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A, self.fake_B), 1))
    #         # fake_AB = torch.cat((self.real_A, self.fake_B), 1)

    #     pred_image_fake = self.netD_image(fake_AB.detach())
    #     self.loss_D_image_fake = self.criterionGAN_image(pred_image_fake, False)  #MSELoss
    #     self.acc_D_image_fake = networks.calc_accuracy(pred_image_fake, False, self.device)

    #     #### REAL ####
    #     real_AB = torch.cat((self.real_A, self.real_B), 1)
    #     pred_image_real = self.netD_image(real_AB)
    #     self.loss_D_image_real = self.criterionGAN_image(pred_image_real, True)
    #     self.acc_D_image_real = networks.calc_accuracy(pred_image_real, True, self.device)

    #     #### LOSS ####
    #     if self.training:
    #         # combine loss and calculate gradients
    #         self.loss_D_image = (self.loss_D_image_fake + self.loss_D_image_real) * 0.5
    #         self.loss_D_image.backward()

    def backward_D_image(self):
        """Calculate GAN loss for the image discriminator"""
        #### FAKE #### 
        # stop backprop to the generator by detaching fake_B
        # we use conditional GANs; we need to feed both input and output to the discriminator
        #### ADDED ImagePool used in psgan code, not in pix2pix code
        if self.opt.netD_image in self.single_input_netD and self.training: # if self.opt.netD_image == 'resnet18' and self.training:
            fake_AB = self.fake_AB_pool.query(self.fake_B)
        elif self.opt.netD_image in self.single_input_netD and not self.training:
            fake_AB = self.fake_B
        elif not self.training:  # Do not use ImagePool for validation images
            fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        elif self.use_fake_B_display:
            fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A, self.fake_B_display), 1))
            # fake_AB = torch.cat((self.real_A, self.fake_B_display), 1)
        else:
            fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A, self.fake_B), 1))
            # fake_AB = torch.cat((self.real_A, self.fake_B), 1)

        pred_image_fake = self.netD_image(fake_AB.detach())
        # print('pred_image_fake', pred_image_fake)
        self.loss_D_image_fake = self.criterionGAN_image(pred_image_fake, False)  #MSELoss
        self.acc_D_image_fake = networks.calc_accuracy(pred_image_fake, False, self.device)

        #### REAL ####
        if self.opt.netD_image in self.single_input_netD:
            real_AB = self.real_B
        else:
            real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_image_real = self.netD_image(real_AB)
        self.loss_D_image_real = self.criterionGAN_image(pred_image_real, True)
        self.acc_D_image_real = networks.calc_accuracy(pred_image_real, True, self.device)

        #### LOSS ####
        if self.training:
            # combine loss and calculate gradients
            self.loss_D_image = (self.loss_D_image_fake + self.loss_D_image_real) * 0.5
            self.loss_D_image.backward()


    def backward_D_person(self):
        """Calculate GAN loss for the person discriminator"""
        #### Fake ####
        # stop backprop to the generator by detaching person_crop_fake
        if not self.training:
            pred_person_fake, _ = self.netD_person(self.person_crop_fake.detach())   #### ORIGINAL - No ImagePool for validation images
            self.loss_D_person_fake = self.criterionGAN_person(pred_person_fake, False)
            self.acc_D_person_fake = networks.calc_accuracy(pred_person_fake, False, self.device)

        elif self.batch_size == 1:                           #### BATCH SIZE 1 VERSION
            fake_person_crop = self.fake_person_pool.query(self.person_crop_fake)   #### ADDED ImagePool
            pred_person_fake, _ = self.netD_person(fake_person_crop.detach())
            # pred_person_fake = self.netD_person(self.person_crop_fake.detach())   #### ORIGINAL

            self.loss_D_person_fake = self.criterionGAN_person(pred_person_fake, False)
            self.acc_D_person_fake = networks.calc_accuracy(pred_person_fake, False, self.device)
            # print('self.loss_D_person_fake', self.loss_D_person_fake)

        elif self.use_padding:                             #### PADDED BATCH VERSION
            # print()
            # print('use padding fake')
            # fake_person_crop = self.fake_person_pool.query(self.person_crop_fake)  ## ImagePool
            # pred_person_fake = self.netD_person(fake_person_crop.detach())
            pred_person_fake, _ = self.netD_person(self.person_crop_fake.detach(), self.bbox)      ## No ImagePool (for padding)
            self.loss_D_person_fake = self.criterionGAN_person(pred_person_fake, False)
            self.acc_D_person_fake = networks.calc_accuracy(pred_person_fake, False, self.device)
            # print('self.loss_D_person_fake', self.loss_D_person_fake)

        else:                                               #### ITERATIVE BATCH VERSION
            loss_D_person_fake_batch = []
            acc_D_person_fake_batch = []
            for fake_img in self.person_crop_fake_batch:
                fake_person_crop = self.fake_person_pool.query(fake_img)  ## ImagePool Version
                pred_person_fake, _ = self.netD_person(fake_person_crop.detach())
                loss_D_person_fake = self.criterionGAN_person(pred_person_fake, False)
                acc_D_person_fake = networks.calc_accuracy(pred_person_fake, False, self.device)
                loss_D_person_fake_batch.append(loss_D_person_fake)
                acc_D_person_fake_batch.append(acc_D_person_fake)
            self.loss_D_person_fake = sum(loss_D_person_fake_batch)/len(loss_D_person_fake_batch)
            self.acc_D_person_fake = sum(acc_D_person_fake_batch)/len(acc_D_person_fake_batch)
            # print('self.loss_D_person_fake', self.loss_D_person_fake)
            # raise

        #### Real ####
        if self.batch_size == 1 or not self.training:                #### BATCH SIZE 1 VERSION
            pred_person_real, _ = self.netD_person(self.person_crop_real.detach())  ##### CHECK -- should be torch.Size([1, 21])
            self.loss_D_person_real = self.criterionGAN_person(pred_person_real, True)
            self.acc_D_person_real = networks.calc_accuracy(pred_person_real, True, self.device)

        elif self.use_padding:                  #### PADDED BATCH VERSION
            # print()
            # print('use padding real')
            # print('self.person_crop_real', self.person_crop_real.size())
            pred_person_real, _ = self.netD_person(self.person_crop_real.detach(), self.bbox)  ##### CHECK -- should be torch.Size([1, 21])
            # print('pred_person_real', pred_person_real.size())  
            self.loss_D_person_real = self.criterionGAN_person(pred_person_real, True)
            self.acc_D_person_real = networks.calc_accuracy(pred_person_real, True, self.device)

        else:                                 #### ITERATIVE BATCH VERSION
            loss_D_person_real_batch = []
            acc_D_person_real_batch = []
            for real_img in self.person_crop_real_batch:
                pred_person_real, _ = self.netD_person(real_img.detach())
                loss_D_person_real = self.criterionGAN_person(pred_person_real, True)
                acc_D_person_real = networks.calc_accuracy(pred_person_real, True, self.device)
                loss_D_person_real_batch.append(loss_D_person_real)
                acc_D_person_real_batch.append(acc_D_person_real)
            self.loss_D_person_real = sum(loss_D_person_real_batch)/len(loss_D_person_real_batch)
            self.acc_D_person_real = sum(acc_D_person_real_batch)/len(acc_D_person_real_batch)
            # print('self.loss_D_person_real', self.loss_D_person_real)

        #### LOSS ####
        if self.training:
            ## combine loss and calculate gradients
            self.loss_D_person = (self.loss_D_person_fake + self.loss_D_person_real) * 0.5
            self.loss_D_person.backward()


    # def backward_G(self):
    #     """Calculate GAN and L1 loss for the generator"""
    #     #### IMAGE BACKGROUND LOSS ####
    #     ## G(A) should fake the image discriminator
    #     if self.use_fake_B_display:
    #         fake_AB = torch.cat((self.real_A, self.fake_B_display), 1)
    #     else:
    #         fake_AB = torch.cat((self.real_A, self.fake_B), 1)
    #     pred_fake_image = self.netD_image(fake_AB)
    #     self.loss_G_image = self.criterionGAN_image(pred_fake_image, True)

    #     #### PERSON LOSS ####
    #     ## G(A) should fake the person discriminator
    #     if self.batch_size == 1:                           #### BATCH SIZE 1 VERSION
    #         pred_person_fake = self.netD_person(self.person_crop_fake)
    #         self.loss_G_person = self.criterionGAN_person(pred_person_fake, True)

    #     elif self.use_padding:                             #### PADDED BATCH  VERSION
    #         pred_person_fake = self.netD_person(self.person_crop_fake, self.bbox)
    #         self.loss_G_person = self.criterionGAN_person(pred_person_fake, True)

    #     else:                                              #### ITERATIVE BATCH VERSION
    #         loss_G_person_fake_batch = []
    #         for fake_img in self.person_crop_fake_batch:
    #             pred_person_fake = self.netD_person(fake_img)
    #             loss_G_person_fake = self.criterionGAN_person(pred_person_fake, True)
    #             loss_G_person_fake_batch.append(loss_G_person_fake)
    #         self.loss_G_person = sum(loss_G_person_fake_batch)/len(loss_G_person_fake_batch)
    #     # print('self.loss_G_person', self.loss_G_person)

    #     #### L1 LOSS ####
    #     ## G(A) = B
    #     # if self.use_fake_B_display:
    #     #     self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
    #     #     # self.loss_G_L1 = self.criterionL1(self.fake_B_display, self.real_B) * self.opt.lambda_L1
    #     # else:
    #         # self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
    #     if self.use_L1_mask:
    #         self.loss_G_L1 = self.criterionL1(self.fake_B*self.inv_mask, self.real_B*self.inv_mask) * self.opt.lambda_L1  ##ADDED MASK TO L1 LOSS
    #     else:
    #         self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1

    #     #### COMBINED GENERATOR LOSS ####
    #     # combine loss and calculate gradients
    #     self.loss_G = self.loss_G_image + self.loss_G_person + self.loss_G_L1
    #     self.loss_G.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        #### IMAGE BACKGROUND LOSS ####
        ## G(A) should fake the image discriminator
        if self.opt.netD_image in self.single_input_netD:
            if self.opt.use_resnet_mask:
                fake_AB = self.fake_B * self.mask + (self.fake_B * self.inv_mask).detach() ## masked version
            else:
                fake_AB = self.fake_B  ## non-masked version
        elif self.use_fake_B_display:
            fake_AB = torch.cat((self.real_A, self.fake_B_display), 1)
        else:
            fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake_image = self.netD_image(fake_AB)

        if self.opt.EL_image:
            if self.opt.netD_image in self.single_input_netD:
                if self.opt.use_resnet_mask:
                    real_AB = self.real_B * self.mask + (self.real_B * self.inv_mask).detach() ## masked version
                else:
                    real_AB = self.real_B  ## non-masked version
            else:
                real_AB = torch.cat((self.real_A, self.real_B), 1)
            pred_image_real = self.netD_image(real_AB)
            self.loss_G_image = self.criterionL2(pred_fake_image, pred_image_real) * self.opt.lambda_image
        else:
            self.loss_G_image = self.criterionGAN_image(pred_fake_image, True) * self.opt.lambda_image


        #### PERSON LOSS ####
        ## G(A) should fake the person discriminator
        if self.batch_size == 1:                           #### BATCH SIZE 1 VERSION
            pred_person_fake, embed_person_fake = self.netD_person(self.person_crop_fake)
        
            if self.opt.EL_person:
                # print('EL person')
                pred_person_real, embed_person_real = self.netD_person(self.person_crop_real)

                self.loss_G_person = 0
                for i in range(len(embed_person_real)):
                    self.loss_G_person += self.criterionL2(embed_person_fake[i], embed_person_real[i])
                    # print('self.loss_G_person', self.loss_G_person)
                self.loss_G_person *= self.opt.lambda_person
                # print(' ----  loss_G_person ---- ', self.loss_G_person)

                # self.loss_G_person = (self.criterionL2(embed_person_fake[0], embed_person_real[0]) +
                #                     self.criterionL2(embed_person_fake[1], embed_person_real[1]) + 
                #                     self.criterionL2(embed_person_fake[2], embed_person_real[2]) + 
                #                     self.criterionL2(embed_person_fake[3], embed_person_real[3]) +
                #                     self.criterionL2(embed_person_fake[4], embed_person_real[4])) * self.opt.lambda_person
            else:
                self.loss_G_person = self.criterionGAN_person(pred_person_fake, True) * self.opt.lambda_person

        elif self.use_padding:                             #### PADDED BATCH  VERSION
            pred_person_fake, embed_person_fake = self.netD_person(self.person_crop_fake, self.bbox)
            self.loss_G_person = self.criterionGAN_person(pred_person_fake, True)  #### CHECK that EL code works with larger batches
            # raise

        else:                                              #### ITERATIVE BATCH VERSION
            loss_G_person_fake_batch = []
            for fake_img in self.person_crop_fake_batch:
                pred_person_fake = self.netD_person(fake_img)
                loss_G_person_fake = self.criterionGAN_person(pred_person_fake, True)
                loss_G_person_fake_batch.append(loss_G_person_fake)
            self.loss_G_person = sum(loss_G_person_fake_batch)/len(loss_G_person_fake_batch)
            raise
        # print('self.loss_G_person', self.loss_G_person)



        #### L1 LOSS ####
        ## G(A) = B
        # if self.use_fake_B_display:
        #     self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        #     # self.loss_G_L1 = self.criterionL1(self.fake_B_display, self.real_B) * self.opt.lambda_L1
        # else:
            # self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        if self.use_L1_mask:
            self.loss_G_L1 = self.criterionL1(self.fake_B*self.inv_mask, self.real_B*self.inv_mask) * self.opt.lambda_L1  ##ADDED MASK TO L1 LOSS
        else:
            self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1

        #### COMBINED GENERATOR LOSS ####
        # combine loss and calculate gradients
        # self.loss_G = self.loss_G_image + self.loss_G_person + self.loss_G_L1
        if self.opt.save_grads and self.save_iter_data:
            # print()
            # print('G_image')
            self.optimizer_G.zero_grad() 
            self.loss_G_image.backward(retain_graph=True)
            self.grad_G_outer_image = torch.norm(self.netG_outer_layer.weight.grad) #torch.norm(self.netG.model.model[3].weight.grad)
            self.grad_G_inner_image = torch.norm(self.netG_inner_layer.weight.grad)
            self.grad_G_mid_image = torch.norm(self.netG_mid_layer.weight.grad)

            # print('G_person')
            self.optimizer_G.zero_grad() 
            self.loss_G_person.backward(retain_graph=True)
            self.grad_G_outer_person = torch.norm(self.netG_outer_layer.weight.grad)
            self.grad_G_inner_person = torch.norm(self.netG_inner_layer.weight.grad)
            self.grad_G_mid_person = torch.norm(self.netG_mid_layer.weight.grad)

            # print('G_L1')
            self.optimizer_G.zero_grad() 
            self.loss_G_L1.backward(retain_graph=True)
            self.grad_G_outer_L1 = torch.norm(self.netG_outer_layer.weight.grad)
            self.grad_G_inner_L1 = torch.norm(self.netG_inner_layer.weight.grad)
            self.grad_G_mid_L1 = torch.norm(self.netG_mid_layer.weight.grad)

            # print('G_sum')
            self.optimizer_G.zero_grad() 
            self.loss_G = self.loss_G_image + self.loss_G_person + self.loss_G_L1
            self.loss_G.backward()
            self.grad_G_outer = torch.norm(self.netG_outer_layer.weight.grad)
            self.grad_G_inner = torch.norm(self.netG_inner_layer.weight.grad)
            self.grad_G_mid = torch.norm(self.netG_mid_layer.weight.grad)

        else:
            self.loss_G = self.loss_G_image + self.loss_G_person + self.loss_G_L1
            self.loss_G.backward()


    def optimize_parameters(self, total_iters, save_iter_data=False):  #### CHANGE -- added total_iters
        self.save_iter_data = save_iter_data

        # if total_iters > 12000:   # Set Lambda_L1 to 0 after so many iters
        #     self.opt.lambda_L1 = 0

        for i in range(self.generator_steps):
            # forward
            # print('forward')
            self.forward()                   # compute fake images: G(A)

            # update G
            # print('update G')
            self.set_requires_grad(self.netD_image, False)  # D_image requires no gradients when optimizing G
            self.set_requires_grad(self.netD_person, False)  # D_person requires no gradients when optimizing G
            self.optimizer_G.zero_grad()        # set G's gradients to zero
            self.backward_G()                   # calculate graidents for G
            torch.nn.utils.clip_grad_value_(self.netG.parameters(), clip_value=self.clip_value)  # clip gradients
            self.optimizer_G.step()             # udpate G's weights
            if self.opt.save_grads and save_iter_data:
                # print(self.netG.model.model[3].weight.grad.size())
                self.grad_G_outer_clip = torch.norm(self.netG_outer_layer.weight.grad)
                self.grad_G_inner_clip = torch.norm(self.netG_inner_layer.weight.grad)
                self.grad_G_mid_clip = torch.norm(self.netG_mid_layer.weight.grad)
                # print('grad_G_clipped')
                # print(self.grad_G_outer_clip)
                # print(self.grad_G_inner_clip)                


        # print('Net G -- after G')
        # for name, param in self.netG.named_parameters():
        #     if param.requires_grad:
        #         # print(torch.max(param.grad))
        #         # print(torch.min(param.grad))
        #         print(torch.mean(param.grad))

        # update D - Image
        # print('update D-Image')
        self.set_requires_grad(self.netD_image, True)  # enable backprop for D - image
        self.optimizer_D_image.zero_grad()     # set D's gradients to zero
        self.backward_D_image()                # calculate gradients for D
        torch.nn.utils.clip_grad_value_(self.netD_image.parameters(), clip_value=self.clip_value)  # clip gradients
        self.optimizer_D_image.step()          # update D's weights

        for i in range(self.person_disc_steps):
            if i > 0:
                # print('forward')
                self.forward()                   # compute fake images: G(A)
            # update D - Person
            # print('update D-Person')
            self.set_requires_grad(self.netD_person, True)  # enable backprop for D - person
            self.optimizer_D_person.zero_grad()     # set D's gradients to zero
            self.backward_D_person()                # calculate gradients for D
            torch.nn.utils.clip_grad_value_(self.netD_person.parameters(), clip_value=self.clip_value)  # clip gradients
            self.optimizer_D_person.step()          # update D's weights

        # print('Net G -- after D')
        # for name, param in self.netG.named_parameters():
        #     if param.requires_grad:
        #         # print(torch.max(param.grad))
        #         # print(torch.min(param.grad))
        #         print(torch.mean(param.grad))





        # # print(list(self.netG.named_parameters()))
        # print('Net G')
        # for name, param in self.netG.named_parameters():
        #     if param.requires_grad:
        #         print(torch.max(param.grad))
        #         print(torch.min(param.grad))
        #         print(torch.mean(param.grad))

        # print()
        # print('netD_image')
        # for name, param in self.netD_image.named_parameters():
        #     if param.requires_grad:
        #         print(torch.max(param.grad))
        #         print(torch.min(param.grad))
        #         print(torch.mean(param.grad))
        #         # print(name)
        # print()
        # print('netD_person')
        # for name, param in self.netD_person.named_parameters():
        #     if param.requires_grad:
        #         print(torch.max(param.grad))
        #         print(torch.min(param.grad))
        #         print(torch.mean(param.grad))
        #         # print(name)
        # raise