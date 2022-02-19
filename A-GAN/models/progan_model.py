"""

"""
import torch
from .base_model import BaseModel
from . import networks
from util.image_pool import ImagePool
import torchvision.transforms as transforms

class PROGANModel(BaseModel):
    """ This class implements the progressive resolution two-stage model for anonymizing people, which is based on the PS-GAN model for person synthesis and the pix2pix model for image translation.

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    pix2pix code: https://phillipi.github.io/pix2pix/
    PS-GAN paper: https://arxiv.org/abs/1804.02047
    PS-GAN code: https://github.com/yueruchen/Pedestrian-Synthesis-GAN
    
    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD_image basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).
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
        # changing the default values for progressive two-stage model
        parser.set_defaults(netG='unet_128', netG2='unet_256', netD_image='n_layers', n_layers_D=2, netD_person='spp_128', netG_noise='decoder')

        if is_train:
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--L1_mask', action='store_true', help='remove bbox region with mask from L1 loss')
            parser.add_argument('--stage_1_epochs', type=int, default=100, help='number of epochs to train stage 1 network')
            parser.add_argument('--lambda_person', type=float, default=1.0, help='weight for generator person loss')
            parser.add_argument('--lambda_image', type=float, default=1.0, help='weight for generator image loss')
            parser.add_argument('--lambda_L1_stage_2', type=float, default=100.0, help='weight for generator 2 L1 loss')
            parser.add_argument('--L1_mask_stage_2', action='store_true', help='remove bbox region with mask from L1 loss in generator 2')
            parser.add_argument('--lambda_person_stage_2', type=float, default=1.0, help='weight for generator 2 person loss')
            parser.add_argument('--lambda_image_stage_2', type=float, default=1.0, help='weight for generator 2 image loss')
            parser.add_argument('--EL_person', action='store_true', help='use Embedding Loss for person in generator update')
            parser.add_argument('--EL_image', action='store_true', help='use Embedding Loss for image in generator update')
            parser.add_argument('--gan_mode_image2', type=str, default='lsgan', help='the type of GAN objective for stage 2 (entire image). [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')  ## Changed from gan_mode
            parser.add_argument('--gan_mode_person2', type=str, default='vanilla', help='the type of GAN objective for stage 2 (person). [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.') ## ADDED
            parser.add_argument('--disc_label_noise_stage_2', type=float, default=0.0, help='noise added to stage 2 image discriminator labels')
            parser.add_argument('--pretrain_iters_stage_2', type=int, default=0, help='number of iterations to pretrain stage two discriminators')
        parser.add_argument('--fake_B_display', action='store_true', help='use display version of fake_B')
        parser.add_argument('--use_padding', action='store_true', help='pad batches of cropped people for person discriminator')
        parser.add_argument('--unet_diff_map', action='store_true', help='Generator in stage 2 uses difference map version of unet')
        parser.add_argument('--netD2_person', type=str, default='spp', help='specify stage 2 person discriminator architecture [spp | conv | gap].')
        parser.add_argument('--netD2_image', type=str, default='basic', help='specify stage 2 image discriminator architecture [basic | n_layers | pixel]. The basic model is a 70x70 PatchGAN. n_layers allows you to specify the layers in the discriminator')
        parser.add_argument('--n_layers_D2', type=int, default=3, help='only used if netD2_image==n_layers')
        parser.add_argument('--stage_2_composite_input', action='store_true', help='input to stage 2 uses composite input (real background with resized generated stage 1 bbox region instead of fully generated resized stage 1 images)')
        parser.add_argument('--netG2_mask_input', action='store_true', help='add mask to stage 2 generator as additional channel on input')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)

        #### ADDED - Specify whether to use the display version of fake_B
        self.use_fake_B_display = opt.fake_B_display

        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_image', 'G_person', 'G_L1', 'D_image_real', 'D_image_fake', 'D_person_real', 'D_person_fake']
        # specify the accuracy names to print out.
        self.acc_names = ['acc_D_image_real', 'acc_D_image_fake', 'acc_D_person_real', 'acc_D_person_fake']
        self.grad_names = ['grad_G_outer_image', 'grad_G_outer_person', 'grad_G_outer_L1', 'grad_G_outer', 'grad_G_outer_clip',
                         'grad_G_inner_image', 'grad_G_inner_person', 'grad_G_inner_L1', 'grad_G_inner', 'grad_G_inner_clip',
                         'grad_G_mid_image', 'grad_G_mid_person', 'grad_G_mid_L1', 'grad_G_mid', 'grad_G_mid_clip']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        if self.use_fake_B_display:
            self.visual_names = ['real_A', 'fake_B_display', 'real_B'] #person_crop_real                                                              ############################# CHANGE to add cropped people
        elif not self.isTrain and opt.unet_diff_map:
            self.visual_names = ['real_A', 'diff_B', 'fake_B', 'fake_B_display', 'real_B']
        else:
            self.visual_names = ['real_A', 'fake_B', 'fake_B_display', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G1', 'G2', 'D_image1', 'D_image2', 'D_person1', 'D_person2']
        else:  # during test time, only load G
            self.model_names = ['G1', 'G2']
        # define networks (both generator and discriminator)
        print('netG1')
        self.netG1 = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, opt.netG_mask_input, opt.netG_noise, True) # stage 1
        print('netG2')
        self.netG2 = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG2, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, opt.netG2_mask_input, opt.netG_noise, False, opt.unet_diff_map) # stage 2 

        if self.isTrain and opt.save_grads:
            if self.gpu_ids:
                # Stage 1
                self.netG1_outer_layer = self.netG1.module.model.model[3]
                if opt.netG == 'unet_256':
                    self.netG1_inner_layer = self.netG1.module.model.model[1].model[3].model[3].model[3].model[3].model[3].model[3].model[5]
                    self.netG1_mid_layer = self.netG1.module.model.model[1].model[3].model[3].model[3].model[7]
                elif opt.netG == 'unet_128':
                    self.netG1_inner_layer = self.netG1.module.model.model[1].model[3].model[3].model[3].model[3].model[3].model[5]
                    self.netG1_mid_layer = self.netG1.module.model.model[1].model[3].model[3].model[7]
                elif opt.netG == 'unet_64':
                    self.netG1_inner_layer = self.netG1.module.model.model[1].model[3].model[3].model[3].model[3].model[5]
                    self.netG1_mid_layer = self.netG1.module.model.model[1].model[3].model[3].model[7]
                # Stage 2
                self.netG2_outer_layer = self.netG2.module.model.model[3]
                if opt.netG2 == 'unet_256':
                    self.netG2_inner_layer = self.netG2.module.model.model[1].model[3].model[3].model[3].model[3].model[3].model[3].model[5]
                    self.netG2_mid_layer = self.netG2.module.model.model[1].model[3].model[3].model[3].model[7]
                elif opt.netG2 == 'unet_128':
                    self.netG2_inner_layer = self.netG2.module.model.model[1].model[3].model[3].model[3].model[3].model[3].model[5]
                    self.netG2_mid_layer = self.netG2.module.model.model[1].model[3].model[3].model[7]
                elif opt.netG2 == 'unet_64':
                    self.netG2_inner_layer = self.netG2.module.model.model[1].model[3].model[3].model[3].model[3].model[5]
                    self.netG2_mid_layer = self.netG2.module.model.model[1].model[3].model[3].model[7]
            else:
                # Stage 1
                self.netG1_outer_layer = self.netG1.model.model[3]
                if opt.netG == 'unet_256':
                    self.netG1_inner_layer = self.netG1.model.model[1].model[3].model[3].model[3].model[3].model[3].model[3].model[5]
                    self.netG1_mid_layer = self.netG1.model.model[1].model[3].model[3].model[3].model[7]
                elif opt.netG == 'unet_128':
                    self.netG1_inner_layer = self.netG1.model.model[1].model[3].model[3].model[3].model[3].model[3].model[5]
                    self.netG1_mid_layer = self.netG1.model.model[1].model[3].model[3].model[7]
                elif opt.netG == 'unet_64':
                    self.netG1_inner_layer = self.netG1.model.model[1].model[3].model[3].model[3].model[3].model[5]
                    self.netG1_mid_layer = self.netG1.model.model[1].model[3].model[3].model[7]
                # Stage 2
                self.netG2_outer_layer = self.netG2.model.model[3]
                if opt.netG2 == 'unet_256':
                    self.netG2_inner_layer = self.netG2.model.model[1].model[3].model[3].model[3].model[3].model[3].model[3].model[5]
                    self.netG2_mid_layer = self.netG2.model.model[1].model[3].model[3].model[3].model[7]
                elif opt.netG2 == 'unet_128':
                    self.netG2_inner_layer = self.netG2.model.model[1].model[3].model[3].model[3].model[3].model[3].model[5]
                    self.netG2_mid_layer = self.netG2.model.model[1].model[3].model[3].model[7]
                elif opt.netG2 == 'unet_64':
                    self.netG2_inner_layer = self.netG2.model.model[1].model[3].model[3].model[3].model[3].model[5]
                    self.netG2_mid_layer = self.netG2.model.model[1].model[3].model[3].model[7]

        if self.isTrain:  
            # define image discriminators; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            print('netD_image1')
            self.netD_image1 = networks.define_image_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD_image, 
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            print('netD_image2')
            self.netD_image2 = networks.define_image_D(opt.input_nc, opt.ndf, opt.netD2_image,                     ## version with one image input
                                          opt.n_layers_D2, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            # self.netD_image2 = networks.define_image_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD2_image,   ## original version with two image input
            #                               opt.n_layers_D2, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

            # define person discriminators
            print('netD_person1')
            self.netD_person1 = networks.define_person_D(opt.input_nc, opt.ndf, opt.netD_person, 
                                          opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            print('netD_person2')
            self.netD_person2 = networks.define_person_D(opt.input_nc, opt.ndf, opt.netD2_person, 
                                          opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)


            # define loss functions
            self.criterionGAN_image = networks.GANLoss(opt.gan_mode_image, self.device, label_noise=opt.disc_label_noise).to(self.device)
            self.criterionGAN_person = networks.GANLoss(opt.gan_mode_person, self.device, label_noise=opt.disc_label_noise).to(self.device)
            self.criterionGAN_image2 = networks.GANLoss(opt.gan_mode_image2, self.device, label_noise=opt.disc_label_noise_stage_2).to(self.device)
            self.criterionGAN_person2 = networks.GANLoss(opt.gan_mode_person2, self.device, label_noise=opt.disc_label_noise_stage_2).to(self.device)
            self.criterionL1 = torch.nn.L1Loss(reduction='mean')
            self.criterionL2 = torch.nn.MSELoss(reduction='mean')

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G1 = torch.optim.Adam(self.netG1.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_G2 = torch.optim.Adam(self.netG2.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_image1 = torch.optim.Adam(self.netD_image1.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_image2 = torch.optim.Adam(self.netD_image2.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_person1 = torch.optim.Adam(self.netD_person1.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_person2 = torch.optim.Adam(self.netD_person2.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G1)
            self.optimizers.append(self.optimizer_G2)
            self.optimizers.append(self.optimizer_D_image1)
            self.optimizers.append(self.optimizer_D_image2)
            self.optimizers.append(self.optimizer_D_person1)
            self.optimizers.append(self.optimizer_D_person2)

            # Image Pooling -- used in psgan code, not in pix2pix code  #### CHECK
            self.fake_AB_pool = ImagePool(opt.pool_size_image)
            self.fake_person_pool = ImagePool(opt.pool_size_person)

            # list of image discriminators that accept a single image as input
            self.single_input_netD = ['resnet18', 'densenet']

            # convert stage_1 epochs to stage_1 iters
            self.stage_1_steps = opt.stage_1_epochs * opt.dataset_size
            print('stage_1_steps',self.stage_1_steps)
        
        else: #### Test Phase
            self.stage_1 = False


    def set_input(self, input, training=True):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A_big = input['A_big' if AtoB else 'B_big'].to(self.device)  ## real image with noise bbox
        self.real_A_small = input['A_small' if AtoB else 'B_small'].to(self.device)  ## real image with noise bbox
        self.real_B_big = input['B_big' if AtoB else 'A_big'].to(self.device)  ## real image
        self.real_B_small = input['B_small' if AtoB else 'A_small'].to(self.device)  ## real image
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        self.bbox_big = input['bbox_big']  #[x,y,w,h]
        self.bbox_small = input['bbox_small']  #[x,y,w,h]

        self.training = training

    def crop_person(self):
        """Create cropped real and fake images around person using bbox coordinates"""
        img_shape = self.real_A.shape

        ## ORIGINAL VERSION -- Only works with batch size = 1
        if self.opt.batch_size == 1 or not self.training:
            x1,y1,x2,y2 = self.bbox
            self.person_crop_real = self.real_B[:,:,y1[0]:y2[0],x1[0]:x2[0]]   #### !!!! Only takes first values in bbox list, so can only use with batch size = 1
            self.person_crop_fake = self.fake_B[:,:,y1[0]:y2[0],x1[0]:x2[0]]

            self.fake_B_display = self.real_B.clone().detach()      #### ADDED fake_B_display
            self.fake_B_display[:,:,y1[0]:y2[0],x1[0]:x2[0]] = self.person_crop_fake

        ## PADDED VERSION - Add padding to person crop real/fake images
        elif self.opt.use_padding:
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
        if self.stage_1:
            # print('FORWARD PASS STAGE 1')
            #### DATA SIZE ####
            self.real_A = self.real_A_small
            self.real_B = self.real_B_small
            self.bbox = self.bbox_small
            
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

            #### FORWARD PASS THROUGH GENERATOR 1 ####
            if self.opt.netG_mask_input:
                # print('masked_real_A',masked_real_A.size())
                self.fake_B = self.netG1(masked_real_A)  # G1(A) ## Masked Version
                # print('self.fake_B.size',self.fake_B.size())
            else:
                self.fake_B = self.netG1(self.real_A)  # G1(A)  ## Original Non-Masked Version

            #### PERSON CROPPED IMAGES ####
            self.crop_person()
        
        else:
            # print('FORWARD PASS STAGE 2')
            #### DATA SIZE ####
            self.real_A = self.real_A_big
            self.real_B = self.real_B_big
            self.bbox = self.bbox_big
            
            #### MASKS ####
            ## Add mask to noisy image before sending to generator
            img_shape = self.real_A.shape
            img_shape_small = self.real_A_small.shape
            g1_mask = torch.ones((img_shape_small[0], 1, img_shape_small[2], img_shape_small[3])).to(self.device)
            g2_mask = torch.ones((img_shape[0], 1, img_shape[2], img_shape[3])).to(self.device)
            self.mask = torch.zeros((img_shape[0], 1, img_shape[2], img_shape[3])).to(self.device)
            for i in range(img_shape[0]):
                g1_mask[i, :, self.bbox_small[1][i]:self.bbox_small[3][i], self.bbox_small[0][i]:self.bbox_small[2][i]] = -1
                g2_mask[i, :, self.bbox[1][i]:self.bbox[3][i], self.bbox[0][i]:self.bbox[2][i]] = -1
                self.mask[i, :, self.bbox[1][i]:self.bbox[3][i], self.bbox[0][i]:self.bbox[2][i]] = 1
            self.inv_mask = 1 - self.mask
            masked_real_A_small = torch.cat((self.real_A_small, g1_mask), 1)
            # masked_real_A = torch.cat((self.real_A, g2_mask), 1)

            #### FORWARD PASS THROUGH GENERATOR 1 ####
            if self.opt.netG_mask_input:
                fake_B_small = self.netG1(masked_real_A_small).detach()  # G1(A) ## Masked Version
                # print('fake_B_small', fake_B_small.size())
            else:
                fake_B_small = self.netG1(self.real_A_small).detach()  # G1(A)  ## Original Non-Masked Version

            #### RESIZE fake_B_small ####
            fake_B_resized = transforms.functional.resize(fake_B_small, img_shape[-1])   ### CHECK --- CONFIRM WORKS WITH BATCH SIZE > 1 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # fake_B_resized = transforms.functional.to_tensor(transforms.functional.resize(transforms.functional.to_pil_image(fake_B_small.squeeze(),mode='RGB'), img_shape[-1])).unsqueeze(0)  #### CHANGE pytorch version & torchvision version

            #### CREATE FAKE_A  
            if self.opt.stage_2_composite_input:
                # Composite input(self.real_A == real_A_with_fake_B_person)
                for i in range(img_shape[0]):
                    self.real_A[i, :, self.bbox[1][i]:self.bbox[3][i], self.bbox[0][i]:self.bbox[2][i]] = fake_B_resized[i, :, self.bbox[1][i]:self.bbox[3][i], self.bbox[0][i]:self.bbox[2][i]]
            else:
                # Fully Generated input
                self.real_A = fake_B_resized

            #### FORWARD PASS THROUGH GENERATOR 2 ####
            if self.opt.unet_diff_map:
                if self.opt.netG2_mask_input:
                    masked_real_A = torch.cat((self.real_A, g2_mask), 1)
                    self.fake_B, self.diff_B = self.netG2(masked_real_A.detach())
                else:
                    self.fake_B, self.diff_B = self.netG2(self.real_A.detach()) 
            else:
                if self.opt.netG2_mask_input:
                    masked_real_A = torch.cat((self.real_A, g2_mask), 1)
                    self.fake_B = self.netG2(masked_real_A.detach())
                else:
                    self.fake_B = self.netG2(self.real_A.detach())

            #### PERSON CROPPED IMAGES ####
            self.crop_person()


    def backward_D_image(self):
        """Calculate GAN loss for the image discriminator"""
        #### STAGE ####
        if self.stage_1:
            netD_image = self.netD_image1
            single_image_netD = self.opt.netD_image in self.single_input_netD
            criterionGAN_image = self.criterionGAN_image
        else:
            netD_image = self.netD_image2
            single_image_netD = True #self.opt.netD2_image in self.single_input_netD
            criterionGAN_image = self.criterionGAN_image2

        #### FAKE #### 
        # stop backprop to the generator by detaching fake_B
        # we use conditional GANs; we need to feed both input and output to the discriminator
        #### ADDED ImagePool used in psgan code, not in pix2pix code
        if single_image_netD and self.training:
            fake_AB = self.fake_AB_pool.query(self.fake_B)
        elif single_image_netD and not self.training:
            fake_AB = self.fake_B
        elif not self.training:  # Do not use ImagePool for validation images
            fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        elif self.use_fake_B_display:
            fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A, self.fake_B_display), 1))
            # fake_AB = torch.cat((self.real_A, self.fake_B_display), 1)
        else:
            fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A, self.fake_B), 1))
            # fake_AB = torch.cat((self.real_A, self.fake_B), 1)

        pred_image_fake = netD_image(fake_AB.detach())
        self.loss_D_image_fake = criterionGAN_image(pred_image_fake, False)  #MSELoss
        self.acc_D_image_fake = networks.calc_accuracy(pred_image_fake, False, self.device)

        #### REAL ####
        if single_image_netD:
            real_AB = self.real_B
        else:
            real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_image_real = netD_image(real_AB.detach())
        self.loss_D_image_real = criterionGAN_image(pred_image_real, True)
        self.acc_D_image_real = networks.calc_accuracy(pred_image_real, True, self.device)

        #### LOSS ####
        if self.training:
            # combine loss and calculate gradients
            self.loss_D_image = (self.loss_D_image_fake + self.loss_D_image_real) * 0.5
            self.loss_D_image.backward()


    def backward_D_person(self):
        """Calculate GAN loss for the person discriminator"""
        #### STAGE ####
        if self.stage_1:
            netD_person = self.netD_person1
            criterionGAN_person = self.criterionGAN_person
        else:
            netD_person = self.netD_person2
            criterionGAN_person = self.criterionGAN_person2

        #### Fake ####
        # stop backprop to the generator by detaching person_crop_fake
        if not self.training:
            pred_person_fake, _ = netD_person(self.person_crop_fake.detach())   #### ORIGINAL
            self.loss_D_person_fake = criterionGAN_person(pred_person_fake, False)
            self.acc_D_person_fake = networks.calc_accuracy(pred_person_fake, False, self.device)
            # print('self.loss_D_person_fake', self.loss_D_person_fake)

        elif self.opt.batch_size == 1:                           #### BATCH SIZE 1 VERSION
            fake_person_crop = self.fake_person_pool.query(self.person_crop_fake)   #### ADDED ImagePool
            pred_person_fake, _ = netD_person(fake_person_crop.detach())
            # pred_person_fake = netD_person(self.person_crop_fake.detach())   #### ORIGINAL

            self.loss_D_person_fake = criterionGAN_person(pred_person_fake, False)
            self.acc_D_person_fake = networks.calc_accuracy(pred_person_fake, False, self.device)
            # print('self.loss_D_person_fake', self.loss_D_person_fake)

        elif self.opt.use_padding:                             #### PADDED BATCH VERSION
            # print()
            # print('use padding fake')
            pred_person_fake, _ = netD_person(self.person_crop_fake.detach(), self.bbox)      ## No ImagePool (for padding)
            self.loss_D_person_fake = criterionGAN_person(pred_person_fake, False)
            self.acc_D_person_fake = networks.calc_accuracy(pred_person_fake, False, self.device)
            # print('self.loss_D_person_fake', self.loss_D_person_fake)

        else:                                               #### ITERATIVE BATCH VERSION
            loss_D_person_fake_batch = []
            acc_D_person_fake_batch = []
            for fake_img in self.person_crop_fake_batch:
                fake_person_crop = self.fake_person_pool.query(fake_img)  ## ImagePool Version
                pred_person_fake, _ = netD_person(fake_person_crop.detach())
                loss_D_person_fake = criterionGAN_person(pred_person_fake, False)
                acc_D_person_fake = networks.calc_accuracy(pred_person_fake, False, self.device)
                loss_D_person_fake_batch.append(loss_D_person_fake)
                acc_D_person_fake_batch.append(acc_D_person_fake)
            self.loss_D_person_fake = sum(loss_D_person_fake_batch)/len(loss_D_person_fake_batch)
            self.acc_D_person_fake = sum(acc_D_person_fake_batch)/len(acc_D_person_fake_batch)
            # print('self.loss_D_person_fake', self.loss_D_person_fake)
            # raise

        #### Real ####
        if self.opt.batch_size == 1 or not self.training:                #### BATCH SIZE 1 VERSION
            pred_person_real, _ = netD_person(self.person_crop_real.detach())  ##### CHECK -- should be torch.Size([1, 21])
            self.loss_D_person_real = criterionGAN_person(pred_person_real, True)
            self.acc_D_person_real = networks.calc_accuracy(pred_person_real, True, self.device)

        elif self.opt.use_padding:                  #### PADDED BATCH VERSION
            # print()
            # print('use padding real')
            # print('self.person_crop_real', self.person_crop_real.size())
            pred_person_real, _ = netD_person(self.person_crop_real.detach(), self.bbox)  ##### CHECK -- should be torch.Size([1, 21])
            # print('pred_person_real', pred_person_real.size())  
            self.loss_D_person_real = criterionGAN_person(pred_person_real, True)
            self.acc_D_person_real = networks.calc_accuracy(pred_person_real, True, self.device)

        else:                                 #### ITERATIVE BATCH VERSION
            loss_D_person_real_batch = []
            acc_D_person_real_batch = []
            for real_img in self.person_crop_real_batch:
                pred_person_real, _ = netD_person(real_img.detach())
                loss_D_person_real = criterionGAN_person(pred_person_real, True)
                acc_D_person_real = networks.calc_accuracy(pred_person_real, True, self.device)
                loss_D_person_real_batch.append(loss_D_person_real)
                acc_D_person_real_batch.append(acc_D_person_real)
            self.loss_D_person_real = sum(loss_D_person_real_batch)/len(loss_D_person_real_batch)
            self.acc_D_person_real = sum(acc_D_person_real_batch)/len(acc_D_person_real_batch)
            # print('self.loss_D_person_real', self.loss_D_person_real)

        if self.training:
            ## combine loss and calculate gradients
            self.loss_D_person = (self.loss_D_person_fake + self.loss_D_person_real) * 0.5
            self.loss_D_person.backward()



    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        #### STAGE ####
        if self.stage_1:
            netD_image = self.netD_image1
            single_image_netD = self.opt.netD_image in self.single_input_netD
            netD_person = self.netD_person1
            criterionGAN_image = self.criterionGAN_image
            criterionGAN_person = self.criterionGAN_person
        else:
            netD_image = self.netD_image2
            single_image_netD = True ## Image Disc is unconditional in second stage
            netD_person = self.netD_person2
            criterionGAN_image = self.criterionGAN_image2
            criterionGAN_person = self.criterionGAN_person2
            # self.opt.lambda_L1 = 100
            # self.opt.L1_mask = False

        #### IMAGE BACKGROUND LOSS ####
        ## G(A) should fake the image discriminator
        if single_image_netD:
            if self.opt.stage_2_composite_input:
                fake_AB = self.fake_B * self.mask + (self.fake_B * self.inv_mask).detach() ## masked version
            else:
                fake_AB = self.fake_B
        elif self.use_fake_B_display:
            fake_AB = torch.cat((self.real_A.detach(), self.fake_B_display), 1)
        else:
            fake_AB = torch.cat((self.real_A.detach(), self.fake_B), 1)
        pred_fake_image = netD_image(fake_AB)

        if self.opt.EL_image:
            if single_image_netD:
                if self.opt.stage_2_composite_input:
                    real_AB = self.real_B * self.mask + (self.real_B * self.inv_mask).detach() ## masked version
                else:
                    real_AB = self.real_B.detach()
            else:
                real_AB = torch.cat((self.real_A.detach(), self.real_B), 1)
            pred_image_real = netD_image(real_AB)
            self.loss_G_image = self.criterionL2(pred_fake_image, pred_image_real) * self.opt.lambda_image
        else:
            self.loss_G_image = criterionGAN_image(pred_fake_image, True) * self.opt.lambda_image  #### CHANGE to add new loss var

        #### PERSON LOSS ####
        ## G(A) should fake the person discriminator
        if self.opt.batch_size == 1:                           #### BATCH SIZE 1 VERSION
            pred_person_fake, embed_person_fake = netD_person(self.person_crop_fake)
            if self.opt.EL_person:
                pred_person_real, embed_person_real = netD_person(self.person_crop_real.detach())  #.detach()
                
                self.loss_G_person = 0
                for i in range(len(embed_person_real)):
                    self.loss_G_person += self.criterionL2(embed_person_fake[i], embed_person_real[i])
                    # print('self.loss_G_person', self.loss_G_person)
                self.loss_G_person *= self.opt.lambda_person
                # print(' ----  loss_G_person ---- ', self.loss_G_person)
            else:
                self.loss_G_person = criterionGAN_person(pred_person_fake, True) * self.opt.lambda_person

        elif self.opt.use_padding:                             #### PADDED BATCH  VERSION
            pred_person_fake = netD_person(self.person_crop_fake, self.bbox)
            self.loss_G_person = criterionGAN_person(pred_person_fake, True)
            raise

        else:                                              #### ITERATIVE BATCH VERSION
            loss_G_person_fake_batch = []
            for fake_img in self.person_crop_fake_batch:
                pred_person_fake = netD_person(fake_img)
                loss_G_person_fake = criterionGAN_person(pred_person_fake, True)
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
        if self.opt.L1_mask:
            self.loss_G_L1 = self.criterionL1(self.fake_B*self.inv_mask, self.real_B*self.inv_mask) * self.opt.lambda_L1  ##ADDED MASK TO L1 LOSS
        else:
            self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1

        #### COMBINED GENERATOR LOSS ####
        # combine loss and calculate gradients
        if self.opt.save_grads and self.save_iter_data:
            if self.stage_1:
                self.netG_outer_layer = self.netG1_outer_layer
                self.netG_inner_layer = self.netG1_inner_layer
                self.netG_mid_layer = self.netG1_mid_layer
                optimizer_G = self.optimizer_G1
            else:
                self.netG_outer_layer = self.netG2_outer_layer
                self.netG_inner_layer = self.netG2_inner_layer
                self.netG_mid_layer = self.netG2_mid_layer
                optimizer_G = self.optimizer_G2
                
            # print()
            # print('G_image')
            optimizer_G.zero_grad() 
            self.loss_G_image.backward(retain_graph=True)
            self.grad_G_outer_image = torch.norm(self.netG_outer_layer.weight.grad) #torch.norm(self.netG.model.model[3].weight.grad)
            self.grad_G_inner_image = torch.norm(self.netG_inner_layer.weight.grad)
            self.grad_G_mid_image = torch.norm(self.netG_mid_layer.weight.grad)

            # print('G_person')
            optimizer_G.zero_grad() 
            self.loss_G_person.backward(retain_graph=True)
            self.grad_G_outer_person = torch.norm(self.netG_outer_layer.weight.grad)
            self.grad_G_inner_person = torch.norm(self.netG_inner_layer.weight.grad)
            self.grad_G_mid_person = torch.norm(self.netG_mid_layer.weight.grad)

            # print('G_L1')
            optimizer_G.zero_grad() 
            self.loss_G_L1.backward(retain_graph=True)
            self.grad_G_outer_L1 = torch.norm(self.netG_outer_layer.weight.grad)
            self.grad_G_inner_L1 = torch.norm(self.netG_inner_layer.weight.grad)
            self.grad_G_mid_L1 = torch.norm(self.netG_mid_layer.weight.grad)

            # print('G_sum')
            optimizer_G.zero_grad() 
            self.loss_G = self.loss_G_image + self.loss_G_person + self.loss_G_L1
            self.loss_G.backward()
            self.grad_G_outer = torch.norm(self.netG_outer_layer.weight.grad)
            self.grad_G_inner = torch.norm(self.netG_inner_layer.weight.grad)
            self.grad_G_mid = torch.norm(self.netG_mid_layer.weight.grad)

        else:
            self.loss_G = self.loss_G_image + self.loss_G_person + self.loss_G_L1
            self.loss_G.backward()



    ## PS-GAN Training Order
    def optimize_parameters(self, total_iters, save_iter_data=False):  #### CHANGE -- added total_iters
        self.save_iter_data = save_iter_data

        # if total_iters > 12000:   # Set Lambda_L1 to 0 after so many iters
        #     self.opt.lambda_L1 = 0

        #### STAGE 1 ####
        # if total_iters <= self.opt.stage_1_steps: #### CHANGE - epochs instead of steps?
        if total_iters <= self.stage_1_steps:
            if total_iters == 1:
                print('------- STAGE 1 -------')
                
            self.stage_1 = True

            # print('forward')
            self.forward()                   # compute fake images: G(A)
            # update D - Image
            # print('update D-Image')
            self.set_requires_grad(self.netD_image1, True)  # enable backprop for D - image
            self.optimizer_D_image1.zero_grad()     # set D's gradients to zero
            self.backward_D_image()                # calculate gradients for D
            torch.nn.utils.clip_grad_value_(self.netD_image1.parameters(), clip_value=self.opt.clip_value)  # clip gradients
            self.optimizer_D_image1.step()          # update D's weights

            # update D - Person
            for i in range(self.opt.person_disc_steps):
                # print('forward')
                self.forward()                   # compute fake images: G(A)
                # print('update D-Person')
                self.set_requires_grad(self.netD_person1, True)  # enable backprop for D - person
                self.optimizer_D_person1.zero_grad()     # set D's gradients to zero
                self.backward_D_person()                # calculate gradients for D
                torch.nn.utils.clip_grad_value_(self.netD_person1.parameters(), clip_value=self.opt.clip_value)  # clip gradients
                self.optimizer_D_person1.step()          # update D's weights

            # update G
            for i in range(self.opt.generator_steps):
                # forward
                # print('forward')
                self.forward()                   # compute fake images: G(A)
                # print('update G')
                self.set_requires_grad(self.netD_image1, False)  # D_image requires no gradients when optimizing G
                self.set_requires_grad(self.netD_person1, False)  # D_person requires no gradients when optimizing G
                self.optimizer_G1.zero_grad()        # set G's gradients to zero
                self.backward_G()                   # calculate graidents for G
                torch.nn.utils.clip_grad_value_(self.netG1.parameters(), clip_value=self.opt.clip_value)  # clip gradients
                self.optimizer_G1.step()             # udpate G's weights
                if self.opt.save_grads and self.save_iter_data:
                    self.grad_G_outer_clip = torch.norm(self.netG1_outer_layer.weight.grad)
                    self.grad_G_inner_clip = torch.norm(self.netG1_inner_layer.weight.grad)
                    self.grad_G_mid_clip = torch.norm(self.netG1_mid_layer.weight.grad)



        #### STAGE 2 ####
        else:
            # if total_iters == self.opt.stage_1_steps + 1:
            if total_iters == self.stage_1_steps + 1:
                print('------- STAGE 2 -------')
                self.stage_1 = False
                self.fake_AB_pool = ImagePool(self.opt.pool_size_image) # Reset ImagePool for image discriminator
                self.fake_person_pool = ImagePool(self.opt.pool_size_person) # Reset ImagePool for person discriminator
                self.set_requires_grad(self.netG1, False)  # netG1 requires no gradients when optimizing G
                self.set_requires_grad(self.netD_image1, False)  # D_image1 requires no gradients when optimizing G
                self.set_requires_grad(self.netD_person1, False)  # D_person1 requires no gradients when optimizing G
                if self.opt.unet_diff_map:
                    self.visual_names = ['real_A', 'diff_B', 'fake_B', 'fake_B_display', 'real_B']
                self.opt.lambda_L1 = self.opt.lambda_L1_stage_2
                self.opt.L1_mask = self.opt.L1_mask_stage_2
                self.opt.lambda_image = self.opt.lambda_image_stage_2
                self.opt.lambda_person = self.opt.lambda_person_stage_2

            # print('forward')
            self.forward()                   # compute fake images: G(A)
            # update D - Image
            # print('update D-Image')
            self.set_requires_grad(self.netD_image2, True)  # enable backprop for D - image
            self.optimizer_D_image2.zero_grad()     # set D's gradients to zero
            self.backward_D_image()                # calculate gradients for D
            torch.nn.utils.clip_grad_value_(self.netD_image2.parameters(), clip_value=self.opt.clip_value)  # clip gradients
            self.optimizer_D_image2.step()          # update D's weights

            # update D - Person
            for i in range(self.opt.person_disc_steps):
                # print('forward')
                self.forward()                   # compute fake images: G(A)
                # print('update D-Person')
                self.set_requires_grad(self.netD_person2, True)  # enable backprop for D - person
                self.optimizer_D_person2.zero_grad()     # set D's gradients to zero
                self.backward_D_person()                # calculate gradients for D
                torch.nn.utils.clip_grad_value_(self.netD_person2.parameters(), clip_value=self.opt.clip_value)  # clip gradients
                self.optimizer_D_person2.step()          # update D's weights
 
            # update G
            for i in range(self.opt.generator_steps):
                # forward
                # print('forward')
                self.forward()                   # compute fake images: G(A)
                # print('update G')
                self.set_requires_grad(self.netD_image2, False)  # D_image2 requires no gradients when optimizing G
                self.set_requires_grad(self.netD_person2, False)  # D_person2 requires no gradients when optimizing G
                self.optimizer_G2.zero_grad()        # set G's gradients to zero
                self.backward_G()                   # calculate gradients for G
                torch.nn.utils.clip_grad_value_(self.netG2.parameters(), clip_value=self.opt.clip_value)  # clip gradients
                if total_iters > self.stage_1_steps + self.opt.pretrain_iters_stage_2:
                    self.optimizer_G2.step()             # udpate G's weights
                # else:
                    # print('pretraining stage two discrminators - skipped G2 update')
                if self.opt.save_grads and self.save_iter_data:
                    self.grad_G_outer_clip = torch.norm(self.netG2_outer_layer.weight.grad)
                    self.grad_G_inner_clip = torch.norm(self.netG2_inner_layer.weight.grad)
                    self.grad_G_mid_clip = torch.norm(self.netG2_mid_layer.weight.grad)             
