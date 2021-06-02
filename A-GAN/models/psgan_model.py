"""
PS-GAN version uses ImagePool
"""
import torch
from .base_model import BaseModel
from . import networks
from util.image_pool import ImagePool


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
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  
            # define image discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD_image = networks.define_image_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD_image, 
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

            # define person discriminator
            self.netD_person = networks.define_person_D(opt.input_nc, opt.ndf, opt.netD_person, 
                                          opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)


            # define loss functions
            self.criterionGAN_image = networks.GANLoss(opt.gan_mode_image, self.device, label_noise=opt.disc_label_noise).to(self.device)
            self.criterionGAN_person = networks.GANLoss(opt.gan_mode_person, self.device, label_noise=opt.disc_label_noise).to(self.device)
            self.criterionL1 = torch.nn.L1Loss(reduction='mean')

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
            self.person_disc_steps = opt.person_disc_steps


    def set_input(self, input):
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

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        ## Add mask to noisy image before sending to generator                #### ADDED MASK
        img_shape = self.real_A.shape
        mask = torch.ones((img_shape[0], 1, img_shape[2], img_shape[3])).to(self.device)
        for i in range(img_shape[0]):
            mask[i, :, self.bbox[1][i]:self.bbox[3][i], self.bbox[0][i]:self.bbox[2][i]] = -1
        masked_real_A = torch.cat((self.real_A, mask), 1)

        self.fake_B = self.netG(masked_real_A)  # G(A) ## Masked Version
        # self.fake_B = self.netG(self.real_A)  # G(A)  ## Original Non-Masked Version


        ## ORIGINAL VERSION -- Only works with batch size = 1
        if self.batch_size == 1:
            x1,y1,x2,y2 = self.bbox
            self.person_crop_real = self.real_B[:,:,y1[0]:y2[0],x1[0]:x2[0]]   #### !!!! Only takes first values in bbox list, so can only use with batch size = 1
            self.person_crop_fake = self.fake_B[:,:,y1[0]:y2[0],x1[0]:x2[0]]

            self.fake_B_display = self.real_B.clone().detach()      #### ADDED fake_B_display
            self.fake_B_display[:,:,y1[0]:y2[0],x1[0]:x2[0]] = self.person_crop_fake

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


    def backward_D_image(self, train=True):
        """Calculate GAN loss for the image discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        # we use conditional GANs; we need to feed both input and output to the discriminator
        #### ADDED ImagePool used in psgan code, not in pix2pix code
        if self.use_fake_B_display:
            fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A, self.fake_B_display), 1))
            # fake_AB = torch.cat((self.real_A, self.fake_B_display), 1)
        else:
            fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A, self.fake_B), 1))
            # fake_AB = torch.cat((self.real_A, self.fake_B), 1)

        pred_image_fake = self.netD_image(fake_AB.detach())
        self.loss_D_image_fake = self.criterionGAN_image(pred_image_fake, False)  #MSELoss
        self.acc_D_image_fake = networks.calc_accuracy(pred_image_fake.detach(), False, self.device)

        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_image_real = self.netD_image(real_AB)
        self.loss_D_image_real = self.criterionGAN_image(pred_image_real, True)
        self.acc_D_image_real = networks.calc_accuracy(pred_image_real.detach(), True, self.device)

        if train:
            # combine loss and calculate gradients
            self.loss_D_image = (self.loss_D_image_fake + self.loss_D_image_real) * 0.5
            self.loss_D_image.backward()


    def backward_D_person(self, train=True):
        """Calculate GAN loss for the person discriminator"""
        ## Fake; stop backprop to the generator by detaching person_crop_fake
        if self.batch_size == 1:                           #### BATCH SIZE 1 VERSION
            fake_person_crop = self.fake_person_pool.query(self.person_crop_fake)   #### ADDED ImagePool
            pred_person_fake = self.netD_person(fake_person_crop.detach())
            # pred_person_fake = self.netD_person(self.person_crop_fake.detach())   #### ORIGINAL

            self.loss_D_person_fake = self.criterionGAN_person(pred_person_fake, False)
            self.acc_D_person_fake = networks.calc_accuracy(pred_person_fake.detach(), False, self.device)
            # print('self.loss_D_person_fake', self.loss_D_person_fake)

        elif self.use_padding:                             #### PADDED BATCH VERSION
            # print()
            # print('use padding fake')
            # fake_person_crop = self.fake_person_pool.query(self.person_crop_fake)  ## ImagePool
            # pred_person_fake = self.netD_person(fake_person_crop.detach())
            pred_person_fake = self.netD_person(self.person_crop_fake.detach(), self.bbox)      ## No ImagePool (for padding)
            self.loss_D_person_fake = self.criterionGAN_person(pred_person_fake, False)
            self.acc_D_person_fake = networks.calc_accuracy(pred_person_fake.detach(), False, self.device)
            # print('self.loss_D_person_fake', self.loss_D_person_fake)

        else:                                               #### ITERATIVE BATCH VERSION
            loss_D_person_fake_batch = []
            acc_D_person_fake_batch = []
            for fake_img in self.person_crop_fake_batch:
                fake_person_crop = self.fake_person_pool.query(fake_img)  ## ImagePool Version
                pred_person_fake = self.netD_person(fake_person_crop.detach())
                loss_D_person_fake = self.criterionGAN_person(pred_person_fake, False)
                acc_D_person_fake = networks.calc_accuracy(pred_person_fake.detach(), False, self.device)
                loss_D_person_fake_batch.append(loss_D_person_fake)
                acc_D_person_fake_batch.append(acc_D_person_fake)
            self.loss_D_person_fake = sum(loss_D_person_fake_batch)/len(loss_D_person_fake_batch)
            self.acc_D_person_fake = sum(acc_D_person_fake_batch)/len(acc_D_person_fake_batch)
            # print('self.loss_D_person_fake', self.loss_D_person_fake)
            # raise

        ## Real
        if self.batch_size == 1:                #### BATCH SIZE 1 VERSION
            pred_person_real = self.netD_person(self.person_crop_real.detach())  ##### CHECK -- should be torch.Size([1, 21])
            self.loss_D_person_real = self.criterionGAN_person(pred_person_real, True)
            self.acc_D_person_real = networks.calc_accuracy(pred_person_real.detach(), True, self.device)

        elif self.use_padding:                  #### PADDED BATCH VERSION
            # print()
            # print('use padding real')
            # print('self.person_crop_real', self.person_crop_real.size())
            pred_person_real = self.netD_person(self.person_crop_real.detach(), self.bbox)  ##### CHECK -- should be torch.Size([1, 21])
            # print('pred_person_real', pred_person_real.size())  
            self.loss_D_person_real = self.criterionGAN_person(pred_person_real, True)
            self.acc_D_person_real = networks.calc_accuracy(pred_person_real.detach(), True, self.device)

        else:                                 #### ITERATIVE BATCH VERSION
            loss_D_person_real_batch = []
            acc_D_person_real_batch = []
            for real_img in self.person_crop_real_batch:
                pred_person_real = self.netD_person(real_img.detach())
                loss_D_person_real = self.criterionGAN_person(pred_person_real, True)
                acc_D_person_real = networks.calc_accuracy(pred_person_real.detach(), True, self.device)
                loss_D_person_real_batch.append(loss_D_person_real)
                acc_D_person_real_batch.append(acc_D_person_real)
            self.loss_D_person_real = sum(loss_D_person_real_batch)/len(loss_D_person_real_batch)
            self.acc_D_person_real = sum(acc_D_person_real_batch)/len(acc_D_person_real_batch)
            # print('self.loss_D_person_real', self.loss_D_person_real)

        if train:
            ## combine loss and calculate gradients
            self.loss_D_person = (self.loss_D_person_fake + self.loss_D_person_real) * 0.5
            self.loss_D_person.backward()


    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        ## G(A) should fake the image discriminator
        if self.use_fake_B_display:
            fake_AB = torch.cat((self.real_A, self.fake_B_display), 1)
        else:
            fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake_image = self.netD_image(fake_AB)
        self.loss_G_image = self.criterionGAN_image(pred_fake_image, True)

        ## G(A) should fake the person discriminator
        if self.batch_size == 1:                           #### BATCH SIZE 1 VERSION
            pred_person_fake = self.netD_person(self.person_crop_fake)
            self.loss_G_person = self.criterionGAN_person(pred_person_fake, True)

        elif self.use_padding:                             #### PADDED BATCH  VERSION
            pred_person_fake = self.netD_person(self.person_crop_fake, self.bbox)
            self.loss_G_person = self.criterionGAN_person(pred_person_fake, True)

        else:                                              #### ITERATIVE BATCH VERSION
            loss_G_person_fake_batch = []
            for fake_img in self.person_crop_fake_batch:
                pred_person_fake = self.netD_person(fake_img)
                loss_G_person_fake = self.criterionGAN_person(pred_person_fake, True)
                loss_G_person_fake_batch.append(loss_G_person_fake)
            self.loss_G_person = sum(loss_G_person_fake_batch)/len(loss_G_person_fake_batch)
        # print('self.loss_G_person', self.loss_G_person)

        # G(A) = B
        if self.use_fake_B_display:
            self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
            # self.loss_G_L1 = self.criterionL1(self.fake_B_display, self.real_B) * self.opt.lambda_L1
        else:
            self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_image + self.loss_G_person + self.loss_G_L1
        self.loss_G.backward()


    def optimize_parameters(self, total_iters):  #### CHANGE -- added total_iters

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