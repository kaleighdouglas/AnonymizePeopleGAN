"""
PS-GAN version uses ImagePool
"""
import torch
from .base_model import BaseModel
from . import networks
from util.image_pool import ImagePool #### CHECK: used in psgan code, not in pix2pix code


class PersonGANModel(BaseModel):
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
            # parser.set_defaults(pool_size=0, gan_mode='vanilla')
            # parser.set_defaults(pool_size=0, gan_mode_image='lsgan', gan_mode_person='vanilla')    #### CHECK pool_size default
            # parser.set_defaults(gan_mode_image='lsgan', gan_mode_person='vanilla') 
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_person', 'G_L1', 'D_person_real', 'D_person_fake']
        # specify the accuracy names to print out.
        self.acc_names = ['acc_D_person_real', 'acc_D_person_fake']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']                                                                  ############################# CHANGE to add cropped people
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D_person']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  
            # # define image discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            # self.netD_image = networks.define_image_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD_image, 
            #                               opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

            # define person discriminator
            self.netD_person = networks.define_person_D(opt.input_nc, opt.ndf, opt.netD_person, 
                                          opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)


            # define loss functions
            # self.criterionGAN_image = networks.GANLoss(opt.gan_mode_image).to(self.device)
            self.criterionGAN_person = networks.GANLoss(opt.gan_mode_person).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            # self.optimizer_D_image = torch.optim.Adam(self.netD_image.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_person = torch.optim.Adam(self.netD_person.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            # self.optimizers.append(self.optimizer_D_image)
            self.optimizers.append(self.optimizer_D_person)

            # specify gradient clipping
            self.clip_value = opt.clip_value

            # Image Pooling -- used in psgan code, not in pix2pix code  #### CHECK
            self.fake_AB_pool = ImagePool(opt.pool_size)

            # specify number of generator steps per iteration
            self.generator_steps = opt.generator_steps

        # print('---------- Networks initialized -------------')
        # networks.print_network(self.netG)
        # if self.isTrain:
        #     networks.print_network(self.netD_image)
        #     networks.print_network(self.netD_person)
        # print('-----------------------------------------------')

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        self.bbox = input['bbox']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)  # G(A)
        # print('fake B size', self.fake_B.size())
        # print(self.fake_B)
        # print()
        # raise

        x1,y1,x2,y2 = self.bbox
        self.person_crop_real = self.real_B[:,:,y1[0]:y2[0],x1[0]:x2[0]]
        self.person_crop_fake = self.fake_B[:,:,y1[0]:y2[0],x1[0]:x2[0]]


    def backward_D_person(self):
        """Calculate GAN loss for the person discriminator"""
        # Fake; stop backprop to the generator by detaching person_crop_fake
        pred_person_fake = self.netD_person(self.person_crop_fake.detach())
        self.loss_D_person_fake = self.criterionGAN_person(pred_person_fake, False)
        self.acc_D_person_fake = networks.calc_accuracy(pred_person_fake.detach(), False, self.device)

        # Real
        pred_person_real = self.netD_person(self.person_crop_real)
        self.loss_D_person_real = self.criterionGAN_person(pred_person_real, True)
        self.acc_D_person_real = networks.calc_accuracy(pred_person_real.detach(), True, self.device)

        # combine loss and calculate gradients
        self.loss_D_person = (self.loss_D_person_fake + self.loss_D_person_real) * 0.5
        self.loss_D_person.backward()

    def backward_G(self, total_iters):
        """Calculate GAN and L1 loss for the generator"""

        # G(A) should fake the person discriminator
        pred_fake_person = self.netD_person(self.person_crop_fake)
        self.loss_G_person = self.criterionGAN_person(pred_fake_person, True)

        # G(A) = B
        if total_iters > 64:  #### CHANGE - ADDED - only use L1 loss during first 64 iterations
            self.loss_G_L1 = 0.0
        else:
            self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1

        # combine loss and calculate gradients
        self.loss_G =  self.loss_G_person + self.loss_G_L1
        self.loss_G.backward()
        

    def optimize_parameters(self, total_iters):  #### CHANGE -- added total_iters

        for i in range(self.generator_steps):
            # forward
            self.forward()                   # compute fake images: G(A)

            # update G
            # self.set_requires_grad(self.netD_image, False)  # D_image requires no gradients when optimizing G
            self.set_requires_grad(self.netD_person, False)  # D_person requires no gradients when optimizing G
            self.optimizer_G.zero_grad()        # set G's gradients to zero
            self.backward_G(total_iters)                   # calculate graidents for G
            torch.nn.utils.clip_grad_value_(self.netG.parameters(), clip_value=self.clip_value)  # clip gradients
            # if total_iters > 30:  #### CHANGE - test only generator
            #     self.optimizer_G.step()             # udpate G's weights
            self.optimizer_G.step()             # udpate G's weights


        # update D - Person
        self.set_requires_grad(self.netD_person, True)  # enable backprop for D - person
        self.optimizer_D_person.zero_grad()     # set D's gradients to zero
        self.backward_D_person()                # calculate gradients for D
        torch.nn.utils.clip_grad_value_(self.netD_person.parameters(), clip_value=self.clip_value)  # clip gradients
        # if total_iters <= 30:  #### CHANGE - test only generator
        #     self.optimizer_D_person.step()          # update D's weights
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