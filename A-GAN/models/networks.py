import torch
from torch._C import device
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import math


###############################################################################
# Helper Functions
###############################################################################


class Identity(nn.Module):
    def forward(self, x):
        return x


def calc_accuracy(prediction, target_is_real, device):

    pred_bool = prediction >= 0.5
    if target_is_real:
        real = torch.ones(prediction.size(), device=device)
    else:
        real = torch.zeros(prediction.size(), device=device)

    pred_correct = torch.eq(pred_bool, real)
    perc_correct = round(torch.sum(pred_correct).item() / len(torch.flatten(prediction)), 2)

    return perc_correct
        


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, net_type='', init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>
    if net_type=='unet_128_2_diff_map':                     ## ADDED Initialize outer layer weights and biases to 0 in stage 2 generator
        init.constant_(net.model.model[3].weight.data, 0.0)  
        init.constant_(net.model.model[3].bias.data, 0.0)
        # print('layer weight', net.model.model[3].weight)
        # print('layer weight', net.model.model[3].bias)
        print('-- initialized output layer to 0 for difference map unet')


def init_net(net, net_type='', init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, net_type, init_type, init_gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[], stage_1=True, diff_map=False):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a generator

    Our current implementation provides two types of generators:
        U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
        The original U-Net paper: https://arxiv.org/abs/1505.04597

        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
        Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
        We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).


    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'resnet_9blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif netG == 'unet_64':
        net = UnetGenerator(input_nc, output_nc, 6, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids, stage_1=stage_1, diff_map=diff_map)  #### ADDED for input 64x64
    elif netG == 'unet_128':
        net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids, stage_1=stage_1, diff_map=diff_map)  #### CHANGED added gpu_ids, stage_1, diff_map
    elif netG == 'unet_256':
        net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids, stage_1=stage_1, diff_map=diff_map)  #### CHANGED added gpu_ids, stage_1, diff_map
        #networks.define_G(opt.input_nc=3, opt.output_nc=3, opt.ngf=64)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    stage = '_1' if stage_1 else '_2'
    version = '_diff_map' if diff_map else ''
    return init_net(net, netG+stage+version, init_type, init_gain, gpu_ids)


def define_image_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create the image discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns image discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you can specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    elif netD == 'pixel':     # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, netD, init_type, init_gain, gpu_ids)

def define_person_D(input_nc, ndf, netD, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):   ##### CHECK: USE BATCH NORM?
    """Create the person discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: spp | conv | gap
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns person discriminator

    Our current implementation...
    The discriminator has been initialized by <init_net>.
    """

    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'spp':
        net = SPP_NET(input_nc, ndf, norm_layer=norm_layer)
    elif netD == 'spp_128':
        net = SPP_NET_128(input_nc, ndf, norm_layer=norm_layer)
    elif netD == 'conv':
        net = CONV_NET(input_nc, ndf, norm_layer=norm_layer)
    elif netD == 'gap':
        net = GAP_NET(input_nc, ndf, norm_layer=norm_layer)
    else:
        print('------------------------ Person Discriminator not defined -----------')
    return init_net(net, netD, init_type, init_gain, gpu_ids)


##############################################################################
# Classes
##############################################################################
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, device, target_real_label=1.0, target_fake_label=0.0, label_noise=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)
        self.device = device  #### ADDED
        self.label_noise = label_noise #### ADDED

    def get_target_tensor(self, prediction, target_is_real, label_noise=0.0):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """
        # print('prediction', prediction)
        # print('target_is_real', target_is_real)
        # print('label_noise', label_noise)

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        target_tensor = target_tensor.expand_as(prediction)

        if label_noise:
            # print('label_noise yes', label_noise)
            random_noise_vector = torch.FloatTensor(target_tensor.size()).uniform_(label_noise*-1, label_noise).to(self.device)
            return target_tensor.add(random_noise_vector)
        else:
            # print('label_noise no', label_noise)
            return target_tensor

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode == 'lsgan': #MSELoss
            target_tensor = self.get_target_tensor(prediction, target_is_real, self.label_noise)  #### ADDED label_noise 0.05
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'vanilla': #BCEWithLogitsLoss
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( ||gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps  #### CHECK
        return gradient_penalty, gradients
    else:
        return 0.0, None


class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[], stage_1=True, diff_map=False): #input_nc=3, output_nc=3, num_downs=8, ngf=64
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, innermost=True, addrandomnoise=True, norm_layer=norm_layer, gpu_ids=gpu_ids, stage_1=stage_1)  # add the innermost layer   #### CHECK submodule w/ sequential
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, addrandomnoise=True, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids, stage_1=stage_1)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, addrandomnoise=True, norm_layer=norm_layer, gpu_ids=gpu_ids, stage_1=stage_1)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, addrandomnoise=True, norm_layer=norm_layer, gpu_ids=gpu_ids, stage_1=stage_1)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, addrandomnoise=True, norm_layer=norm_layer, gpu_ids=gpu_ids, stage_1=stage_1)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer, gpu_ids=gpu_ids, stage_1=stage_1, diff_map=diff_map)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)

class AddNoise(nn.Module):
    def __init__(self, noise_len=10, gpu_ids=[]):
        super().__init__()
        self.noise_len = noise_len
        self.device = torch.device('cuda:{}'.format(gpu_ids[0])) if gpu_ids else torch.device('cpu')

    def forward(self, encoder_output):
        # print('encoder_output size', encoder_output.size())
        # print('encoder_output std',torch.std(encoder_output))

        shape = encoder_output.size()
        random_noise_vector = torch.normal(0, 0.8, size=(self.noise_len * shape[0],)).to(self.device)
        random_noise = random_noise_vector.repeat_interleave(shape[2] * shape[3]).resize_(shape[0], self.noise_len, shape[2], shape[3])
        # print('random_noise:',random_noise.size())
        # print()
        # print('random_noise', random_noise)
        return torch.cat((encoder_output, random_noise), 1)

class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, addrandomnoise=False, norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[], stage_1=True, diff_map=False):  #### CHECK BatchNorm2d
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        # self.innermost = innermost
        self.stage_1 = stage_1
        self.diff_map = diff_map
        if type(norm_layer) == functools.partial:             #### CHECK - What does this mean?
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        if outermost and stage_1:    #### Added to account for addition of mask
            input_nc += 1
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)


        ## Size at each layer ##
        # size x torch.Size([1, 3, 256, 256])  ### OR [1, 4, 256, 256] with mask
        # size x torch.Size([1, 64, 128, 128])
        # size x torch.Size([1, 128, 64, 64])
        # size x torch.Size([1, 256, 32, 32])
        # size x torch.Size([1, 512, 16, 16])
        # size x torch.Size([1, 512, 8, 8])
        # size x torch.Size([1, 512, 4, 4])
        # size x torch.Size([1, 512, 2, 2])
        # size x torch.Size([1, 512, 1, 1])
        # encoder_output size torch.Size([1, 512, 1, 1])
        # encoder_output size torch.Size([1, 1024, 2, 2])
        # encoder_output size torch.Size([1, 1024, 4, 4])
        # encoder_output size torch.Size([1, 1024, 8, 8])

        if outermost: #and stage_1:
            if not addrandomnoise:
                ## NO ADDED NOISE
                upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,   ### INIT weights & bias to 0 for diff map version
                                            kernel_size=4, stride=2,
                                            padding=1)
                down = [downconv]
                up = [uprelu, upconv, nn.Tanh()]
                model = down + [submodule] + up

            else:
                ## ADDED NOISE AFTER RELU
                # upconv = nn.ConvTranspose2d((inner_nc * 2)+noise_len, outer_nc,
                #                             kernel_size=4, stride=2,
                #                             padding=1)
                # down = [downconv]
                # up = [uprelu, addnoise, upconv, nn.Tanh()]
                # model = down + [submodule] + up
                noise_len = 128 #256 #128
                addnoise = AddNoise(noise_len, gpu_ids=gpu_ids)

                upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                            kernel_size=4, stride=2,
                                            padding=1)
                noiseconv = nn.ConvTranspose2d((inner_nc * 2)+noise_len, (inner_nc * 2),
                                            kernel_size=1, stride=1,
                                            padding=0, bias=use_bias)
                down = [downconv]
                up = [uprelu, upconv, nn.Tanh()]
                noise = [addnoise, noiseconv]
                model = down + [submodule] + noise + up

        # elif outermost and not stage_1:
        #     if not addrandomnoise:
        #         ## NO ADDED NOISE
        #         inputconv = nn.Conv2d((input_nc * 2), input_nc,
        #                                     kernel_size=1, stride=1,
        #                                     padding=0, bias=use_bias)

        #         # downconv = nn.Conv2d((input_nc * 2), inner_nc, kernel_size=4,
        #         #              stride=2, padding=1, bias=use_bias)

        #         upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
        #                                     kernel_size=4, stride=2,
        #                                     padding=1)
        #         input = [inputconv]
        #         down = [downconv]
        #         up = [uprelu, upconv, nn.Tanh()]
        #         model = down + [submodule] + up
        #         model = input + down + [submodule] + up
        #     else:
        #         ## ADDED NOISE AFTER RELU
        #         # upconv = nn.ConvTranspose2d((inner_nc * 2)+noise_len, outer_nc,
        #         #                             kernel_size=4, stride=2,
        #         #                             padding=1)
        #         # down = [downconv]
        #         # up = [uprelu, addnoise, upconv, nn.Tanh()]
        #         # model = down + [submodule] + up

        #         noise_len = 128 #256 #128
        #         addnoise = AddNoise(noise_len, gpu_ids=gpu_ids)

        #         inputconv = nn.Conv2d((input_nc * 2), input_nc,
        #                                     kernel_size=1, stride=1,
        #                                     padding=0, bias=use_bias)
        #         # downconv = nn.Conv2d((input_nc * 2), inner_nc, kernel_size=4,
        #         #              stride=2, padding=1, bias=use_bias)

        #         upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
        #                                     kernel_size=4, stride=2,
        #                                     padding=1)
        #         noiseconv = nn.ConvTranspose2d((inner_nc * 2)+noise_len, (inner_nc * 2),
        #                                     kernel_size=1, stride=1,
        #                                     padding=0, bias=use_bias)
        #         input = [inputconv]
        #         down = [downconv]
        #         up = [uprelu, upconv, nn.Tanh()]
        #         noise = [addnoise, noiseconv]
        #         # model = down + [submodule] + noise + up
        #         model = input + down + [submodule] + noise + up
                

        elif innermost:
            if not addrandomnoise:
                ## NO ADDED NOISE
                upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                            kernel_size=4, stride=2,
                                            padding=1, bias=use_bias)
                down = [downrelu, downconv]
                up = [uprelu, upconv, upnorm]
                model = down + up

            else:
                ## ADDED NOISE AFTER RELU
                # upconv = nn.ConvTranspose2d(inner_nc+noise_len, outer_nc,
                #                             kernel_size=4, stride=2,
                #                             padding=1, bias=use_bias)
                # down = [downrelu, downconv]
                # up = [uprelu, addnoise, upconv, upnorm]
                # model = down + up  

                ## ADDED NOISE WITH CONV
                noise_len = 256 #128
                addnoise = AddNoise(noise_len, gpu_ids=gpu_ids)

                upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                            kernel_size=4, stride=2,
                                            padding=1, bias=use_bias)
                noiseconv = nn.ConvTranspose2d(inner_nc+noise_len, inner_nc,
                                            kernel_size=1, stride=1,
                                            padding=0, bias=use_bias)
                down = [downrelu, downconv]
                up = [uprelu, upconv, upnorm]
                noise = [addnoise, noiseconv]
                model = down + noise + up

        else:
            if not addrandomnoise:
                # NO ADDED NOISE
                upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                            kernel_size=4, stride=2,
                                            padding=1, bias=use_bias)
                down = [downrelu, downconv, downnorm]
                up = [uprelu, upconv, upnorm]
                if use_dropout:
                    model = down + [submodule] + up + [nn.Dropout(0.5)]
                else:
                    model = down + [submodule] + up

            else:
                # ADDED NOISE AFTER RELU
                # upconv = nn.ConvTranspose2d((inner_nc * 2)+noise_len, outer_nc,
                #                             kernel_size=4, stride=2,
                #                             padding=1, bias=use_bias)
                # down = [downrelu, downconv, downnorm]
                # up = [uprelu, addnoise, upconv, upnorm]
                # if use_dropout:
                #     model = down + [submodule] + up + [nn.Dropout(0.5)]
                # else:
                #     model = down + [submodule] + up

                ## ADDED NOISE WITH CONV
                noise_len = 128 #256 #128
                addnoise = AddNoise(noise_len, gpu_ids=gpu_ids)

                upconv = nn.ConvTranspose2d((inner_nc * 2), outer_nc,
                                            kernel_size=4, stride=2,
                                            padding=1, bias=use_bias)
                noiseconv = nn.ConvTranspose2d((inner_nc * 2)+noise_len, (inner_nc * 2),
                                            kernel_size=1, stride=1,
                                            padding=0, bias=use_bias)
                down = [downrelu, downconv, downnorm]
                up = [uprelu, upconv, upnorm]
                noise = [addnoise, noiseconv]
                if use_dropout:
                    model = down + [submodule] + noise + up + [nn.Dropout(0.5)]
                else:
                    model = down + [submodule] + noise + up
    

        self.model = nn.Sequential(*model)

    def forward(self, x):
        # print('size x', x.size())
        if self.outermost:
            out = self.model(x)
            if self.diff_map:     ## DIFFERENCE MAP VERSION
                return out.add(x)
            return out
        else:   # add skip connections
            # out = self.model(x)
            # print('out size', out.size())
            # return torch.cat([x, out], 1)
            return torch.cat([x, self.model(x)], 1)


class NLayerDiscriminator(nn.Module):   #### USED FOR IMAGE DISCRIMINATOR
    """Defines a PatchGAN discriminator"""
    # input size torch.Size([1, 6, 256, 256])
    # output size torch.Size([1, 1, 30, 30])

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1   #### CHECK different padw than orig psgan code
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),  #### CHECK bias arg not in orig psgan code
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),  #### CHECK bias arg not in orig psgan code
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class PixelDiscriminator(nn.Module):    #### NOT USED
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)

#### Network for person discriminator -- without spp
class CONV_NET(nn.Module):
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        super(CONV_NET, self).__init__()

        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        
        self.conv1 = nn.Conv2d(input_nc, ndf, 4, 2, 1, bias=use_bias)
        self.LReLU1 = nn.LeakyReLU(0.2, inplace=True)
        
        self.conv2 = nn.Conv2d(ndf, ndf * 2, 4, 1, 1, bias=use_bias)
        self.BN1 = norm_layer(ndf * 2)
        self.LReLU2 = nn.LeakyReLU(0.2, inplace=True)

        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 1, 1, bias=use_bias)
        self.BN2 = norm_layer(ndf * 4)
        self.LReLU3 = nn.LeakyReLU(0.2, inplace=True)

        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, 4, 1, 1, bias=use_bias)
        self.BN3 = norm_layer(ndf * 8)
        self.LReLU4 = nn.LeakyReLU(0.2, inplace=True)

        # self.conv5 = nn.Conv2d(ndf * 8, 1, 4, 1, 1, bias=use_bias)  ## Added padding of 1
        self.conv5 = nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=use_bias)  ## Original


    def forward(self,x):
        # print('x', x.size())  #31  #### bbox must be at least 32 wide to start
        x = self.conv1(x)
        x = self.LReLU1(x)
        # print('conv1', x.size())  #15

        x = self.conv2(x)
        x = self.LReLU2(self.BN1(x))
        # print('conv2', x.size())  #14

        x = self.conv3(x)
        x = self.LReLU3(self.BN2(x))
        # print('conv3', x.size())  #13

        x = self.conv4(x)
        x = self.LReLU4(self.BN3(x))
        # print('conv4', x.size())  #12

        x = self.conv5(x)
        # print('conv5', x.size())  #9   #### must be at least 10 to avoid error

        return x


#### Network for person discriminator -- conv + gap
class GAP_NET(nn.Module):
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        super(GAP_NET, self).__init__()

        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        
        self.conv1 = nn.Conv2d(input_nc, ndf, 4, 2, 1, bias=use_bias)
        self.LReLU1 = nn.LeakyReLU(0.2, inplace=True)
        
        self.conv2 = nn.Conv2d(ndf, ndf * 2, 4, 1, 1, bias=use_bias)
        self.BN1 = norm_layer(ndf * 2)
        self.LReLU2 = nn.LeakyReLU(0.2, inplace=True)

        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 1, 1, bias=use_bias)
        self.BN2 = norm_layer(ndf * 4)
        self.LReLU3 = nn.LeakyReLU(0.2, inplace=True)

        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, 4, 1, 1, bias=use_bias)

        self.GAP = nn.AdaptiveAvgPool2d(1)
        self.GELU = nn.GELU()
        self.FC = nn.Linear(512, 1)

    def forward(self,x):
        # print('x', x.size())  #31  #### bbox must be at least 32 wide to start
        x = self.conv1(x)
        x = self.LReLU1(x)
        # print('conv1', x.size())  #15

        x = self.conv2(x)
        x = self.LReLU2(self.BN1(x))
        # print('conv2', x.size())  #14

        x = self.conv3(x)
        x = self.LReLU3(self.BN2(x))
        # print('conv3', x.size())  #13

        x = self.conv4(x)

        # global average pooling
        x = self.GAP(x)
        # print('global average pool', x.size())
        x = self.GELU(x)
        # print('gelu', x.size())
        x = torch.squeeze(x)
        # print('squeezed', x.size())
        x = self.FC(x)
        # print('fully connected', x.size())
        # print('out', x)

        return x


#### Network for person discriminator -- conv network with spp
class SPP_NET_original(nn.Module):    #### CHANGE hardcoded BatchNorm2d
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        super(SPP_NET, self).__init__()

        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.output_num = [4,2,1]

        #torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.conv1 = nn.Conv2d(input_nc, ndf, 4, 2, 1, bias=use_bias) 
        self.LReLU1 = nn.LeakyReLU(0.2, inplace=True)
        
        self.conv2 = nn.Conv2d(ndf, ndf * 2, 4, 1, 1, bias=use_bias)
        self.BN1 = norm_layer(ndf * 2)
        self.LReLU2 = nn.LeakyReLU(0.2, inplace=True)

        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 1, 1, bias=use_bias)
        self.BN2 = norm_layer(ndf * 4)
        self.LReLU3 = nn.LeakyReLU(0.2, inplace=True)

        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, 4, 1, 1, bias=use_bias)
        self.BN3 = norm_layer(ndf * 8)
        self.LReLU4 = nn.LeakyReLU(0.2, inplace=True)

        # self.conv5 = nn.Conv2d(ndf * 8, 1, 4, 1, 1, bias=use_bias)  ## Added padding of 1
        self.conv5 = nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=use_bias)  ## Original

    def spatial_pyramid_pool(self,previous_conv, num_sample, previous_conv_size, out_pool_size):
        '''
        previous_conv: a tensor vector of previous convolution layer
        num_sample: an int number of image in the batch
        previous_conv_size: an int vector [height, width] of the matrix features size of previous convolution layer
        out_pool_size: a int vector of expected output size of max pooling layer  [4, 2, 1]
    
        returns: a tensor vector with shape [1 x n] is the concentration of multi-level pooling
        '''    
        for i in range(len(out_pool_size)):
            # print()
            # print('previous_conv_size', previous_conv_size)
            # print('out_pool_size', out_pool_size)
            # print('i', i)

            h_wid = int(math.ceil(previous_conv_size[0] / out_pool_size[i]))
            w_wid = int(math.ceil(previous_conv_size[1] / out_pool_size[i]))
            # print('h_wid', h_wid)
            # print('w_wid', w_wid)

            h_pad = (h_wid*out_pool_size[i] - previous_conv_size[0] + 1)//2  ## Changed from / to //
            w_pad = (w_wid*out_pool_size[i] - previous_conv_size[1] + 1)//2  ## Changed from / to //
            # print('h_pad', h_pad)
            # print('w_pad', w_pad)
            
            maxpool = nn.MaxPool2d((h_wid, w_wid), stride=(h_wid, w_wid), padding=(h_pad, w_pad))
            x = maxpool(previous_conv)
            # print('after spp maxpool', x.size())
            
            #### CHANGED to keep batch dim
            if(i == 0):
                spp = torch.reshape(x, (x.size()[0], -1))
            else:
                x = torch.reshape(x, (x.size()[0], -1))
                spp = torch.cat((spp, x), 1)
            # print('after spp cat', spp.size())

        return spp

    def forward(self, x):
        # print()
        print('x initial', x.size())  #31  #### bbox must be at least 32 wide to start
        # print('x initial', x)
        x = self.conv1(x)
        x = self.LReLU1(x)
        print('conv1', x.size())  #15

        x = self.conv2(x)
        x = self.LReLU2(self.BN1(x))
        print('conv2', x.size())  #14

        x = self.conv3(x)
        x = self.LReLU3(self.BN2(x))
        print('conv3', x.size())  #13

        x = self.conv4(x)
        x = self.LReLU4(self.BN3(x))
        print('conv4', x.size())  #12

        x = self.conv5(x)
        print('conv5', x.size())  #9   #### must be at least 10 to avoid error
        # print('conv out', x)

        spp = self.spatial_pyramid_pool(x, 1, [int(x.size(2)),int(x.size(3))], self.output_num)
        print('spp', spp.size())

        return spp



#### Network for person discriminator -- conv network with BATCH spp
class SPP_NET(nn.Module):    #### CHANGE hardcoded BatchNorm2d
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        super(SPP_NET, self).__init__()

        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.output_num = [4,2,1]

        #torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.conv1 = nn.Conv2d(input_nc, ndf, 4, 2, 1, bias=use_bias) 
        self.LReLU1 = nn.LeakyReLU(0.2, inplace=True)
        
        self.conv2 = nn.Conv2d(ndf, ndf * 2, 4, 1, 1, bias=use_bias)
        self.BN1 = norm_layer(ndf * 2)
        self.LReLU2 = nn.LeakyReLU(0.2, inplace=True)

        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 1, 1, bias=use_bias)
        self.BN2 = norm_layer(ndf * 4)
        self.LReLU3 = nn.LeakyReLU(0.2, inplace=True)

        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, 4, 1, 1, bias=use_bias)
        self.BN3 = norm_layer(ndf * 8)
        self.LReLU4 = nn.LeakyReLU(0.2, inplace=True)

        # self.conv5 = nn.Conv2d(ndf * 8, 1, 4, 1, 1, bias=use_bias)  ## Added padding of 1
        self.conv5 = nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=use_bias)  ## Original

    def spatial_pyramid_pool(self,previous_conv, num_sample, previous_conv_size, out_pool_size):
        '''
        previous_conv: a tensor vector of previous convolution layer
        num_sample: an int number of image in the batch
        previous_conv_size: an int vector [height, width] of the matrix features size of previous convolution layer
        out_pool_size: a int vector of expected output size of max pooling layer  [4, 2, 1]
    
        returns: a tensor vector with shape [1 x n] is the concentration of multi-level pooling
        '''    
        for i in range(len(out_pool_size)):
            # print()
            # print('previous_conv_size', previous_conv_size)
            # print('out_pool_size', out_pool_size)
            # print('i', i)

            h_wid = int(math.ceil(previous_conv_size[0] / out_pool_size[i]))
            w_wid = int(math.ceil(previous_conv_size[1] / out_pool_size[i]))
            # print('h_wid', h_wid)
            # print('w_wid', w_wid)

            h_pad = (h_wid*out_pool_size[i] - previous_conv_size[0] + 1)//2  ## Changed from / to //
            w_pad = (w_wid*out_pool_size[i] - previous_conv_size[1] + 1)//2  ## Changed from / to //
            # print('h_pad', h_pad)
            # print('w_pad', w_pad)
            
            maxpool = nn.MaxPool2d((h_wid, w_wid), stride=(h_wid, w_wid), padding=(h_pad, w_pad))
            x = maxpool(previous_conv)
            # print('after spp maxpool', x.size())
            
            #### CHANGED to keep batch dim
            if(i == 0):
                spp = torch.reshape(x, (x.size()[0], -1))
            else:
                x = torch.reshape(x, (x.size()[0], -1))
                spp = torch.cat((spp, x), 1)
            # print('after spp cat', spp.size())

        return spp

    def forward(self, x, bbox=[]):

        # print()
        # print('x initial', x.size())  #31  #### bbox must be at least 32 wide to start
        # print('x initial', x)
        x = self.conv1(x)
        x = self.LReLU1(x)
        # print('conv1', x.size())  #15

        x = self.conv2(x)
        x = self.LReLU2(self.BN1(x))
        # print('conv2', x.size())  #14

        x = self.conv3(x)
        x = self.LReLU3(self.BN2(x))
        # print('conv3', x.size())  #13

        x = self.conv4(x)
        x = self.LReLU4(self.BN3(x))
        # print('conv4', x.size())  #12

        x = self.conv5(x)
        # print('conv5', x.size())  #9   #### must be at least 10 to avoid error
        # print('conv out', x)

        ## If batch size == 1, call spp directly
        if x.size()[0] == 1:
            spp = self.spatial_pyramid_pool(x, 1, [int(x.size(2)), int(x.size(3))], self.output_num)
            # print('spp', spp.size())
            return spp

        #### If batch size > 1, iterate through batch to call spp on each img  ####
        ## Padded height & width
        padded_conv_height = x.size()[2]
        padded_conv_width = x.size()[3]

        for i in range(x.size()[0]):
            x1, y1, x2, y2 = bbox

            ## Height and width of unpadded cropped person
            unpadded_height = y2[i] - y1[i] #50
            unpadded_width = x2[i] - x1[i] #36
            # print()
            # print('unpadded_height', unpadded_height)
            # print('unpadded_width', unpadded_width)

            ## Unpadded height and width after convolutions
            unpadded_conv_height = unpadded_height // 2 - 6
            unpadded_conv_width = unpadded_width // 2 - 6

            ## Amount of padding after convolutions
            conv_padding_height = padded_conv_height - unpadded_conv_height
            conv_padding_width = padded_conv_width - unpadded_conv_width

            ## Indices of unpadded x after convolutions
            yc1 = conv_padding_height // 2
            yc2 = yc1 + unpadded_conv_height
            xc1 = conv_padding_width // 2
            xc2 = xc1 + unpadded_conv_width

            x_crop = torch.unsqueeze(x[i, :, yc1:yc2, xc1:xc2], 0)
            # print('x_crop', x_crop.size())

            if i == 0:
                spp = self.spatial_pyramid_pool(x_crop, 1, [int(x_crop.size(2)), int(x_crop.size(3))], self.output_num)
                # print('spp_0', spp)
            else:
                spp_i = self.spatial_pyramid_pool(x_crop, 1, [int(x_crop.size(2)), int(x_crop.size(3))], self.output_num)
                # print('spp_', str(i), spp_i)
                spp = torch.cat((spp, spp_i), 0)
                
        # print('spp', spp.size())
        return spp


#### Network for person discriminator (128x128 images) -- conv network with BATCH spp
class SPP_NET_128(nn.Module):    #### CHANGE hardcoded BatchNorm2d
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        super(SPP_NET_128, self).__init__()

        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.output_num = [2,1] #[4,2,1]

        #torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.conv1 = nn.Conv2d(input_nc, ndf, 4, 2, 1, bias=use_bias)
        self.LReLU1 = nn.LeakyReLU(0.2, inplace=True)
        
        self.conv2 = nn.Conv2d(ndf, ndf * 2, 4, 1, 1, bias=use_bias)
        self.BN1 = norm_layer(ndf * 2)
        self.LReLU2 = nn.LeakyReLU(0.2, inplace=True)

        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 1, 1, bias=use_bias)
        self.BN2 = norm_layer(ndf * 4)
        self.LReLU3 = nn.LeakyReLU(0.2, inplace=True)

        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, 4, 1, 1, bias=use_bias)
        self.BN3 = norm_layer(ndf * 8)
        self.LReLU4 = nn.LeakyReLU(0.2, inplace=True)

        # self.conv5 = nn.Conv2d(ndf * 8, 1, 4, 1, 1, bias=use_bias)  ## Added padding of 1
        self.conv5 = nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=use_bias)  ## Original

    def spatial_pyramid_pool(self,previous_conv, num_sample, previous_conv_size, out_pool_size):
        '''
        previous_conv: a tensor vector of previous convolution layer
        num_sample: an int number of image in the batch
        previous_conv_size: an int vector [height, width] of the matrix features size of previous convolution layer
        out_pool_size: a int vector of expected output size of max pooling layer  [4, 2, 1]
    
        returns: a tensor vector with shape [1 x n] is the concentration of multi-level pooling
        '''    
        for i in range(len(out_pool_size)):
            # print()
            # print('previous_conv_size', previous_conv_size)
            # print('out_pool_size', out_pool_size)
            # print('i', i)

            h_wid = int(math.ceil(previous_conv_size[0] / out_pool_size[i]))
            w_wid = int(math.ceil(previous_conv_size[1] / out_pool_size[i]))
            # print('h_wid', h_wid)
            # print('w_wid', w_wid)

            h_pad = (h_wid*out_pool_size[i] - previous_conv_size[0] + 1)//2  ## Changed from / to //
            w_pad = (w_wid*out_pool_size[i] - previous_conv_size[1] + 1)//2  ## Changed from / to //
            # print('h_pad', h_pad)
            # print('w_pad', w_pad)
            
            maxpool = nn.MaxPool2d((h_wid, w_wid), stride=(h_wid, w_wid), padding=(h_pad, w_pad))
            x = maxpool(previous_conv)
            # print('after spp maxpool', x.size())
            
            #### CHANGED to keep batch dim
            if(i == 0):
                spp = torch.reshape(x, (x.size()[0], -1))
            else:
                x = torch.reshape(x, (x.size()[0], -1))
                spp = torch.cat((spp, x), 1)
            # print('after spp cat', spp.size())

        return spp

    def forward(self, x, bbox=[]):

        # print()
        # print('x initial', x.size())  #17  #### bbox must be at least 17 wide to start
        # print('x initial', x)
        x = self.conv1(x)
        x = self.LReLU1(x)
        # print('conv1', x.size())  #16

        x = self.conv2(x)
        x = self.LReLU2(self.BN1(x))
        # print('conv2', x.size())  #15

        x = self.conv3(x)
        x = self.LReLU3(self.BN2(x))
        # print('conv3', x.size())  #14

        x = self.conv4(x)
        x = self.LReLU4(self.BN3(x))
        # print('conv4', x.size())  #13

        x = self.conv5(x)
        # print('conv5', x.size())  #10   #### must be at least 10 to avoid error
        # print('conv out', x)

        ## If batch size == 1, call spp directly
        if x.size()[0] == 1:
            spp = self.spatial_pyramid_pool(x, 1, [int(x.size(2)), int(x.size(3))], self.output_num)
            # print('spp', spp.size())
            return spp

        #### If batch size > 1, iterate through batch to call spp on each img  ####
        ## Padded height & width
        padded_conv_height = x.size()[2]
        padded_conv_width = x.size()[3]

        for i in range(x.size()[0]):
            x1, y1, x2, y2 = bbox

            ## Height and width of unpadded cropped person
            unpadded_height = y2[i] - y1[i] #50
            unpadded_width = x2[i] - x1[i] #36
            # print()
            # print('unpadded_height', unpadded_height)
            # print('unpadded_width', unpadded_width)

            ## Unpadded height and width after convolutions
            unpadded_conv_height = unpadded_height // 2 - 6
            unpadded_conv_width = unpadded_width // 2 - 6

            ## Amount of padding after convolutions
            conv_padding_height = padded_conv_height - unpadded_conv_height
            conv_padding_width = padded_conv_width - unpadded_conv_width

            ## Indices of unpadded x after convolutions
            yc1 = conv_padding_height // 2
            yc2 = yc1 + unpadded_conv_height
            xc1 = conv_padding_width // 2
            xc2 = xc1 + unpadded_conv_width

            x_crop = torch.unsqueeze(x[i, :, yc1:yc2, xc1:xc2], 0)
            # print('x_crop', x_crop.size())

            if i == 0:
                spp = self.spatial_pyramid_pool(x_crop, 1, [int(x_crop.size(2)), int(x_crop.size(3))], self.output_num)
                # print('spp_0', spp)
            else:
                spp_i = self.spatial_pyramid_pool(x_crop, 1, [int(x_crop.size(2)), int(x_crop.size(3))], self.output_num)
                # print('spp_', str(i), spp_i)
                spp = torch.cat((spp, spp_i), 0)
                
        # print('spp', spp.size())
        return spp
