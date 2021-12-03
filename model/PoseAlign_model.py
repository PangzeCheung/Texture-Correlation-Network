import torch
from model.base_model import BaseModel
from model.networks import base_function, external_function
import model.networks as network
from util import task, util
import itertools
import data as Dataset
import numpy as np
from itertools import islice
import random
import os
from model.networks.generator import PGPAN


class PoseAlign(BaseModel):
    def name(self):
        return "Pre-train flow estimator for human pose image generation"

    @staticmethod
    def modify_options(parser, is_train=True):
        """Add new options and rewrite default values for existing options"""
        parser.add_argument('--netG', type=str, default='poseshapenet', help='The name of net Generator')
        parser.add_argument('--init_type', type=str, default='orthogonal', help='Initial type')
        parser.add_argument('--netD', type=str, default='res', help='The name of net Discriminator')
        parser.add_argument('--use_spect_g', action='store_false')
        parser.add_argument('--use_spect_d', action='store_false')
        parser.add_argument('--encoder_layer', type=int, default=3, help='number of layers in G')

        # if is_train:
        parser.add_argument('--ratio_g2d', type=float, default=0.1, help='learning rate ratio G to D')
        parser.add_argument('--lambda_rec', type=float, default=5.0, help='weight for image reconstruction loss')
        parser.add_argument('--lambda_g', type=float, default=1.0, help='weight for generation loss')
        parser.add_argument('--discriminator_use_pair', action='store_true', default=False)


        parser.set_defaults(use_spect_g=False)
        parser.set_defaults(use_spect_d=True)
        return parser


    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names = ['app_gen']
        self.model_names = ['G1']
        self.visual_names = ['input_P1','input_P2', 'img_gen','input_BP1', 'input_BP2']

        self.discriminator_use_pair = opt.discriminator_use_pair
        self.FloatTensor = torch.cuda.FloatTensor if len(self.gpu_ids)>0 \
            else torch.FloatTensor
        self.ByteTensor = torch.cuda.ByteTensor if len(self.gpu_ids)>0 \
            else torch.ByteTensor

        self.net_G1 = PGPAN(image_nc=opt.image_nc, structure_nc=opt.structure_nc, ngf=64, img_f=512,
                                      encoder_layer=3, norm='instance', activation='LeakyReLU',
                                       use_spect=opt.use_spect_g)

        self.net_G1 = torch.nn.DataParallel(self.net_G1, device_ids=self.gpu_ids)


        if self.isTrain:
            # define the loss functions
            self.L1loss = torch.nn.L1Loss()
            # define the optimizer
            self.optimizer_G = torch.optim.Adam(itertools.chain(
                filter(lambda p: p.requires_grad, self.net_G1.parameters())),
                lr=opt.lr, betas=(0.0, 0.999))
            self.optimizers.append(self.optimizer_G)
        # load the pretrained model and schedulers
        self.setup(opt)


    def set_input(self, input):
        # move to GPU and change data types
        self.input = input
        input_P1, input_BP1 = input['P1'], input['BP1']
        input_P2, input_BP2 = input['P2'], input['BP2']

        if len(self.gpu_ids) > 0:
            #self.input_P1 = input_P1.cuda(self.gpu_ids[0], async=True)
            self.input_P1 = input_P1.cuda()
            self.input_BP1 = input_BP1.cuda()
            self.input_P2 = input_P2.cuda()
            self.input_BP2 = input_BP2.cuda()

        self.image_paths=[]
        for i in range(self.input_P1.size(0)):
            self.image_paths.append(os.path.splitext(input['P1_path'][i])[0] + '_2_' + input['P2_path'][i])


    def forward(self):
        """Run forward processing to get the inputs"""
        self.img_gen = self.net_G1(self.input_P1, self.input_BP1, self.input_BP2)

    def test(self):
        """Forward function used in test time"""
        img_gen1 = self.net_G1(self.input_P1, self.input_BP1, self.input_BP2)


        self.save_results(img_gen1, data_name='vis')


    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator"""
        # Real
        D_real = netD(real)
        D_real_loss = self.GANloss(D_real, True, True)
        # fake
        D_fake = netD(fake.detach())
        D_fake_loss = self.GANloss(D_fake, False, True)
        # loss for discriminator
        D_loss = (D_real_loss + D_fake_loss) * 0.5
        # gradient penalty for wgan-gp
        if self.opt.gan_mode == 'wgangp':
            gradient_penalty, gradients = external_function.cal_gradient_penalty(netD, real, fake.detach())
            D_loss += gradient_penalty

        D_loss.backward()

        return D_loss


    def backward_D(self):
        """Calculate the GAN loss for the discriminators"""
        base_function._unfreeze(self.net_D)
        if self.discriminator_use_pair:
            self.loss_dis_img_gen = self.backward_D_basic(self.net_D, torch.cat((self.input_P2, self.input_P1),1), torch.cat((self.img_gen, self.input_P1),1))
        else:
            self.loss_dis_img_gen = self.backward_D_basic(self.net_D, self.input_P2, self.img_gen)


    def backward_G(self):
        loss_app_gen = self.L1loss(self.img_gen, self.input_P2)
        self.loss_app_gen = loss_app_gen * self.opt.lambda_rec
        total_loss = 0

        for name in self.loss_names:
            if name != 'dis_img_rec' and name != 'dis_img_gen':
                total_loss += getattr(self, "loss_" + name)
        total_loss.backward()


    def optimize_parameters(self):
        """update netowrk weights"""
        self.forward()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
