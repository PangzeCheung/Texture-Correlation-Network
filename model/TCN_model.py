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
from model.networks.generator import PGPAN, TCAN


class TCN(BaseModel):
    def name(self):
        return "Texture Correlation Network"

    @staticmethod
    def modify_options(parser, is_train=True):
        """Add new options and rewrite default values for existing options"""
        parser.add_argument('--netG', type=str, default='poseshapenet', help='The name of net Generator')
        parser.add_argument('--init_type', type=str, default='orthogonal', help='Initial type')
        parser.add_argument('--netD', type=str, default='res', help='The name of net Discriminator')
        parser.add_argument('--use_spect_g', action='store_false')
        parser.add_argument('--use_spect_d', action='store_false')
        parser.add_argument('--encoder_layer', type=int, default=3, help='number of layers in G')
        parser.add_argument('--lambda_style', type=float, default=1000.0, help='weight for the VGG19 style loss')
        parser.add_argument('--lambda_content', type=float, default=1, help='weight for the VGG19 content loss')
        parser.add_argument('--layers', type=int, default=3, help='number of layers in G')
        parser.add_argument('--save_input', action='store_true', help="whether save the input images when testing")


        # if is_train:
        parser.add_argument('--ratio_g2d', type=float, default=0.1, help='learning rate ratio G to D')
        parser.add_argument('--lambda_rec', type=float, default=5.0, help='weight for image reconstruction loss')
        parser.add_argument('--lambda_g', type=float, default=2.0, help='weight for generation loss')
        parser.add_argument('--discriminator_use_pair', action='store_true', default=False)
        parser.add_argument('--use_d', action='store_true', default=False)
        parser.add_argument('--lambda_lg1', type=float, default=2.0, help='weight for generation loss')
        parser.add_argument('--lambda_lg2', type=float, default=5.0, help='weight for generation loss')


        parser.set_defaults(use_spect_g=False)
        parser.set_defaults(use_spect_d=True)
        return parser


    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.use_d = opt.use_d
        self.loss_names = ['app_gen1', 'app_gen2', 'ad_gen', 'dis_img_gen', 'content_gen', 'style_gen']
        self.model_names = ['G1', 'G2', 'D']
        self.visual_names = ['input_P1','input_P2', 'img_gen1','input_BP1', 'input_BP2', 'img_gen2', 'warp', 'masks', 'flows', 'uni_flow' ]

        self.discriminator_use_pair = opt.discriminator_use_pair
        self.FloatTensor = torch.cuda.FloatTensor if len(self.gpu_ids)>0 \
            else torch.FloatTensor
        self.ByteTensor = torch.cuda.ByteTensor if len(self.gpu_ids)>0 \
            else torch.ByteTensor
        self.flow2color = util.flow2color()

        self.net_G1 = PGPAN(image_nc=opt.image_nc, structure_nc=opt.structure_nc, ngf=64, img_f=512,
                                      encoder_layer=3, norm='instance', activation='LeakyReLU',
                                       use_spect=opt.use_spect_g,
                                       )
        self.net_G2 = TCAN(image_nc=opt.image_nc, structure_nc=opt.structure_nc, ngf=64, img_f=512,
                                             layers=opt.layers, num_blocks=2, use_spect=opt.use_spect_g,
                                             norm='instance', activation='LeakyReLU', dataset = self.opt.dataset_mode)
        self.net_G2.print_network()


        if self.use_d and self.discriminator_use_pair:
            self.net_D = network.define_d(opt, input_nc=6, ndf=32, img_f=128, layers=4, use_spect=opt.use_spect_d)
        else:
            if self.opt.dataset_mode == 'fashion'or self.opt.dataset_mode == 'fashionalpha':
                self.net_D = network.define_d(opt, ndf=32, img_f=128, layers=4, use_spect=opt.use_spect_d)
            elif self.opt.dataset_mode== 'market':
                self.net_D = network.define_d(opt, ndf=32, img_f=128, layers=3, use_spect=opt.use_spect_d)

        self.net_G1 = torch.nn.DataParallel(self.net_G1, device_ids=self.gpu_ids)
        self.net_G2 = torch.nn.DataParallel(self.net_G2, device_ids=self.gpu_ids)
        if self.use_d:
            self.net_D = torch.nn.DataParallel(self.net_D, device_ids=self.gpu_ids)


        if self.isTrain:
            # define the loss functions

            self.GANloss = external_function.AdversarialLoss(opt.gan_mode).to(opt.device)
            self.L1loss = torch.nn.L1Loss()
            self.Vggloss = external_function.VGGLoss().to(opt.device)
            # define the optimizer
            self.optimizer_G = torch.optim.Adam(itertools.chain(
                filter(lambda p: p.requires_grad, self.net_G1.parameters()),
                filter(lambda p: p.requires_grad, self.net_G2.parameters())),
                lr=opt.lr, betas=(0.0, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizer_D = torch.optim.Adam(itertools.chain(
                filter(lambda p: p.requires_grad, self.net_D.parameters())),
                lr=opt.lr * opt.ratio_g2d, betas=(0.0, 0.999))
            self.optimizers.append(self.optimizer_D)
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
        self.img_gen1 = self.net_G1(self.input_P1, self.input_BP1, self.input_BP2)

        self.img_gen2, self.masks, self.flows, self.grids = self.net_G2(self.input_P1, self.input_BP1, self.img_gen1,
                                                                        self.input_BP2)

        self.warp = self.visi(self.grids[-1])

        [ _, h, w] = self.flows[0][-1].size()
        x = torch.arange(w).view(1, -1).expand(h, -1).float()
        y = torch.arange(h).view(-1, 1).expand(-1, w).float()
        x = 2*x/(w-1)-1
        y = 2*y/(h-1)-1

        self.uni_flow = torch.stack([x,y], dim=0).float()

    def test(self):
        """Forward function used in test time"""

        img_gen1 = self.net_G1(self.input_P1, self.input_BP1, self.input_BP2)


        img_gen2, masks, _, _ = self.net_G2(self.input_P1, self.input_BP1, img_gen1, self.input_BP2, test = True)
        if self.opt.dataset_mode == 'fashion' or self.opt.dataset_mode == 'fashionalpha':
            self.save_results(img_gen2, data_name='vis', data='fashion')
        else:
            self.save_results(img_gen2, data_name='vis', data='market')



    def visi(self, flow_field):
        [b, _, h, w] = flow_field.size()

        source_copy = torch.nn.functional.interpolate(self.input_P1, (h, w))

        flow_x = (flow_field[:, 0, :, :]*2 / (w - 1) - 1).view(b, 1, h, w)
        flow_y = (flow_field[:, 1, :, :]*2 / (h - 1) - 1).view(b, 1, h, w)
        flow = torch.cat((flow_x, flow_y), 1)
        grid = flow.permute(0, 2, 3, 1)
        warp = torch.nn.functional.grid_sample(source_copy, grid)
        return warp


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
            self.loss_dis_img_gen = self.backward_D_basic(self.net_D, torch.cat((self.input_P2, self.input_P1),1), torch.cat((self.img_gen2, self.input_P1),1))
        else:
            self.loss_dis_img_gen = self.backward_D_basic(self.net_D, self.input_P2, self.img_gen2)


    def backward_G(self):
        loss_app_gen1 = self.L1loss(self.img_gen1, self.input_P2)
        self.loss_app_gen1 = loss_app_gen1 * self.opt.lambda_lg1
        loss_app_gen2 = self.L1loss(self.img_gen2, self.input_P2)
        self.loss_app_gen2 = loss_app_gen2 * self.opt.lambda_lg2

        # Calculate GAN loss
        base_function._freeze(self.net_D)
        D_fake = self.net_D(self.img_gen2)
        self.loss_ad_gen = self.GANloss(D_fake, True, False) * self.opt.lambda_g

        # Calculate perceptual loss
        loss_content_gen, loss_style_gen = self.Vggloss(self.img_gen2, self.input_P2)
        self.loss_style_gen = loss_style_gen * self.opt.lambda_style
        self.loss_content_gen = loss_content_gen * self.opt.lambda_content


        total_loss = self.loss_app_gen1 + self.loss_app_gen2 + self.loss_ad_gen + self.loss_style_gen + self.loss_content_gen

        total_loss.backward()


    def optimize_parameters(self):
        """update netowrk weights"""
        self.forward()

        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
