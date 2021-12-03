import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.networks.base_network import BaseNetwork
from model.networks.base_function import *
from torch.nn.utils.spectral_norm import spectral_norm as SpectralNorm


######################################################################################################
# Human Pose Image Generation 
######################################################################################################


class SourceEncoder(BaseNetwork):
    def __init__(self, input_nc=3, ngf=64, img_f=1024, layers=6, norm='batch',
                activation='ReLU', use_spect=True, use_coord=False):  
        super(SourceEncoder, self).__init__()
        self.layers = layers
        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)

        # encoder part CONV_BLOCKS
        self.block0 = EncoderBlock(input_nc, ngf, norm_layer,
                                 nonlinearity, use_spect, use_coord)
        mult = 1
        for i in range(layers-1):
            mult_prev = mult
            mult = min(2 ** (i + 1), img_f//ngf)
            block = EncoderBlock(ngf*mult_prev, ngf*mult, norm_layer,
                                 nonlinearity, use_spect, use_coord)
            setattr(self, 'encoder' + str(i), block)        


    def forward(self, source):
        feature_list=[source]
        out = self.block0(source)
        feature_list.append(out)
        for i in range(self.layers-1):
            model = getattr(self, 'encoder' + str(i))
            out = model(out) 
            feature_list.append(out)

        feature_list = list(reversed(feature_list))
        return feature_list


class PGPAN(BaseNetwork):
    def __init__(self, image_nc=3, structure_nc=18, ngf=64, img_f=1024, encoder_layer=5, norm='batch',
                 activation='ReLU', use_spect=True, use_coord=False, output_nc=3):
        super(PGPAN, self).__init__()

        self.encoder_layer = encoder_layer
        self.decoder_layer = encoder_layer

        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)
        input_nc = 2 * structure_nc + image_nc

        self.block0 = EncoderBlock(input_nc, ngf, norm_layer,
                                   nonlinearity, use_spect, use_coord)
        mult = 1
        for i in range(encoder_layer - 1):
            mult_prev = mult
            mult = min(2 ** (i + 1), img_f // ngf)
            block = EncoderBlock(ngf * mult_prev, ngf * mult, norm_layer,
                                 nonlinearity, use_spect, use_coord)
            setattr(self, 'encoder' + str(i), block)

        for i in range(self.decoder_layer):
            mult_prev = mult
            mult = min(2 ** (encoder_layer - i - 2), img_f // ngf) if i != encoder_layer - 1 else 1
            up = ResBlockDecoder(ngf * mult_prev, ngf * mult, ngf * mult, norm_layer,
                                 nonlinearity, use_spect, use_coord)
            setattr(self, 'decoder' + str(i), up)

        self.outconv = Output(ngf, output_nc, 3, None, nonlinearity, use_spect, use_coord)

    def forward(self, source, source_B, target_B):
        inputs = torch.cat((source, source_B, target_B), 1)
        out = self.block0(inputs)
        result = [out]
        for i in range(self.encoder_layer - 1):
            model = getattr(self, 'encoder' + str(i))
            out = model(out)
            result.append(out)
        for i in range(self.decoder_layer):
            model = getattr(self, 'decoder' + str(i))
            out = model(out)
        out_image = self.outconv(out)
        return out_image



class MainBranch(BaseNetwork):
    def __init__(self, image_nc=3, structure_nc=18, output_nc=3, ngf=64, img_f=1024, layers=6, num_blocks=2,
                norm='batch', activation='ReLU', attn_layer=[1,2], extractor_kz={'1':5,'2':5}, use_spect=True, use_coord=False, dataset = 'fashion'):
        super(MainBranch, self).__init__()

        self.layers = layers
        self.attn_layer = attn_layer
        self.is_fashion = (dataset == 'fashion')

        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)


        self.block0 = EncoderBlock(21, ngf, norm_layer,
                                 nonlinearity, use_spect, use_coord)
        mult = 1
        for i in range(layers-1):
            mult_prev = mult
            mult = min(2 ** (i + 1), img_f//ngf)
            block = EncoderBlock(ngf*mult_prev, ngf*mult, norm_layer,
                                 nonlinearity, use_spect, use_coord)
            setattr(self, 'encoder' + str(i), block)


        # decoder part
        mult = min(2 ** (layers-1), img_f//ngf)
        for i in range(layers):
            mult_prev = mult
            mult = min(2 ** (layers - i - 2), img_f // ngf) if i != layers - 1 else 1

            if i == 0:
                if self.is_fashion:
                    texture_correlation_attention = TCAM(ksize = 1, in_channel = 256)
                    setattr(self, 'TCAM' + str(i), texture_correlation_attention)
                else:
                    texture_correlation_attention = TCAM(ksize=1, rate = 1, in_channel=256)
                    setattr(self, 'TCAM' + str(i), texture_correlation_attention)

            if self.is_fashion and i == 1:
                texture_correlation_attention = TCAM(ksize = 3, in_channel = 128)
                setattr(self, 'TCAM' + str(i), texture_correlation_attention)

            if num_blocks == 1:
                up = nn.Sequential(ResBlockDecoder(ngf*mult_prev, ngf*mult, None, norm_layer,
                                         nonlinearity, use_spect, use_coord))
            else:
                up = nn.Sequential(ResBlocks(num_blocks-1, ngf*mult_prev, None, None, norm_layer,
                                             nonlinearity, False, use_spect, use_coord),
                                   ResBlockDecoder(ngf*mult_prev, ngf*mult, None, norm_layer,
                                             nonlinearity, use_spect, use_coord))
            setattr(self, 'decoder' + str(i), up)

        self.outconv = Output(ngf, output_nc, 3, None, nonlinearity, use_spect, use_coord)


    def forward(self, source, input, feature_list, test=False):
        out = self.block0(input)
        target_feature = []
        for i in range(self.layers-1):
            model = getattr(self, 'encoder' + str(i))
            out = model(out)
            target_feature.append(out)

        t_out = self.block0(source)
        source_feature = []
        for i in range(self.layers-1):
            model = getattr(self, 'encoder' + str(i))
            t_out = model(t_out)
            source_feature.append(t_out)

        flows = []
        masks = []
        grids = []
        for i in range(self.layers):

            if i == 0:
                model = getattr(self, 'TCAM' + str(i))
                attention_out, mask, offset_flow, grid = model(target_feature[1], source_feature[1], feature_list[0], test)

                flows.append(offset_flow)
                masks.append(mask)
                grids.append(grid)
                out = attention_out*mask + (1-mask)*out

            if self.is_fashion and i == 1:
                model = getattr(self, 'TCAM' + str(i))
                attention_out, mask, offset_flow, grid = model(target_feature[0], source_feature[0], feature_list[1], test)
                flows.append(offset_flow)
                masks.append(mask)
                grids.append(grid)
                out = attention_out*mask + (1-mask)*out

            model = getattr(self, 'decoder' + str(i))
            out = model(out)

        out_image = self.outconv(out)
        return out_image, masks, flows, grids


class TCAN(BaseNetwork):
    def __init__(self,  image_nc=3, structure_nc=18, output_nc=3, ngf=64, img_f=1024, layers=6, num_blocks=2,
                norm='batch', activation='ReLU', attn_layer=[1,2], extractor_kz={'1':5,'2':5}, use_spect=True, use_coord=False, dataset = 'fashion'):
        super(TCAN, self).__init__()
        self.source = SourceEncoder(3, ngf, img_f, layers,
                                                    norm, activation, use_spect, use_coord)
        self.target = MainBranch(image_nc, structure_nc, output_nc, ngf, img_f, layers, num_blocks,
                                                norm, activation, attn_layer, extractor_kz, use_spect, use_coord, dataset)

    def forward(self, source, source_B, input, target_B, test=False):
        inputs_source = torch.cat((source, source_B), 1)
        inputs_target = torch.cat((input, target_B), 1)
        feature_list = self.source(source)
        image_gen, flows, masks, grids = self.target(inputs_source, inputs_target, feature_list, test)
        return image_gen, flows, masks, grids

