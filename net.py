import torch.nn as nn
import torch 
import numpy as np

from function import adaptive_instance_normalization as adain
from function import calc_mean_std
from net_autoencoder import Encoder

class Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""
    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=6):
        super(Discriminator, self).__init__()
        layers = []
        layers.append(nn.Conv2d(1, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        kernel_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)
        
    def forward(self, x):
        h = self.main(x)
        out_src = self.conv1(h)
        out_cls = self.conv2(h)
        return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))

decoder = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU(),
    nn.ConvTranspose2d(256, 256, 2, stride=2, padding=0),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(),
    # nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ConvTranspose2d(128, 128, 2, stride=2, padding=0),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    # nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ConvTranspose2d(64, 64, 2, stride=2, padding=0),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 2, (3, 3)),
)

decoder_large = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),
    nn.ConvTranspose2d(512, 512, 2, stride=2, padding=0),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU(),
    nn.ConvTranspose2d(256, 256, 2, stride=2, padding=0),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(),
    # nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ConvTranspose2d(128, 128, 2, stride=2, padding=0),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    # nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ConvTranspose2d(64, 64, 2, stride=2, padding=0),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 2, (3, 3)),
)

vgg = nn.Sequential(
    nn.Conv2d(1, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, [:31]
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1, [:44]
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)


class Net(nn.Module):
    def __init__(self, encoder, decoder, decoder_ph=None, distance_g=None):
        super(Net, self).__init__()
        
        if isinstance(encoder, str):
            enc = Encoder()
            enc.load_state_dict(torch.load(encoder))
            enc.eval()
            enc_layers = list(enc.children())
            self.enc_1 = nn.Sequential(*enc_layers[:1])  # input -> relu1_1
            self.enc_2 = nn.Sequential(*enc_layers[1:4])  # relu1_1 -> relu2_1
            self.enc_3 = nn.Sequential(*enc_layers[4:7])  # relu2_1 -> relu3_1
            self.enc_4 = nn.Sequential(*enc_layers[7:10])  # relu3_1 -> relu4_1
        else:
            enc_layers = list(encoder.children())
            self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
            self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
            self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
            self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
            if len(enc_layers) > 35:
                self.enc_5 = nn.Sequential(*enc_layers[31:44])  # relu4_1 -> relu5_1
        # 
        self.decoder = decoder
        self.decoder_ph = decoder_ph
        self.distance_g = distance_g
        
        self.mse_loss = nn.MSELoss()
        
        self.eca = eca_layer(channel=512)
        self.n_layer = 0
        # fix the encoder
        if len(enc_layers) > 35:
            enc_list = ['enc_1', 'enc_2', 'enc_3', 'enc_4', 'enc_5']
        else:
            enc_list = ['enc_1', 'enc_2', 'enc_3', 'enc_4']
            
        for name in enc_list:
            self.n_layer += 1
            for param in getattr(self, name).parameters():
                param.requires_grad = False

    # extract relu1_1, relu2_1, relu3_1, relu4_1 from input image
    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(self.n_layer):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    # extract relu4_1 from input image
    def encode(self, input):
        for i in range(self.n_layer):
            input = getattr(self, 'enc_{:d}'.format(i + 1))(input)
        return input

    def calc_content_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        return self.mse_loss(input, target)

    def calc_style_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)

    def forward(self, content, style, alpha=1.0, field_retrieval=False, unkonwn_distance=False):
        assert 0 <= alpha <= 1
        style_feats = self.encode_with_intermediate(style)
        content_feat = self.encode(content)
        t = adain(content_feat, style_feats[-1])
        t = alpha * t + (1 - alpha) * content_feat

        # style4recon = self.eca(style_feats[-1])

        style_re = self.decoder(style_feats[-1])[:, :1]
        g_t = self.decoder(t)  # content diffraction pattern -> style diffraction pattern
        g_t_phase = g_t[:, 1:]
        g_t = g_t[:, :1]
        # g_t = self.decoder(torch.cat([t, style4recon], dim=1))  # content diffraction pattern -> style diffraction pattern
        g_t_feats = self.encode_with_intermediate(g_t)

        loss_c = self.calc_content_loss(g_t_feats[-1], t)  # calculate content loss
        loss_s = self.calc_style_loss(g_t_feats[0], style_feats[0])  # calculate style loss
        for i in range(1, self.n_layer):
            loss_s += self.calc_style_loss(g_t_feats[i], style_feats[i])
            
        if field_retrieval:
            # g_t_phase = self.decoder_ph(t)
            # g_t_feats = self.encode_with_intermediate(g_t_phase)  # calculate style loss for phase
            # for i in range(self.n_layer):
            #     loss_s += self.calc_style_loss(g_t_feats[i], style_feats[i])
            
            if unkonwn_distance:
                d_content = self.distance_g(calc_mean_std(content_feat))
                d_style = self.distance_g(calc_mean_std(style_feats[-1]))
                # d_content = self.distance_g(content_feat)
                # d_style = self.distance_g(style_feats[-1])
                return loss_c, loss_s, g_t, g_t_phase, style_re, d_content, d_style
            else:    
                return loss_c, loss_s, g_t, g_t_phase, style_re
        else:
            return loss_c, loss_s
        
    def field_retrieval(self, content, style_feats, alpha=1.0, unknown_distance=False):
        assert 0 <= alpha <= 1
        

        content_feat = self.encode(content)
        
        size = content_feat.size()
        if style_feats.shape[0]>2:  # multiple style method
            style_mean, style_std = style_feats[:style_feats.shape[0]//2], style_feats[style_feats.shape[0]//2:]
        else: # single style method
            style_mean, style_std = style_feats[:1].repeat(content.shape[0], 1, 1, 1), style_feats[1:].repeat(content.shape[0], 1, 1, 1)
            
        content_mean, content_std = calc_mean_std(content_feat)
        t = (content_feat - content_mean.expand(size)) / content_std.expand(size)
        t = t * style_std.expand(size) + style_mean.expand(size)
        
        # t = adain(content_feat, style_feats)
        
        t = alpha * t + (1 - alpha) * content_feat
        # style4recon = self.eca(style_feats)

        g_t = self.decoder(t)  # content diffraction pattern -> style diffraction pattern
        g_t_phase = g_t[:, 1:]
        g_t = g_t[:, :1]
        # g_t_phase = self.decoder_ph(t)
        
        if unknown_distance:
            if content_feat.shape[0] == 1:
                return g_t, g_t_phase, self.distance_g(calc_mean_std(content_feat.repeat(2, 1, 1, 1)))[:1, :] # self.distance_g(content_feat.repeat(2, 1, 1, 1))[:1, :]
            else:
                return g_t, g_t_phase, self.distance_g(calc_mean_std(content_feat))
        else:
            return g_t, g_t_phase


from torch.nn.parameter import Parameter

class eca_layer(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

import torch.nn.functional as F 

class Distance_G(nn.Module):
    def __init__(self):
        super(Distance_G, self).__init__()
        in_fc = 512 * 2
        self.l1 = nn.Linear(in_features=in_fc, out_features=in_fc, bias=True)
        self.b1 = nn.InstanceNorm1d(num_features=in_fc)
        # self.b1 = nn.BatchNorm1d(num_features=in_fc)
        self.d1 = nn.Dropout(p=0.5)
        self.l2 = nn.Linear(in_features=in_fc, out_features=in_fc, bias=True)
        self.b2 = nn.InstanceNorm1d(num_features=in_fc)
        # self.b2 = nn.BatchNorm1d(num_features=in_fc)
        self.d2 = nn.Dropout(p=0.5)
        self.l3 = nn.Linear(in_features=in_fc, out_features=in_fc//2, bias=True)
        self.b3 = nn.InstanceNorm1d(num_features=in_fc//2)
        # self.b3 = nn.BatchNorm1d(num_features=in_fc//2)
        self.d3 = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

        self.out=nn.Linear(in_features=in_fc//2, out_features=1)
        
        self.initialize_weights()

    def forward(self, m_s):
        m, s = m_s

        assert m.shape == s.shape
        b,c,_,_ = m.shape
        m, s = m.view(b, c), s.view(b, c)

        x = torch.cat([m , s], dim=1)
        # x = self.relu(self.b1(self.l1(x)))
        x = self.relu(self.b1(self.d1(self.l1(x))))
        # x = self.relu(self.l1(x))
        # x = self.relu(self.b2(self.l2(x)))
        x = self.relu(self.b2(self.d2(self.l2(x))))
        # x = self.relu(self.l2(x))
        # x = self.relu(self.b3(self.l3(x)))
        x = self.relu(self.b3(self.d3(self.l3(x))))
        # x = self.relu(self.l3(x))

        out = F.sigmoid(self.out(x))
        # out = self.out(x)
        return out
    
    def initialize_weights(self):
        # track all layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)