import torch
import torch.nn.functional as F
from torch import nn
from math import pi

class holo_auto_encoder(nn.Module):
    def __init__(self):
        super(holo_auto_encoder, self).__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x, layer_out=False):
        if layer_out:
            [l1, l2, l3, l4, encoded] = self.encoder(x, layer_out=layer_out)
            decoded = self.decoder(encoded)
            return [l1, l2, l3, l4, encoded], decoded
        else:
            encoded = self.encoder(x, layer_out=layer_out)
            decoded = self.decoder(encoded)
            
            return decoded

class Encoder(nn.Module):
    def __init__(self, input_channel=1):
        super(Encoder, self).__init__()
        self.use_norm = False
        self.input_channel = 1
        self.lrelu_use = False
        self.batch_mode = 'I'
        k=7
        p=3

        c1 = 64
        c2 = c1 * 2
        c3 = c2 * 2
        c4 = c3 * 2
        c5 = c4 * 2
        
        self.l10 = CBR(in_channel=self.input_channel, out_channel=c1, use_norm=False, lrelu_use=self.lrelu_use, kernel=k, padding=p) # 0
        self.l11 = CBR(in_channel=c1, out_channel=c1, use_norm=False, lrelu_use=self.lrelu_use,kernel=k, padding=p) # 1
        self.mpool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 2
        
        self.l20 = CBR(in_channel=c1, out_channel=c2, use_norm=self.use_norm, lrelu_use=self.lrelu_use,
                       batch_mode=self.batch_mode, kernel=k, padding=p) # 3
        self.l21 = CBR(in_channel=c2, out_channel=c2, use_norm=self.use_norm, lrelu_use=self.lrelu_use,
                       batch_mode=self.batch_mode, kernel=k, padding=p) # 4
        self.mpool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 5
        
        self.l30 = CBR(in_channel=c2, out_channel=c3, use_norm=self.use_norm, lrelu_use=self.lrelu_use,
                       batch_mode=self.batch_mode, kernel=k, padding=p) # 6
        self.l31 = CBR(in_channel=c3, out_channel=c3, use_norm=self.use_norm, lrelu_use=self.lrelu_use,
                       batch_mode=self.batch_mode, kernel=k, padding=p) # 7
        self.mpool3 = nn.MaxPool2d(kernel_size=2, stride=2) # 8
        
        self.l40 = CBR(in_channel=c3, out_channel=c4, use_norm=self.use_norm, lrelu_use=self.lrelu_use,
                       batch_mode=self.batch_mode, kernel=k, padding=p) # 9
        self.l41 = CBR(in_channel=c4, out_channel=c4, use_norm=self.use_norm, lrelu_use=self.lrelu_use,
                       batch_mode=self.batch_mode, kernel=k, padding=p) # 10
        self.mpool4 = nn.MaxPool2d(kernel_size=2, stride=2) # 11
        
        self.l50 = CBR(in_channel=c4, out_channel=c5, use_norm=self.use_norm, lrelu_use=self.lrelu_use,
                       batch_mode=self.batch_mode, kernel=k, padding=p)
        self.l51 = CBR(in_channel=c5, out_channel=c5, use_norm=self.use_norm, lrelu_use=self.lrelu_use,
                       batch_mode=self.batch_mode, kernel=k, padding=p)
        

    def forward(self, x, layer_out=False):

        l1 = self.l11(self.l10(x))
        l1_pool = self.mpool1(l1)

        l2 = self.l21(self.l20(l1_pool))
        l2_pool = self.mpool2(l2)

        l3 = self.l31(self.l30(l2_pool))
        l3_pool = self.mpool3(l3)

        l4 = self.l41(self.l40(l3_pool))
        l4_pool = self.mpool4(l4)

        latent = self.l51(self.l50(l4_pool))

        if layer_out:
            return [l1, l2, l3, l4, latent]

        else:
            return latent
        

class Decoder(nn.Module):
    def __init__(self, output_channel=1, skip=False):
        super(Decoder, self).__init__()
        self.use_norm = False
        self.output_channel = output_channel
        self.lrelu_use = False
        self.batch_mode = 'I'

        c1 = 64
        c2 = c1*2
        c3 = c2*2
        c4 = c3*2
        c5 = c4*2


        self.upsampling_module = nn.ModuleList()

        for idx, (i, j) in enumerate(zip([c5, c4, c3, c2], [c4, c3, c2, c1])):
            self.upsampling_module.append(nn.Sequential(CBR(in_channel=i, out_channel=i, use_norm=self.use_norm,
                                                            lrelu_use=self.lrelu_use, batch_mode=self.batch_mode),
                                                        CBR(in_channel=i, out_channel=j, use_norm=self.use_norm,
                                                            lrelu_use=self.lrelu_use, batch_mode=self.batch_mode),
                                                        nn.UpsamplingBilinear2d(scale_factor=2)
                                                        ))


        self.conv_out = nn.Conv2d(in_channels=c1, out_channels=self.output_channel, kernel_size=(1,1), stride=(1,1), padding=(0,0))

        self.activation = nn.LeakyReLU() if self.lrelu_use else nn.ReLU()

    def forward(self, x):

        for _, m in enumerate(self.upsampling_module):
            x = m(x)
            
        x = self.conv_out(x)

        return x

        

class CBR(nn.Module):
    def __init__(self, in_channel, out_channel, padding=1, use_norm=True, kernel=3, stride=1
                 , lrelu_use=False, slope=0.1, batch_mode='I', sampling='down', rate=1):
        super(CBR, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.use_norm = use_norm
        self.lrelu = lrelu_use

        if sampling == 'down':
            self.Conv = nn.Conv2d(self.in_channel, self.out_channel, kernel_size=(kernel, kernel), stride=(stride, stride),
                                  padding=padding, dilation=(rate, rate))
        else:
            self.Conv = nn.ConvTranspose2d(self.in_channel, self.out_channel, kernel_size=(2,2), stride=(2,2), padding=(0,0))
        if self.use_norm:
            if batch_mode == 'I':
                self.Batch = nn.InstanceNorm2d(self.out_channel)
            elif batch_mode == 'G':
                self.Batch = nn.GroupNorm(self.out_channel//16, self.out_channel)
            else:
                self.Batch = nn.BatchNorm2d(self.out_channel)

        if self.lrelu:
            self.activation = nn.LeakyReLU(negative_slope=slope)
        else:
            self.activation = nn.ReLU()

    def forward(self, x):

        if self.use_norm:
            out = self.activation(self.Batch(self.Conv(x)))
        else:
            out = self.activation(self.Conv(x))

        return out
    
if __name__ == "__main__":
    encoder = Encoder()
    enc_layers = list(encoder.children())
    print("1", enc_layers[:2])
    print("2", enc_layers[2:4])
    print("3", enc_layers[4:6])
    print("4", enc_layers[6:8])