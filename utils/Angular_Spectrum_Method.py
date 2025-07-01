
import numpy as np
from math import pi
import torch
from torch.nn.functional import pad

def ASM(O, lamb, d, px, requires_grad=True, zero_padding=False): # torch version

    batch, c, Sh, Sw = O.shape

    if zero_padding:
        O = pad(O, pad=(Sh//2, Sh//2, Sw//2, Sw//2), mode="replicate")
        fx = np.arange(Sh*2)/2 - Sh//2
        fy = np.arange(Sw*2)/2 - Sw//2
    else:
        fx = np.arange(Sh) - Sh//2
        fy = np.arange(Sw) - Sw//2

    fx = fx/(Sh*px)
    fy = fy/(Sw*px)

    G_in = 1 - (lamb**2)*(np.matmul(np.ones([len(fx), 1]), (fx**2).reshape(1, -1)).T + np.matmul(np.ones([len(fy), 1]), (fy**2).reshape(1, -1)))
    G_in = torch.from_numpy(np.repeat((np.sqrt((G_in > 0)*1 * G_in)/lamb).reshape((1, 1, len(fx), len(fy))), batch, axis=0))

    if O.is_cuda:
        G_in = G_in.to(O.device)

    G_in.requires_grad_(requires_grad)
    G_in = torch.exp(1j*2*pi*d*G_in)

    O_fft = torch_fft(O)
    # print(d.shape, G_in.shape, O_fft.shape)
    H = torch_ifft(G_in*O_fft)
    H = center_crop(H, Sh)

    return H

def torch_fft(H):

    H = torch.fft.fftshift(torch.fft.fft2(H), dim=(-2, -1))

    return H

def torch_ifft(H):

    H = torch.fft.ifft2(torch.fft.ifftshift(H, dim=(-2, -1)))

    return H

def center_crop(H, size):
    batch, channel, Nh, Nw = H.size()

    return H[:, :, (Nh - size)//2 : (Nh+size)//2, (Nw - size)//2 : (Nw+size)//2]