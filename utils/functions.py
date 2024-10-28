import numpy as np
from skimage.restoration import unwrap_phase
import torch

def tv_loss(img, norm=False, order=1):
    dh = img[:, :, 1:, :] - img[:, :, :-1, :]
    dw = img[:, :, :, 1:] - img[:, :, :, :-1]

    tv = (torch.norm(dh.abs(), p=order) + torch.norm(dw.abs(), p=order)) / img.size(2) / img.size(3)

    if norm:
        tv = tv / (img.abs().detach().mean())

    return tv


def unwrap(x):
    
    x = x.cpu().detach().numpy()
    if x.ndim>2:
        x = x.squeeze()
    
    x = unwrap_phase(x)
    x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0)
    
    return x


def center_crop(H, size):
    batch, channel, Nh, Nw = H.size()

    return H[:, :, (Nh - size)//2 : (Nh+size)//2, (Nw - size)//2 : (Nw+size)//2]

def center_crop_numpy(H, size):
    Nh = H.shape[0]
    Nw = H.shape[1]

    return H[(Nh - size)//2 : (Nh+size)//2, (Nw - size)//2 : (Nw+size)//2]

def amp_pha_generate(real, imag):
    field = real + 1j*imag
    amplitude = np.abs(field)
    phase = np.angle(field)

    return amplitude, phase

def make_path(path):
    import os
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def save_fig(holo, fake_holo, real_amplitude, fake_amplitude, real_phase, fake_phase, real_distance, fake_distance, args, save_file):
    import matplotlib.pyplot as plt
    from math import pi
    fake_distance = fake_distance*args.distance_normalize
    fig2 = plt.figure(2, figsize=[12, 8])

    plt.subplot(2, 3, 1)
    plt.title('input holography')
    plt.imshow(holo, cmap='gray', vmax=1, vmin=0)
    plt.axis('off')
    plt.subplot(2, 3, 2)
    plt.title('ground truth' + str(real_distance) + 'mm')
    plt.imshow(real_amplitude, cmap='gray', vmax=1, vmin=0)
    plt.axis('off')
    plt.colorbar()
    plt.subplot(2, 3, 3)
    plt.title('output ' + str(np.round(fake_distance, 2)) + 'mm')
    plt.imshow(fake_amplitude, cmap='gray', vmax=1, vmin=0)
    plt.axis('off')
    plt.colorbar()

    plt.subplot(2, 3, 4)
    plt.title('generated_holography')
    plt.imshow(fake_holo, cmap='gray', vmax=1, vmin=0)
    plt.axis('off')
    plt.subplot(2, 3, 5)
    plt.title('ground truth phase')
    plt.imshow(real_phase, cmap='jet', vmax=pi, vmin=-pi)
    plt.axis('off')
    plt.colorbar()
    plt.subplot(2, 3, 6)
    plt.title('output phase')
    plt.imshow(fake_phase, cmap='jet', vmax=pi, vmin=-pi)
    plt.axis('off')
    plt.colorbar()

    fig2.savefig(save_file)
    plt.close(fig2)

def standardization(x):
    return (x-0.05)/0.1

def de_standardization(x):
    return (x+1)/2

