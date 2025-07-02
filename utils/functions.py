import numpy as np
from skimage.restoration import unwrap_phase
import torch

def style_transfer(vgg, decoder, content, style, alpha=1.0,
                   interpolation_weights=None):
    assert (0.0 <= alpha <= 1.0)
    content_f = vgg(content)
    style_f = vgg(style)
    if interpolation_weights:
        _, C, H, W = content_f.size()
        feat = torch.FloatTensor(1, C, H, W).zero_().to(device)
        base_feat = adaptive_instance_normalization(content_f, style_f)
        for i, w in enumerate(interpolation_weights):
            feat = feat + w * base_feat[i:i + 1]
        content_f = content_f[0:1]
    else:
        feat = adaptive_instance_normalization(content_f, style_f)
    feat = feat * alpha + content_f * (1 - alpha)
    return decoder(feat)

def field_retrieval(network, content, style_vector, alpha=1.0, unkonwn_distance=False):
    assert (0.0 <= alpha <= 1.0)
    if unkonwn_distance:
        amplitude, phase, distance_content = network.field_retrieval(content, style_vector, alpha, unkonwn_distance)
        return amplitude, phase, distance_content.view(-1, 1, 1, 1)
    else:
        amplitude, phase = network.field_retrieval(content, style_vector, alpha)
        return amplitude, phase
    

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

def conv1d_r(r_out, kernel_size, stride, dilation=1):
    ''' Computes receptive field size berfore a conv1d layer. '''
    if dilation == 1:
        return r_out * stride + max(kernel_size - stride, 0)
    else:
        raise NotImplementedError('Dilated conv is not implemented yet.')


def conv2d_r(r_out, kernel_size, stride, dilation=1):
    ''' Computes receptive field size berfore a conv2d layer. '''

    assert isinstance(r_out, tuple)
    assert isinstance(kernel_size, (int, tuple))
    assert isinstance(stride, (int, tuple))
    assert isinstance(dilation, (int, tuple))

    kernel_0 = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
    kernel_1 = kernel_size if isinstance(kernel_size, int) else kernel_size[1]
    stride_0 = stride if isinstance(stride, int) else stride[0]
    stride_1 = stride if isinstance(stride, int) else stride[1]
    dilation_0 = dilation if isinstance(dilation, int) else dilation[0]
    dilation_1 = dilation if isinstance(dilation, int) else dilation[1]

    return (conv1d_r(r_out[0], kernel_0, stride_0, dilation_0),
            conv1d_r(r_out[1], kernel_1, stride_1, dilation_1))


def receptive_field_calculator(layers):
    '''
        The receptive field calculator.
        Input:
            layers [list]: a list of layers, each consisting of
                its (1) layer type, (2) kernel size, (3) stride,
                and (4) dilation.
    '''

    # Compute receptive field
    if layers[0][0] == 'conv1d':
        r_field = [1]
    else:
        r_field = [(1, 1)]

    for i, (layer_type, kernel_size, stride, dilation) in \
            reversed(list(enumerate(layers))):
        if layer_type == 'conv1d':
            r_field.append(
                conv1d_r(r_field[-1], kernel_size, stride, dilation))
        elif layer_type == 'conv2d':
            r_field.append(
                conv2d_r(r_field[-1], kernel_size, stride, dilation))
        else:
            raise ValueError(f'Unknown layer type {layer_type}')

    # Print results
    format_str = ' {:<6} {:<10} {:<8} {:<8} {:<10} {:<15}'
    print('-' * 61)
    print(format_str
          .format('layer', 'type', 'kernel', 'stride', 'dilation', 'r field'))
    print('-' * 61)
    for i, (layer_type, kernel_size, stride, dilation) in enumerate(layers):
        print(format_str
              .format(i + 1, layer_type,
                      str(kernel_size), str(stride), str(dilation),
                      str(r_field[-(i + 1)])))
    print('-' * 61)