import argparse
from pathlib import Path

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image, ImageFile
import torchvision
from torchvision import transforms
from tqdm import tqdm
import torch.nn.functional as F

import net
from function import adaptive_instance_normalization, coral

from torchvision.utils import save_image
from utils.Forward_model import Holo_Generator
from utils.Data_loader import *

cudnn.benchmark = True
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
# Disable OSError: image file is truncated
ImageFile.LOAD_TRUNCATED_IMAGES = True


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


parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content_dir', default='./../', type=str,
                    help='Directory path to a batch of content images')
parser.add_argument('--style_dir', default='./../', type=str, 
                    help='Directory path to a batch of style images')
parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')
parser.add_argument('--data_name', type=str, default='MNIST')
parser.add_argument('--test_interpolation', type=int, default=0)
parser.add_argument('--exp_name', type=str, default='MNIST')
parser.add_argument('--device', type=str, default='cuda:1')
parser.add_argument('--decoder', type=str, default='./experiments')
parser.add_argument('--save_ext', default='.jpg',
                    help='The extension name of the output image')
parser.add_argument('--output', type=str, default='output',
                    help='Directory to save the output image(s)')
parser.add_argument('--batch_size', type=int, default=1)
args = parser.parse_args()

## experimental paramter for holography ##

args.wavelength = 532e-9
args.pixel_size = 1.5e-6
args.phase_normalize = 1
args.distance_normalize = 1.0
args.distance_normalize_constant = 0

args.decoder = './experiments/%s/%s_style_transfer/decoder_iter_80000.pth.tar'%(args.data_name, args.exp_name)
args.output = './output/%s/%s'%(args.data_name, args.exp_name) 
args.output += '_test_interpolation' if args.test_interpolation else '_style_tranfser'

output_dir = Path(args.output)
output_dir.mkdir(exist_ok=True, parents=True)

device = torch.device(args.device)
decoder = net.decoder
vgg = net.vgg
decoder[-1] = nn.Conv2d(64, 1, (3, 3))
# print(vgg)
state_dict = torch.load(args.vgg)
state_dict['0.weight'] = state_dict['0.weight'].sum(dim=1, keepdim=True)
vgg.load_state_dict(state_dict)

vgg = nn.Sequential(*list(vgg.children())[:31])
decoder.load_state_dict(torch.load(args.decoder))

vgg.to(device)
decoder.to(device)

model_forward = Holo_Generator(args).to(device)  # ASM, free-space propagator
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics.regression import MeanSquaredError as MSE
from torchmetrics import NormalizedRootMeanSquaredError as MSE
ssim = SSIM(data_range=1.0).to(device)
mse = MSE(normalization="mean").to(device)
if args.data_name == 'MNIST':
    if 'single' in args.exp_name:
        holo_list_style = [0.2]
        holo_list_content = [round(float(i), 3) for i in np.arange(0.4, 0.9, 0.1)]
    else:
        if args.test_interpolation:
            holo_list_style = [0.3, 0.5]
        else:
            holo_list_style = [round(float(i), 3) for i in np.arange(0.3, 0.6, 0.1)]
        holo_list_content = [round(float(i), 3) for i in np.arange(0.6, 0.9, 0.1)]
    transform_img = transforms.Compose([transforms.Resize([64, 64]), transforms.Grayscale(), transforms.ToTensor()])
    dataset = torchvision.datasets.MNIST(root='/mnt/mooo/CS/style transfer based holographic imaging/data', download=True, train=True, transform=transform_img)

elif args.data_name == 'polystyrene_bead':
    if 'single' in args.exp_name:
        holo_list_style = [7]
        holo_list_content = [round(float(i), 3) for i in np.arange(9, 13, 1)]
    else:
        if args.test_interpolation:
            holo_list_style = [7, 10]
        else:
            holo_list_style = [round(float(i), 3) for i in np.arange(7, 11, 1)]
        holo_list_content = [round(float(i), 3) for i in np.arange(11, 14, 1)]
        
    transform_img = transforms.Compose([transforms.ToTensor(), transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()])
    dataset = Holo_loader(root='/mnt/mooo/CS/style transfer based holographic imaging/data/polystyrene_bead_holo_only', image_set='train', transform=transform_img, holo_list=holo_list_style)    
    dataset_style = iter(DataLoader(dataset, batch_size=1, shuffle=True))
    dataset = Holo_loader(root='/mnt/mooo/CS/style transfer based holographic imaging/data/polystyrene_bead_holo_only', image_set='test_content', transform=transform_img, holo_list=holo_list_content)    
    dataset_content = iter(DataLoader(dataset, batch_size=1, shuffle=True))
    
for i in tqdm(range(200)):
    if args.data_name == 'MNIST':
        style_holo, content_holo, style_distance, content_ditance, gt_amplitude, gt_phase = mnist_loader_test(args, dataset, holo_list_style, holo_list_content, model_forward, device, test_interpolation=args.test_interpolation)
    elif args.data_name == 'polystyrene_bead':
        style_holo, content_holo = next(dataset_style), next(dataset_content)
    
    style_images=torch.sqrt(style_holo).to(device).float() #.repeat(1, 3, 1, 1)
    content_images=torch.sqrt(content_holo).to(device).float() #.repeat(1, 3, 1, 1)

    if args.test_interpolation:
        num_split=4
        ssim_list = []
        mse_list = []
        output = []
        output_true = []
        for weights in [[1-i/num_split, i/num_split] for i in range(num_split+1)]:
            c2s_image = style_transfer(vgg, decoder, content_images, style_images, alpha=1.0, interpolation_weights=weights)
            output.append(c2s_image)
        else:
            
            if True:
                style_distance = style_distance.to(device).float()
                min_style_d, max_style_d = style_distance[:1], style_distance[1:]
                for j in range(num_split+1):
                    out_true = model_forward(gt_amplitude, gt_phase, min_style_d + j*(max_style_d-min_style_d)/num_split).to(device).float().detach().sqrt()
                    output_true.append(out_true)
                min_style_d -= 0.1
                max_style_d += 0.1
                N=30
                for out_tmp in output:
                    ssim_list.append([])
                    mse_list.append([])
                    # out_tmp = out_tmp[:, :, 16:112, 16:112]
                    for j in range(N):
                        out_true_tmp = model_forward(gt_amplitude, gt_phase, min_style_d + j*(max_style_d-min_style_d)/N).to(device).float().detach().sqrt()
                        # print(out_true_tmp.shape, out_tmp.shape)
                        # out_true_tmp = out_true_tmp[:, :, 16:112, 16:112]
                        # print(out_true_tmp.shape, out_tmp.shape)
                        mse_list[-1].append(mse(out_tmp, out_true_tmp).item())
                        ssim_list[-1].append(ssim(out_tmp, out_true_tmp).item())
                        
                import matplotlib.pyplot as plt
                plt.figure(1)
                sx = [min_style_d.item() + j*(max_style_d.item()-min_style_d.item())/N for j in range(N)]
                for s in ssim_list:
                    plt.plot(sx, s, marker='o', markersize=4)
                    
                plt.tight_layout()
                plt.savefig(output_dir / 'output_ssim{:d}{:s}'.format(i+1,args.save_ext))
                plt.savefig(output_dir / 'output_ssim{:d}{:s}'.format(i+1,'.eps'))
                plt.close()
                plt.figure(1)
                sx = [min_style_d.item() + j*(max_style_d.item()-min_style_d.item())/N for j in range(N)]
                for s in mse_list:
                    plt.plot(sx, s, marker='o', markersize=4)
                    
                plt.tight_layout()
                plt.savefig(output_dir / 'output_mse{:d}{:s}'.format(i+1,args.save_ext))
                plt.savefig(output_dir / 'output_mse{:d}{:s}'.format(i+1,'.eps'))
                plt.close()
                        
            output = torch.cat(output, dim=3)
            output_true = torch.cat(output_true, dim=3)
            output = torch.cat([output, output_true], dim=2)
            output_name = output_dir / 'style{:d}{:s}'.format(i+1, args.save_ext)
            save_image(style_images, str(output_name))
            output_name = output_dir / 'content{:d}{:s}'.format(i+1, args.save_ext)
            save_image(content_images[:1], str(output_name))
            output_name = output_dir / 'output{:d}{:s}'.format(i+1, args.save_ext)
            save_image(output, str(output_name))

    else:
        c2s_image = style_transfer(vgg, decoder, content_images, style_images, alpha=1)
        output = torch.cat([style_images, content_images, c2s_image], dim=3)
        output_name = output_dir / 'output{:d}{:s}'.format(i+1, args.save_ext)
        save_image(output, str(output_name))
