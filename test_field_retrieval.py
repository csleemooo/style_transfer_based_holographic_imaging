import argparse
from pathlib import Path
import shutil

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data import DataLoader
from PIL import Image, ImageFile
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm
import torch.nn.functional as F

import net
from function import adaptive_instance_normalization, coral
import lpips

import utils
from utils.Forward_model import Holo_Generator
from utils.functions import unwrap, tv_loss
from utils.Data_loader import *

from math import pi

cudnn.benchmark = True
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
# Disable OSError: image file is truncated
ImageFile.LOAD_TRUNCATED_IMAGES = True


def train_transform():
    transform_list = [
        transforms.Resize(size=(512, 512)),
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)


class FlatFolderDataset(data.Dataset):
    def __init__(self, root, transform):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        self.paths = list(Path(self.root).glob('*'))
        self.transform = transform

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(str(path)).convert('RGB')
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'FlatFolderDataset'


def adjust_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = args.lr / (1.0 + args.lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

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

def field_retrieval(network, content, style, alpha=1.0, unkonwn_distance=False):
    assert (0.0 <= alpha <= 1.0)
    if unkonwn_distance:
        amplitude, phase, distance_content = network.field_retrieval(content, style, alpha, unkonwn_distance)
        return amplitude, phase, distance_content.view(-1, 1, 1, 1)
    else:
        amplitude, phase = network.field_retrieval(content, style, alpha)
        return amplitude, phase

####  code from https://github.com/yunjey/stargan/blob/master/solver.py   ####
def cls_loss(logit, target):
    return F.binary_cross_entropy_with_logits(logit, target, size_average=False)/logit.size(0)

def gradient_penalty(y, x, device):
    """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
    weight = torch.ones(y.size()).to(device)
    dydx = torch.autograd.grad(outputs=y,
                                inputs=x,
                                grad_outputs=weight,
                                retain_graph=True,
                                create_graph=True,
                                only_inputs=True)[0]

    dydx = dydx.view(dydx.size(0), -1)
    dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
    return torch.mean((dydx_l2norm-1)**2)

def label2onehot(labels, dim):
    """Convert label indices to one-hot vectors."""
    batch_size = labels.size(0)
    out = torch.zeros(batch_size, dim)
    out[np.arange(batch_size), labels.long()] = 1
    return out
###############################################################################

def weight_init_xavier_uniform(submodule):
    if isinstance(submodule, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform_(submodule.weight)
        submodule.bias.data.fill_(0.01)
    elif isinstance(submodule, torch.nn.BatchNorm2d):
        submodule.weight.data.fill_(1.0)
        submodule.bias.data.zero_()


parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content_dir', default='./../', type=str,
                    help='Directory path to a batch of content images')
parser.add_argument('--style_dir', default='./../', type=str, 
                    help='Directory path to a batch of style images')
parser.add_argument('--data_name', type=str, default='MNIST')
parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')
parser.add_argument('--device', type=str, default='device:0')
parser.add_argument('--exp_name', type=str, default='241104_half_style_half_content_disc')
# training options
parser.add_argument('--save_dir', default='./output',
                    help='Directory to save the model')
parser.add_argument('--log_dir', default='./logs',
                    help='Directory to save the log')
parser.add_argument('--save_ext', default='.jpg',
                    help='Directory to save the log')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--n_layer', type=int, default=4)
# parser.add_argument('--style_weight', type=float, default=10.0)
parser.add_argument('--unknown_distance', type=int, default=0)  # unknown distance for 1, known distance for 0 
args = parser.parse_args()
# test code #
## experimental paramter for holography ##
args.wavelength = 532e-9
args.decoder ='./experiments/%s/%s_field_retrieval/decoder_iter_80000.pth.tar'%(args.data_name, args.exp_name)
args.decoder_ph ='./experiments/%s/%s_field_retrieval/decoder_ph_iter_80000.pth.tar'%(args.data_name, args.exp_name)

device = torch.device(args.device)
args.save_dir = args.save_dir + '/%s/%s_field_retrieval'%(args.data_name, args.exp_name)
output_dir = Path(args.save_dir)
output_dir.mkdir(exist_ok=True, parents=True)

import copy
decoder = net.decoder
decoder_ph = copy.deepcopy(net.decoder)
decoder.load_state_dict(torch.load(args.decoder))
decoder_ph.load_state_dict(torch.load(args.decoder_ph))

vgg = net.vgg

state_dict = torch.load(args.vgg)
state_dict['0.weight'] = state_dict['0.weight'].sum(dim=1, keepdim=True)
vgg.load_state_dict(state_dict)

vgg_layer = 31 if args.n_layer==4 else 44
vgg = nn.Sequential(*list(vgg.children())[:vgg_layer])

args.image_size = 256 if args.data_name == 'polystyrene_bead' else 128

distance_G = net.Distance_G()
network = net.Net(vgg, decoder, decoder_ph, distance_G)

network.eval()
network.to(device)


if args.data_name == 'MNIST':
    # train_holo_list_style = [0.2]
    if 'single' in args.exp_name:
        train_holo_list_style = [0.2]
        train_holo_list_content = [round(float(i), 3) for i in np.arange(0.4, 0.9, 0.1)]
    elif 'half_style' in args.exp_name:
        train_holo_list_style = [round(float(i), 3) for i in np.arange(0.3, 0.6, 0.1)]
        train_holo_list_content = [round(float(i), 3) for i in np.arange(0.6, 0.9, 0.1)]

    args.distance_normalize = 1.0
    args.distance_normalize_constant = 0
    
    transform_img = transforms.Compose([transforms.Resize([64, 64]), transforms.Grayscale(), transforms.ToTensor()])
    dataset = torchvision.datasets.MNIST(root='/mnt/mooo/CS/style transfer based holographic imaging/data', download=True, train=True, transform=transform_img)
    args.pixel_size = 1.5e-6
    args.phase_normalize = 1
    test_num = 100
    
elif args.data_name == 'polystyrene_bead':
    if 'single' in args.exp_name:
        train_holo_list_style = [7]
        train_holo_list_content = [round(float(i), 3) for i in np.arange(9, 13, 1)]
    else:
        train_holo_list_style = [round(float(i), 3) for i in np.arange(7, 11, 1)]
        train_holo_list_content = [round(float(i), 3) for i in np.arange(11, 14, 1)]
            
    args.distance_min = min(train_holo_list_style)
    args.distance_max = max(train_holo_list_content)
    args.distance_normalize = args.distance_max - args.distance_min
    args.distance_normalize_constant = args.distance_min/args.distance_normalize
    args.phase_normalize = 2*pi
    
    transform_img = transforms.Compose([transforms.ToTensor()])
    dataset = Holo_loader(root='/mnt/mooo/CS/style transfer based holographic imaging/data/polystyrene_bead_holo_only', image_set='train', transform=transform_img, holo_list=train_holo_list_style, return_distance=True)    
    dataset_style = iter(DataLoader(dataset, batch_size=args.batch_size, shuffle=True))
    dataset = Holo_loader(root='/mnt/mooo/CS/style transfer based holographic imaging/data/polystyrene_bead_holo_only', image_set='test', transform=transform_img, holo_list=train_holo_list_content, return_distance=True)    
    dataset_content = iter(DataLoader(dataset, batch_size=args.batch_size, shuffle=False))
    args.pixel_size = 6.5e-6
    test_num = len(dataset_content)

model_forward = Holo_Generator(args).to(device)  # ASM, free-space propagator
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics.image import PeakSignalNoiseRatio as PSNR
from torchmetrics import MeanAbsoluteError as MAE
ssim = SSIM()
psnr = PSNR()
mae = MAE()

psnr_list, mae_list = [], []
vis_idx = 0
for i in tqdm(range(test_num)):
    if args.data_name == 'MNIST':
        style_holo, content_holo, distance_style, distance_content, gt_amplitude, gt_phase = mnist_loader(args, dataset, train_holo_list_style, train_holo_list_content, model_forward, device, return_gt=True)
    elif args.data_name == 'polystyrene_bead':
        [style_holo, distance_style], [content_holo, distance_content, gt_amplitude, gt_phase]  = next(dataset_style), next(dataset_content)
        distance_style = -args.distance_normalize_constant + distance_style.reshape([-1, 1, 1, 1]).to(args.device).float()/args.distance_normalize
        distance_content = -args.distance_normalize_constant + distance_content.reshape([-1, 1, 1, 1]).to(args.device).float()/args.distance_normalize
    
    style_images=torch.sqrt(style_holo).to(device).float().detach() #.repeat(1, 3, 1, 1)
    content_images=torch.sqrt(content_holo).to(device).float().detach() #.repeat(1, 3, 1, 1)

    for j in range(args.batch_size):
        vis_idx+=1
        
        gt_phase_tmp = gt_phase[j:j+1]
        gt_amp_tmp = gt_amplitude[j:j+1]
        with torch.no_grad():
            amplitude, phase, distance_content = field_retrieval(network, content_images[j:j+1], style_images[j:j+1], 1.0, True)

            amp_foc, ph_foc = model_forward(amplitude, phase*args.phase_normalize, -distance_style[j:j+1]-2*args.distance_normalize_constant, return_field=True)

            gt_phase_tmp /= torch.max(gt_phase_tmp)
            phase = unwrap(phase.detach())
            phase -= torch.min(phase)
            phase /= torch.max(phase)
            ph_foc = unwrap(ph_foc.detach())
            ph_foc -= torch.min(ph_foc)
            ph_foc /= torch.max(ph_foc)
            
            
            psnr_list.append(psnr(ph_foc, gt_phase_tmp).item())
            mae_list.append(mae(ph_foc, gt_phase_tmp).item())
        
        if vis_idx%10 == 0:
            inputs = torch.cat([content_images[j:j+1], style_images[j:j+1]], dim=2)
            recon_field = torch.cat([amplitude, phase], dim=2)
            gt_field =torch.cat([gt_amp_tmp, gt_phase_tmp], dim=2)
            recon_foc_field = torch.cat([amp_foc.detach(), ph_foc], dim=2)
            
            total = torch.cat([inputs, recon_field, gt_field, recon_foc_field, torch.abs(gt_field-recon_foc_field)], dim=3)
            output_name = output_dir / 'result_field{:d}{:s}'.format(vis_idx, args.save_ext)
            save_image(total, str(output_name))
        if vis_idx == 500:
            print(np.mean(psnr_list))
            print(np.mean(mae_list))
            break
    else:
        print(np.mean(psnr_list))
        print(np.mean(mae_list))
    if vis_idx == 500:
        break