import argparse
from pathlib import Path

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data import DataLoader
from PIL import Image, ImageFile
from tensorboardX import SummaryWriter
import torchvision
from torchvision import transforms
from tqdm import tqdm
import torch.nn.functional as F

import net
from sampler import InfiniteSamplerWrapper
from function import adaptive_instance_normalization, coral

import utils
from utils.Forward_model import Holo_Generator
from utils.Data_loader import *

cudnn.benchmark = True
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
# Disable OSError: image file is truncated
ImageFile.LOAD_TRUNCATED_IMAGES = True

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


parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content_dir', default='./../', type=str,
                    help='Directory path to a batch of content images')
parser.add_argument('--style_dir', default='./../', type=str, 
                    help='Directory path to a batch of style images')
parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')
parser.add_argument('--data_name', type=str, default='MNIST')
parser.add_argument('--exp_name', type=str, default='')
parser.add_argument('--device', type=str, default='cuda:1')

# training options
parser.add_argument('--save_dir', default='./experiments',
                    help='Directory to save the model')
parser.add_argument('--log_dir', default='./logs',
                    help='Directory to save the log')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lr_decay', type=float, default=5e-5)
parser.add_argument('--max_iter', type=int, default=80000)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--style_weight', type=float, default=10.0)
parser.add_argument('--content_weight', type=float, default=1.0)
parser.add_argument('--n_threads', type=int, default=4)
parser.add_argument('--save_model_interval', type=int, default=10000)
parser.add_argument('--vis_interval', type=int, default=1000)
args = parser.parse_args()

## experimental paramter for holography ##

args.wavelength = 532e-9
args.pixel_size = 1.5e-6
args.phase_normalize = 1
args.distance_normalize = 1.0
args.distance_normalize_constant = 0

device = torch.device(args.device)
args.save_dir = args.save_dir + '/%s/%s_style_transfer'%(args.data_name, args.exp_name)
save_dir = Path(args.save_dir)
save_dir.mkdir(exist_ok=True, parents=True)
args.log_dir = args.log_dir + '/%s/%s_style_transfer'%(args.data_name, args.exp_name)
log_dir = Path(args.log_dir)
if os.path.exists(args.log_dir):
    shutil.rmtree(args.log_dir)
writer = SummaryWriter(log_dir=str(log_dir))

decoder = net.decoder
vgg = net.vgg

state_dict = torch.load(args.vgg)
state_dict['0.weight'] = state_dict['0.weight'].sum(dim=1, keepdim=True)
vgg.load_state_dict(state_dict)

vgg = nn.Sequential(*list(vgg.children())[:31])
network = net.Net(vgg, decoder)
network.train()
network.to(device)

model_forward = Holo_Generator(args).to(device)  # ASM, free-space propagator
optimizer = torch.optim.Adam(network.decoder.parameters(), lr=args.lr)

if args.data_name == 'MNIST':
    if 'single' in args.exp_name:
        train_holo_list_style = [0.2]
        train_holo_list_content = [round(float(i), 3) for i in np.arange(0.4, 0.9, 0.1)]
    elif 'half_style' in args.exp_name:
        train_holo_list_style = [round(float(i), 3) for i in np.arange(0.3, 0.6, 0.1)]
        train_holo_list_content = [round(float(i), 3) for i in np.arange(0.6, 0.9, 0.1)]
    else:
        train_holo_list_style = [round(float(i), 3) for i in np.arange(0.3, 0.9, 0.1)]
        train_holo_list_content = [round(float(i), 3) for i in np.arange(0.3, 0.9, 0.1)]
    transform_img = transforms.Compose([transforms.Resize([64, 64]), transforms.Grayscale(), transforms.ToTensor()])
    dataset = torchvision.datasets.MNIST(root='/mnt/mooo/CS/style transfer based holographic imaging/data', download=True, train=True, transform=transform_img)

elif args.data_name == 'polystyrene_bead':
    if 'single' in args.exp_name:
        train_holo_list_style = [7]
        train_holo_list_content = [round(float(i), 3) for i in np.arange(9, 13, 1)]
    else:
        train_holo_list_style = [round(float(i), 3) for i in np.arange(7, 11, 1)]
        train_holo_list_content = [round(float(i), 3) for i in np.arange(11, 14, 1)]
    transform_img = transforms.Compose([transforms.ToTensor(), transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()])
    dataset = Holo_loader(root='/mnt/mooo/CS/style transfer based holographic imaging/data/data/polystyrene_bead_holo_only', image_set='train', transform=transform_img, holo_list=train_holo_list_style)    
    dataset_style = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    dataset = Holo_loader(root='/mnt/mooo/CS/style transfer based holographic imaging/data/data/polystyrene_bead_holo_only', image_set='train', transform=transform_img, holo_list=train_holo_list_content)    
    dataset_content = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
for i in tqdm(range(args.max_iter)):
    if args.data_name == 'MNIST':
        style_holo, content_holo, _, _ = mnist_loader(args, dataset, train_holo_list_style, train_holo_list_content, model_forward, device)
    elif args.data_name == 'polystyrene_bead':
        style_holo, content_holo = next(iter(dataset_style)), next(iter(dataset_content))
    
    style_images=torch.sqrt(style_holo).to(device).float() #.repeat(1, 3, 1, 1)
    content_images=torch.sqrt(content_holo).to(device).float() #.repeat(1, 3, 1, 1)
    
    adjust_learning_rate(optimizer, iteration_count=i)
    
    loss_c, loss_s = network(content_images, style_images)
    loss_c = args.content_weight * loss_c
    loss_s = args.style_weight * loss_s
    loss = loss_c + loss_s

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    writer.add_scalar('loss_content', loss_c.item(), i + 1)
    writer.add_scalar('loss_style', loss_s.item(), i + 1)

    if (i + 1) % args.vis_interval == 0 or (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
        with torch.no_grad():
            output = torch.cat([style_transfer(vgg, decoder, content_images[j:j+1], style_images[j:j+1], 1.0) for j in range(8)], dim=0)
            
        writer.add_images('content', content_images[:8], i+1)
        writer.add_images('style', style_images[:8], i+1)    
        writer.add_images('style_transferred', output, i+1)
        
        if (i + 1) % args.save_model_interval == 0:
            state_dict = network.decoder.state_dict()
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].to(torch.device('cpu'))
            torch.save(state_dict, save_dir /
                    'decoder_iter_{:d}.pth.tar'.format(i + 1))
writer.close()
