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
from tensorboardX import SummaryWriter
import torchvision
from torchvision import transforms
from tqdm import tqdm
import torch.nn.functional as F

import net
from sampler import InfiniteSamplerWrapper
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


parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content_dir', default='./../', type=str,
                    help='Directory path to a batch of content images')
parser.add_argument('--style_dir', default='./../', type=str, 
                    help='Directory path to a batch of style images')
parser.add_argument('--data_name', type=str, default='polystyrene_bead')
parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')
parser.add_argument('--device', type=str, default='device:0')
parser.add_argument('--exp_name', type=str, default='multi_style')
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
parser.add_argument('--tv_weight', type=float, default=10.0)
parser.add_argument('--holo_weight', type=float, default=0.0)
parser.add_argument('--cls_weight', type=float, default=1.0)
parser.add_argument('--identity_weight', type=float, default=0.0)
parser.add_argument('--n_threads', type=int, default=4)
parser.add_argument('--n_layer', type=int, default=4)
# parser.add_argument('--style_weight', type=float, default=10.0)
parser.add_argument('--unknown_distance', type=int, default=0)  # unknown distance for 1, known distance for 0 
parser.add_argument('--save_model_interval', type=int, default=10000)
parser.add_argument('--vis_interval', type=int, default=1000)
args = parser.parse_args()

## experimental paramter for holography ##

args.wavelength = 532e-9
args.decoder = None
# args.decoder='./experiments/%s/%s/decoder_iter_60000.pth.tar'%(args.data_name, args.exp_name)

device = torch.device(args.device)
args.save_dir = args.save_dir + '/%s/%s_field_retrieval'%(args.data_name, args.exp_name)
save_dir = Path(args.save_dir)
save_dir.mkdir(exist_ok=True, parents=True)
args.log_dir = args.log_dir + '/%s/%s_field_retrieval'%(args.data_name, args.exp_name)
log_dir = Path(args.log_dir)
if os.path.exists(args.log_dir):
    shutil.rmtree(args.log_dir)

log_dir.mkdir(exist_ok=True, parents=True)
writer = SummaryWriter(log_dir=str(log_dir))

import copy
decoder = net.decoder
decoder_ph = copy.deepcopy(net.decoder)
vgg = net.vgg

state_dict = torch.load(args.vgg)
state_dict['0.weight'] = state_dict['0.weight'].sum(dim=1, keepdim=True)
vgg.load_state_dict(state_dict)

if args.decoder is not None:
    decoder.load_state_dict(torch.load(args.decoder))

vgg_layer = 31 if args.n_layer==4 else 44
vgg = nn.Sequential(*list(vgg.children())[:vgg_layer])
args.image_size = 256 if args.data_name == 'polystyrene_bead' else 128
disc = net.Discriminator(image_size=args.image_size, c_dim=2)
op_disc = torch.optim.Adam(disc.parameters(), lr=args.lr)
disc.train()
disc.to(device)
style_label = torch.ones(size=(args.batch_size, )).to(device).long()
content_label = torch.zeros(size=(args.batch_size, )).to(device).long()
label = torch.cat([style_label, content_label], dim=0)
label = label2onehot(label, dim=2).to(device)

from itertools import chain
if args.unknown_distance:
    
    distance_G = net.Distance_G()
    network = net.Net(vgg, decoder, decoder_ph, distance_G)
    optimizer = torch.optim.Adam(chain(network.decoder.parameters(), network.decoder_ph.parameters(), network.distance_g.parameters()), lr=args.lr)
else:
    network = net.Net(vgg, decoder, decoder_ph)
    optimizer = torch.optim.Adam(chain(network.decoder.parameters(), network.decoder_ph.parameters()), lr=args.lr)

network.train()
network.to(device)


if args.data_name == 'MNIST':
    # train_holo_list_style = [0.2]
    if 'single' in args.exp_name:
        train_holo_list_style = [0.2]
        train_holo_list_content = [round(float(i), 3) for i in np.arange(0.4, 0.9, 0.1)]
    elif 'half_style' in args.exp_name:
        train_holo_list_style = [round(float(i), 3) for i in np.arange(0.3, 0.6, 0.1)]
        train_holo_list_content = [round(float(i), 3) for i in np.arange(0.6, 0.9, 0.1)]
    else:
        train_holo_list_style = [round(float(i), 3) for i in np.arange(0.3, 0.9, 0.1)]
        train_holo_list_content = [round(float(i), 3) for i in np.arange(0.3, 0.9, 0.1)]
    args.distance_normalize = 1.0
    args.distance_normalize_constant = 0
    
    transform_img = transforms.Compose([transforms.Resize([64, 64]), transforms.Grayscale(), transforms.ToTensor()])
    dataset = torchvision.datasets.MNIST(root='/mnt/mooo/CS/style transfer based holographic imaging/data', download=True, train=True, transform=transform_img)
    args.pixel_size = 1.5e-6
    args.phase_normalize = 1
    
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
    
    transform_img = transforms.Compose([transforms.ToTensor(), transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()])
    dataset = Holo_loader(root='/mnt/mooo/CS/style transfer based holographic imaging/data/polystyrene_bead_holo_only', image_set='train', transform=transform_img, holo_list=train_holo_list_style, return_distance=True)    
    dataset_style = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    dataset = Holo_loader(root='/mnt/mooo/CS/style transfer based holographic imaging/data/polystyrene_bead_holo_only', image_set='test_content', transform=transform_img, holo_list=train_holo_list_content, return_distance=True)    
    dataset_content = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    args.pixel_size = 6.5e-6

model_forward = Holo_Generator(args).to(device)  # ASM, free-space propagator

for i in tqdm(range(args.max_iter)):
    if args.data_name == 'MNIST':
        style_holo, content_holo, distance_style, distance_content = mnist_loader(args, dataset, train_holo_list_style, train_holo_list_content, model_forward, device)
    elif args.data_name == 'polystyrene_bead':
        [style_holo, distance_style], [content_holo, distance_content]  = next(iter(dataset_style)), next(iter(dataset_content))
        distance_style = -args.distance_normalize_constant + distance_style.reshape([-1, 1, 1, 1]).to(args.device).float()/args.distance_normalize
        distance_content = -args.distance_normalize_constant + distance_content.reshape([-1, 1, 1, 1]).to(args.device).float()/args.distance_normalize
  
    c2s_amp_list, c2s_ph_list, amp_list, ph_list, prop_list = [], [], [], [], []
    for j in range(min(args.batch_size, 8)):
        with torch.no_grad():
            if args.unknown_distance:
                amplitude, phase, distance_content = field_retrieval(network, content_images[j:j+1], style_images[j:j+1], 1.0, True)
                # print(distance_content.shape)
            else:
                amplitude, phase = field_retrieval(network, content_images[j:j+1], style_images[j:j+1], 1.0)
                
            amp_foc, ph_foc = model_forward(amplitude, phase*args.phase_normalize, -distance_style[j:j+1]-2*args.distance_normalize_constant, return_field=True)
            if args.unknown_distance:
                prop=model_forward(amplitude, phase*args.phase_normalize, distance_content-distance_style[j:j+1]-args.distance_normalize_constant).sqrt()
            else:
                prop=model_forward(amplitude, phase*args.phase_normalize, distance_content[j:j+1]-distance_style[j:j+1]-args.distance_normalize_constant).sqrt()
                
            c2s_amp_list.append(amplitude)
            phase = unwrap(phase*args.phase_normalize)
            phase /= torch.max(phase)
            c2s_ph_list.append(phase)
            
            amp_list.append(amp_foc)
            ph_foc = unwrap(ph_foc)
            ph_foc /= torch.max(ph_foc)
            ph_list.append(ph_foc)
            
            prop_list.append(prop)
    else:
        
        c2s_amp_list=torch.cat(c2s_amp_list, dim=0)
        c2s_ph_list=torch.cat(c2s_ph_list, dim=0)
        amp_list=torch.cat(amp_list, dim=0)
        ph_list=torch.cat(ph_list, dim=0)
        prop_list=torch.cat(prop_list, dim=0)
            
    writer.add_images('content', content_images[:8], i+1)
    writer.add_images('style', style_images[:8], i+1)    
    writer.add_images('c2s_amplitude', c2s_amp_list, i+1)
    writer.add_images('c2s_phase', c2s_ph_list, i+1)
    writer.add_images('c2s_amplitude_foc', amp_list, i+1)
    writer.add_images('c2s_phase_foc', ph_list, i+1)
    writer.add_images('prop', prop_list, i+1)