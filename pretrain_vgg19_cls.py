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
parser.add_argument('--unknown_distance', type=int, default=0)  # unknown distance for 1, known distance for 0 
parser.add_argument('--save_model_interval', type=int, default=10000)
args = parser.parse_args()

## experimental paramter for holography ##

args.wavelength = 532e-9
args.phase_normalize = 2*pi
args.distance_normalize = 1.0
args.distance_normalize_constant = 0

device = torch.device(args.device)
args.save_dir = args.save_dir + '/%s/%s_cls_pretrain'%(args.data_name, args.exp_name)
save_dir = Path(args.save_dir)
save_dir.mkdir(exist_ok=True, parents=True)
args.log_dir = args.log_dir + '/%s/%s_cls_pretrain'%(args.data_name, args.exp_name)
log_dir = Path(args.log_dir)
if os.path.exists(args.log_dir):
    shutil.rmtree(args.log_dir)

log_dir.mkdir(exist_ok=True, parents=True)
writer = SummaryWriter(log_dir=str(log_dir))


import torchvision.models as models
classifier = models.vgg19(pretrained=False, num_classes=1)
op_cls = torch.optim.Adam(classifier.parameters(), lr=args.lr)
classifier.train()
classifier.to(device)
real_label = torch.ones(size=(args.batch_size, 1)).to(device)*1.0
fake_label = torch.zeros(size=(args.batch_size, 1)).to(device)*0.0


if args.data_name == 'MNIST':
    # train_holo_list_style = [0.2]
    train_holo_list_style = [round(float(i), 3) for i in np.arange(0.3, 0.9, 0.1)]
    train_holo_list_content = [round(float(i), 3) for i in np.arange(0.3, 0.9, 0.1)]
    transform_img = transforms.Compose([transforms.Resize([64, 64]), transforms.Grayscale(), transforms.ToTensor()])
    dataset = torchvision.datasets.MNIST(root='./../../data', download=True, train=True, transform=transform_img)
    args.pixel_size = 1.5e-6
    
elif args.data_name == 'polystyrene_bead':
    train_holo_list_style = [round(float(i), 3) for i in np.arange(7, 11, 1)]
    train_holo_list_content = [round(float(i), 3) for i in np.arange(11, 14, 1)]
    # train_holo_list_style = [5]
    # train_holo_list_content = [round(float(i), 3) for i in np.arange(7, 13, 1)]
    transform_img = transforms.Compose([transforms.ToTensor(), transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()])
    dataset = Holo_loader(root='./../../data/polystyrene_bead_holo_only', image_set='train', transform=transform_img, holo_list=train_holo_list_style, return_distance=True)    
    dataset_style = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    dataset = Holo_loader(root='./../../data/polystyrene_bead_holo_only', image_set='train', transform=transform_img, holo_list=train_holo_list_content, return_distance=True)    
    dataset_content = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    args.pixel_size = 6.5e-6

model_forward = Holo_Generator(args).to(device)  # ASM, free-space propagator

mse_loss = nn.MSELoss() 
for i in tqdm(range(args.max_iter)):
    if args.data_name == 'MNIST':
        style_holo, content_holo, distance_style, distance_content = mnist_loader(args, dataset, train_holo_list_style, train_holo_list_content, model_forward, device)
    elif args.data_name == 'polystyrene_bead':
        [style_holo, distance_style], [content_holo, distance_content]  = next(iter(dataset_style)), next(iter(dataset_content))
        distance_style = -args.distance_normalize_constant + distance_style.reshape([-1, 1, 1, 1]).to(args.device).float()/args.distance_normalize
        distance_content = -args.distance_normalize_constant + distance_content.reshape([-1, 1, 1, 1]).to(args.device).float()/args.distance_normalize
    
    style_images=torch.sqrt(style_holo).to(device).float().detach() #.repeat(1, 3, 1, 1)
    content_images=torch.sqrt(content_holo).to(device).float().detach()#.repeat(1, 3, 1, 1)
    
    adjust_learning_rate(optimizer, iteration_count=i)

    op_cls.zero_grad()
    real_pred = classifier(style_images.repeat(1, 3, 1, 1))
    fake_pred = classifier(content_images.repeat(1, 3, 1, 1))

    loss_cls = mse_loss(real_label, real_pred) + mse_loss(fake_label, fake_pred)
    
    loss_cls.backward()
    op_cls.step()

    writer.add_scalar('loss_cls', loss_cls.item(), i + 1)

    if (i + 1) % args.save_model_interval == 0: 
        state_dict = classifier.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict, save_dir /
                'classifier_iter_{:d}.pth.tar'.format(i + 1))
writer.close()
