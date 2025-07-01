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
import matplotlib.pyplot as plt
import net
from function import adaptive_instance_normalization, coral
import lpips

import utils
from utils.Forward_model import Holo_Generator
from utils.functions import unwrap, tv_loss
from utils.Data_loader import *

from math import pi

import warnings
warnings.filterwarnings("ignore")

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

def field_retrieval(network, content, style_vector, alpha=1.0, unkonwn_distance=False):
    assert (0.0 <= alpha <= 1.0)
    if unkonwn_distance:
        amplitude, phase, distance_content = network.field_retrieval(content, style_vector, alpha, unkonwn_distance)
        return amplitude, phase, distance_content.view(-1, 1, 1, 1)
    else:
        amplitude, phase = network.field_retrieval(content, style_vector, alpha)
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
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--exp_name', type=str, default='241104_single_style_content_disc')
# training options
parser.add_argument('--save_dir', default='./output/MNIST/test_cs_vid2',
                    help='Directory to save the model')
parser.add_argument('--log_dir', default='./logs',
                    help='Directory to save the log')
parser.add_argument('--save_ext', default='.jpg',
                    help='Directory to save the log')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--n_layer', type=int, default=4)
# parser.add_argument('--style_weight', type=float, default=10.0)
parser.add_argument('--unknown_distance', type=int, default=1)  # unknown distance for 1, known distance for 0 
args = parser.parse_args()
# test code #
## experimental paramter for holography ##2
args.wavelength = 532e-9

args.decoder_root='./experiments/MNIST/single_style_disc_instance_field_retrieval'
args.decoder = os.path.join(args.decoder_root, 'decoder_iter_80000.pth.tar')
args.decoder_ph = os.path.join(args.decoder_root, 'decoder_ph_iter_80000.pth.tar')
args.distance_g = os.path.join(args.decoder_root, 'distance_g_iter_80000.pth.tar')
args.style_path = os.path.join('./style_representation', 'MNIST', 'style_vector.pt')

device = torch.device(args.device)
args.save_dir = args.save_dir + '/%s/%s_field_retrieval'%(args.data_name, args.exp_name)
output_dir = Path(args.save_dir)
output_dir.mkdir(exist_ok=True, parents=True)

import copy
decoder = net.decoder.to(device)
decoder_ph = copy.deepcopy(net.decoder)
distance_G = net.Distance_G().to(device)

decoder.load_state_dict(torch.load(args.decoder))
decoder_ph.load_state_dict(torch.load(args.decoder_ph))
distance_G.load_state_dict(torch.load(args.distance_g))

vgg = net.vgg.to(device)

state_dict = torch.load(args.vgg)
state_dict['0.weight'] = state_dict['0.weight'].sum(dim=1, keepdim=True)
vgg.load_state_dict(state_dict)

vgg_layer = 31 if args.n_layer==4 else 44
vgg = nn.Sequential(*list(vgg.children())[:vgg_layer])

args.image_size = 256 if args.data_name == 'polystyrene_bead' else 128


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
    dataset = torchvision.datasets.MNIST(root='/mnt/mooo/CS/style transfer based holographic imaging/data', download=True, train=False, transform=transform_img)
    args.pixel_size = 1.5e-6
    args.phase_normalize = 1

model_forward = Holo_Generator(args).to(device)  # ASM, free-space propagator

from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics.image import PeakSignalNoiseRatio as PSNR
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure as MSSSIM
from torchmetrics.regression import MeanAbsoluteError
ssim = SSIM()
msssim = MSSSIM(data_range=1.0, betas=(0.0448, 0.2856, 0.3001, 0.2363))
psnr = PSNR().to(device)
mae = MeanAbsoluteError()

ssim_list, psnr_list, mae_list, msssim_list = [], [], [], []
distance_list= []
vis_idx = 0
video_array = []
distance_per_psnr, distance_per_mae = {i:[] for i in train_holo_list_content}, {i:[] for i in train_holo_list_content}



from function import calc_mean_std
style_mean, style_std = [], []

if os.path.isfile(args.style_path):
    style_vector = torch.load(args.style_path)
else:
    from function import calc_mean_std
    style_mean, style_std = [], []
    for t in range(50):
        style_holo = torch.load(f"/mnt/mooo/fakhriyya/new_code/style_transfer_based_holographic_imaging/new_dataset_241122/test_style_holo_{t}.pt")
        feat = network.encode(torch.sqrt(style_holo).to(device).float().detach())
        mean, std = calc_mean_std(feat)
        style_mean.append(mean)
        style_std.append(std)
    else:
        style_mean = torch.cat(style_mean, dim=0).mean(dim=0, keepdim=True)
        style_std = torch.cat(style_std, dim=0).mean(dim=0, keepdim=True)
        style_vector = torch.cat([style_mean, style_std], dim=0)
        
    torch.save(style_vector.detach().cpu(), args.style_path)
        
style_vector = style_vector.to(args.device).float()
import test_cs_vid2

for i in range(100):
    if args.data_name == 'MNIST':
        #style_holo, content_holo, distance_style, distance_content, gt_amplitude, gt_phase = mnist_loader(args, dataset, train_holo_list_style, train_holo_list_content, model_forward, device, return_gt=True)
        style_holo = torch.load(f"./test_cs_vid2/test_style_holo_{i}.pt")
        content_holo = torch.load(f"./test_cs_vid2/test_content_holo_{i}.pt")
        distance_style = torch.load(f"./test_cs_vid2/test_distance_style_{i}.pt")
        distance_content = torch.load(f"./test_cs_vid2/test_distance_content_{i}.pt")
        gt_amplitude = torch.load(f"./test_cs_vid2/test_gt_amplitude_{i}.pt")
        gt_phase = torch.load(f"./test_cs_vid2/test_gt_phase_{i}.pt")

        style_images=torch.sqrt(style_holo).to(device).float().detach()
        content_images=torch.sqrt(content_holo).to(device).float().detach()

        distance_style = distance_style.to(args.device).float()

    
        vis_idx+=1
        gt_phase_tmp = gt_phase.to(device)
        gt_amp_tmp = gt_amplitude.to(device)
        _, _, distance_content_pred = field_retrieval(network, content_images, style_vector, 1.0, True)
        
        distance_list.append([distance_content.squeeze().item(), distance_content_pred.squeeze().item()])
        
        with torch.no_grad():
            amplitude, phase, _ = field_retrieval(network, content_images, style_vector, 1.0, True)
            amp_foc, ph_foc = model_forward(amplitude, phase*args.phase_normalize, -distance_style-2*args.distance_normalize_constant, return_field=True, unwrap=True)
            
            
            gt_phase_tmp -= torch.mean(gt_phase_tmp)
            ph_foc -= torch.mean(ph_foc)
            phase -= torch.mean(phase)
            
            
        if vis_idx%1 == 0:
            inputs = torch.cat([content_images, style_images], dim=2)
            recon_field = torch.cat([amplitude.cpu(), phase.cpu()], dim=2)
            gt_field =torch.cat([gt_amp_tmp, gt_phase_tmp], dim=2)
            recon_foc_field = torch.cat([amp_foc.detach().cpu(), ph_foc.cpu()], dim=2)

            #modified below
            video_array.append(ph_foc.cpu())
            
            args.save_ext = ".png"

            inputs_output_name = output_dir / 'inputs{:d}{:s}'.format(vis_idx, args.save_ext)
            recon_field_output_name = output_dir / 'recon_field{:d}{:s}'.format(vis_idx, args.save_ext)
            gt_field_output_name = output_dir / 'gt_field{:d}{:s}'.format(vis_idx, args.save_ext)
            ph_foc_name = output_dir / 'ph_foc{:d}{:s}'.format(vis_idx, args.save_ext)
            recon_foc_field_output_name = output_dir / 'recon_foc_field{:d}{:s}'.format(vis_idx, args.save_ext)
            lastcolum_output_name = output_dir / 'lastcolum{:d}{:s}'.format(vis_idx, args.save_ext)

            save_image(inputs, str(inputs_output_name))
            save_image(recon_field, str(recon_field_output_name))
            save_image(gt_field, str(gt_field_output_name))
            save_image(recon_foc_field, str(recon_foc_field_output_name))
            save_image(torch.abs(gt_field.cpu()-recon_foc_field.cpu()), str(lastcolum_output_name))
            save_image(torch.cat([content_images.cpu(), ph_foc.cpu()], dim=3), str(ph_foc_name))

distance_list = (np.array(distance_list)+args.distance_normalize_constant)*args.distance_normalize*1000
import csv
f = open(output_dir / "distance_prediction.csv", "w")
writer = csv.writer(f)
writer.writerows(distance_list)
f.close()

distance_true, distance_pred = distance_list[:, 0], distance_list[:, 1]
distance = {round(n, 2):m for n,m in zip(distance_true, distance_pred)}
mean_list, std_list = [], []
for key in sorted(list(distance.keys())):
    mean_list.append(np.mean(distance[key]))
    std_list.append(np.std(distance[key]))
plt.figure(figsize=(6, 12))

# plt.scatter(distance_pred, distance_true, s=4)
# plt.errorbar(mean_list, sorted(list(distance.keys())), xerr=std_list, fmt = '.')
# plt.errorbar(sorted(list(distance.keys())), mean_list, yerr=std_list, fmt = '.')
# plt.violinplot([[distance[key]] for key in sorted(list(distance.keys()))])
plt.savefig(output_dir / 'distance_prediction.eps')
plt.savefig(output_dir / 'distance_prediction.png')
plt.close()
from sklearn.metrics import r2_score
print("R2 score: ", r2_score(distance_true, distance_pred))


# Code for video generation #
# Do not remove!
# else:

import numpy as np
import cv2
import os
fps = 10
duration = 10
width, height = (256, 128)
channel=3
out = cv2.VideoWriter(
    f"black2.avi", cv2.VideoWriter_fourcc(*"DIVX"), fps, (width, height)
)

for i in range(100):
    ph_foc_name = output_dir / 'ph_foc{:d}{:s}'.format(i+1, args.save_ext)
    img = cv2.imread(ph_foc_name)
    # text = 'testing 123'
    font = cv2.FONT_HERSHEY_TRIPLEX
    cv2.putText(img, 'Contet', (40, 15), font, 0.4, (255, 255, 255), 1)
    cv2.putText(img, 'Reconstructed', (20+128, 15), font, 0.4, (255, 255, 255), 1)
    out.write(img)
out.release()
