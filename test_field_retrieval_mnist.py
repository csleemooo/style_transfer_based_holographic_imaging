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
from torchmetrics.image import PeakSignalNoiseRatio as PSNR
from torchmetrics.regression import MeanAbsoluteError

import utils
from utils.Forward_model import Holo_Generator
from utils.functions import unwrap, tv_loss, field_retrieval
from utils.Data_loader import *
from function import adaptive_instance_normalization, coral, calc_mean_std

from math import pi

import warnings
warnings.filterwarnings("ignore")

cudnn.benchmark = True
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
ImageFile.LOAD_TRUNCATED_IMAGES = True

parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--data_name', type=str, default='MNIST')
parser.add_argument('--decoder_root', default='./models/MNIST', type=str, 
                    help='Directory path to a batch of style images')
parser.add_argument('--vgg', type=str, default='./models/vgg_normalised.pth')
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--exp_name', type=str, default='MNIST_test')
# training options
parser.add_argument('--save_dir', default='./output',
                    help='Directory to save the model')
parser.add_argument('--save_ext', default='.png',
                    help='Directory to save the log')
parser.add_argument('--img_size', type=int, default=128)
parser.add_argument('--n_layer', type=int, default=4)
parser.add_argument('--unknown_distance', type=int, default=1)

parser.add_argument('--wavelength', type=float, default=532e-9)
parser.add_argument('--pixel_size', type=float, default=1.5e-6)
parser.add_argument('--phase_normalize', type=float, default=1.0)
parser.add_argument('--distance_normalize', type=float, default=1.0)
parser.add_argument('--distance_normalize_constant', type=float, default=0.0)

args = parser.parse_args()

device = torch.device(args.device)
args.save_dir = args.save_dir + '/%s'%args.exp_name
output_dir = Path(args.save_dir)
output_dir.mkdir(exist_ok=True, parents=True)

import copy

#### Trained model parameters / extracted representative style vector ####
args.decoder = os.path.join(args.decoder_root, 'decoder_iter_80000.pth.tar')
args.decoder_ph = os.path.join(args.decoder_root, 'decoder_ph_iter_80000.pth.tar')
args.distance_g = os.path.join(args.decoder_root, 'distance_g_iter_80000.pth.tar')
args.style_path = os.path.join('./style_representation', 'MNIST', 'style_vector.pt')


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
vgg = nn.Sequential(*list(vgg.children())[:31])

network = net.Net(vgg, decoder, decoder_ph, distance_G)
network.eval()
network.to(device)
model_forward = Holo_Generator(args).to(device)  # ASM, free-space propagator
##########################################################################

################     Representative style vector     #####################
if os.path.isfile(args.style_path):
    style_vector = torch.load(args.style_path)
else:
    raise FileNotFoundError
style_vector = style_vector.to(args.device).float()
##########################################################################

psnr = PSNR().to(device)
mae = MeanAbsoluteError()

psnr_list, mae_list, distance_list = [], [], []

vis_idx = 0
for i in range(20):
        
    content_holo = torch.load(f"./test_data/test_content_holo_{i}.pt")
    distance_style = torch.load(f"./test_data/test_distance_style_{i}.pt")
    distance_content = torch.load(f"./test_data/test_distance_content_{i}.pt")
    gt_amplitude = torch.load(f"./test_data/test_gt_amplitude_{i}.pt")
    gt_phase = torch.load(f"./test_data/test_gt_phase_{i}.pt").to(device)

    content_images=torch.sqrt(content_holo).to(device).float().detach()
    distance_style = distance_style.to(args.device).float()
    batch_size = content_images.shape[0]
    
    with torch.no_grad():
        
        amp_field, ph_field, distance_content_pred = field_retrieval(network, content_images, style_vector, 1.0, True)
        amp_foc, ph_foc = model_forward(amp_field, ph_field*args.phase_normalize, -distance_style-2*args.distance_normalize_constant, return_field=True, unwrap=True)

        gt_phase -= torch.mean(gt_phase, dim=(-2, -1), keepdim=True).expand_as(gt_amplitude)
        ph_field -= torch.mean(ph_field, dim=(-2, -1), keepdim=True).expand_as(amp_field)
        ph_foc -= torch.mean(ph_foc, dim=(-2, -1), keepdim=True).expand_as(amp_foc)

        psnr_list.append(psnr(ph_foc, gt_phase).item())
        mae_list.append(mae(ph_foc, gt_phase).item())
        
        for j in range(batch_size):
            vis_idx+=1
        
            distance_list.append([distance_content.squeeze()[j].item(), distance_content_pred[j].squeeze().item()])
                        
            top = torch.cat([content_images[j:j+1].cpu(), amp_field[j:j+1].cpu(), gt_amplitude[j:j+1].cpu(), amp_foc[j:j+1].cpu()], dim=3)
            bot = torch.cat([torch.zeros_like(content_images[j:j+1]), ph_field[j:j+1].cpu(), gt_phase[j:j+1], ph_foc[j:j+1].cpu()], dim=3)
            save_image(torch.cat([top, bot], dim=2), str(output_dir / '{:d}_test{:s}'.format(vis_idx, args.save_ext)))
    
else:
    print("Mean PSNR: ", np.mean(psnr_list))
    print("Mean MAE: ", np.mean(mae_list))
    distance_list = (np.array(distance_list)+args.distance_normalize_constant)*args.distance_normalize*1000
    
    distance_true, distance_pred = distance_list[:, 0], distance_list[:, 1]
    distance = {np.round(n, 2):[] for n in np.unique(distance_true)}
    for n, m in zip(distance_true, distance_pred):
        distance[np.round(n, 2)].append(m)

    plt.figure(figsize=(12, 3))
    plt.boxplot(list(distance.values()), vert=0)
    plt.savefig(output_dir / 'distance_prediction.png')
    plt.close()
    from sklearn.metrics import r2_score
    print("R2 score: ", r2_score(distance_true, distance_pred))

