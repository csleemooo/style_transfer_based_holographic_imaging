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
from torchmetrics.image import PeakSignalNoiseRatio as PSNR
from torchmetrics.regression import MeanAbsoluteError

import utils
from utils.Forward_model import Holo_Generator
from utils.functions import unwrap, tv_loss, field_retrieval
from utils.Data_loader import *

from math import pi

import warnings
warnings.filterwarnings("ignore")

cudnn.benchmark = True
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
# Disable OSError: image file is truncated
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
parser.add_argument('--save_ext', default='.jpg',
                    help='Directory to save the log')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--img_size', type=int, default=128)
parser.add_argument('--n_layer', type=int, default=4)
parser.add_argument('--unknown_distance', type=int, default=1)
args = parser.parse_args()

# Trained model parameters root
args.decoder = os.path.join(args.decoder_root, 'decoder_iter_80000.pth.tar')
args.decoder_ph = os.path.join(args.decoder_root, 'decoder_ph_iter_80000.pth.tar')
args.distance_g = os.path.join(args.decoder_root, 'distance_g_iter_80000.pth.tar')
args.style_path = os.path.join('./style_representation', 'MNIST', 'style_vector.pt')

device = torch.device(args.device)
args.save_dir = args.save_dir + '/%s'%args.exp_name
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
vgg = nn.Sequential(*list(vgg.children())[:31])


network = net.Net(vgg, decoder, decoder_ph, distance_G)

network.eval()
network.to(device)


train_holo_list_style = [0.2]
train_holo_list_content = [round(float(i), 3) for i in np.arange(0.4, 0.9, 0.1)]

args.wavelength = 532e-9
args.pixel_size = 1.5e-6
args.phase_normalize = 1
args.distance_normalize = 1.0
args.distance_normalize_constant = 0

model_forward = Holo_Generator(args).to(device)  # ASM, free-space propagator


psnr = PSNR().to(device)
mae = MeanAbsoluteError()

ssim_list, psnr_list, mae_list, msssim_list = [], [], [], []
distance_list= []
vis_idx = 0
video_array = []
distance_per_psnr, distance_per_mae = {i:[] for i in train_holo_list_content}, {i:[] for i in train_holo_list_content}

from function import calc_mean_std

if os.path.isfile(args.style_path):
    style_vector = torch.load(args.style_path)
else:
    raise FileNotFoundError
        
style_vector = style_vector.to(args.device).float()
import test_cs_vid2

for i in range(20):
        
    content_holo = torch.load(f"/mnt/mooo/fakhriyya/new_code/style_transfer_based_holographic_imaging/new_dataset_241122/test_content_holo_{i}.pt")
    distance_style = torch.load(f"/mnt/mooo/fakhriyya/new_code/style_transfer_based_holographic_imaging/new_dataset_241122/test_distance_style_{i}.pt")
    distance_content = torch.load(f"/mnt/mooo/fakhriyya/new_code/style_transfer_based_holographic_imaging/new_dataset_241122/test_distance_content_{i}.pt")
    gt_amplitude = torch.load(f"/mnt/mooo/fakhriyya/new_code/style_transfer_based_holographic_imaging/new_dataset_241122/test_gt_amplitude_{i}.pt")
    gt_phase = torch.load(f"/mnt/mooo/fakhriyya/new_code/style_transfer_based_holographic_imaging/new_dataset_241122/test_gt_phase_{i}.pt")

    content_images=torch.sqrt(content_holo).to(device).float().detach()


    distance_style = distance_style.to(args.device).float()

    
    for j in range(args.batch_size):
        vis_idx+=1
        gt_phase_tmp = gt_phase[j:j+1]
        gt_amp_tmp = gt_amplitude[j:j+1]
        _, _, distance_content_pred = field_retrieval(network, content_images, style_vector, 1.0, True)
        
        tmp = [[distance_content.squeeze()[i].item(), distance_content_pred.squeeze()[i].item()] for i in range(args.batch_size)]
        
        distance_list.extend(tmp)
        with torch.no_grad():
            amplitude, phase, _ = field_retrieval(network, content_images[j:j+1], style_vector, 1.0, True)
            amp_foc, ph_foc = model_forward(amplitude, phase*args.phase_normalize, -distance_style[j:j+1]-2*args.distance_normalize_constant, return_field=True, unwrap=True)
            
            gt_phase_tmp -= torch.mean(gt_phase_tmp)
            ph_foc -= torch.mean(ph_foc)
            phase -= torch.mean(phase)
            
            psnr_list.append(psnr(ph_foc, gt_phase_tmp).to(device).item())
            mae_list.append(mae(ph_foc, gt_phase_tmp).to(device).item())
            
            
            inputs = torch.cat([content_images[j:j+1], style_images[j:j+1]], dim=2)
            recon_field = torch.cat([amplitude.cpu(), phase.cpu()], dim=2)
            gt_field =torch.cat([gt_amp_tmp, gt_phase_tmp], dim=2)
            recon_foc_field = torch.cat([amp_foc.detach().cpu(), ph_foc.cpu()], dim=2)
            

            
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
            save_image(torch.cat([content_images[j:j+1].cpu(), ph_foc.cpu()], dim=3), str(ph_foc_name))

else:
    print("Mean PSNR: ", np.mean(psnr_list))
    print("Mean MAE: ", np.mean(mae_list))
    print("Mean MS-SSIM: ", np.mean(msssim_list))
    print("Mean SSIM: ", np.mean(ssim_list))
    distance_list = (np.array(distance_list)+args.distance_normalize_constant)*args.distance_normalize*1000
    import csv
    
    f = open(output_dir / "distance_prediction.csv", "w")
    writer = csv.writer(f)
    writer.writerows(distance_list)
    f.close()
    
    distance_true, distance_pred = distance_list[:, 0], distance_list[:, 1]
    distance = {np.round(n, 2):[] for n in np.unique(distance_true)}
    for n, m in zip(distance_true, distance_pred):
        distance[np.round(n, 2)].append(m)
    mean_list, std_list = [], []
    for key in sorted(list(distance.keys())):
        mean_list.append(np.mean(distance[key]))
        std_list.append(np.std(distance[key]))
    print(sorted(list(distance.keys())), mean_list, std_list)
    plt.figure(figsize=(12, 3))
    
    # plt.scatter(distance_pred, distance_true, s=4)
    # plt.errorbar(mean_list, sorted(list(distance.keys())), xerr=std_list, fmt = '.')
    plt.boxplot(list(distance.values()), vert=0)
    # plt.errorbar(sorted(list(distance.keys())), mean_list, yerr=std_list, fmt = '.')
    # plt.violinplot([[distance[key]] for key in sorted(list(distance.keys()))])
    plt.savefig(output_dir / 'distance_prediction.eps')
    plt.savefig(output_dir / 'distance_prediction.png')
    plt.close()
    from sklearn.metrics import r2_score
    print("R2 score: ", r2_score(distance_true, distance_pred))
    psnr_mean, psnr_std = [], []
    mae_mean, mae_std = [], []
    for key in distance_per_psnr.keys():
        psnr_mean.append(np.mean(distance_per_psnr[key]))
        psnr_std.append(np.std(distance_per_psnr[key]))
        mae_mean.append(np.mean(distance_per_mae[key]))
        mae_std.append(np.std(distance_per_mae[key]))
    else:
        print(psnr_mean, psnr_std)
        print(mae_mean, mae_std)
    
    exit()

