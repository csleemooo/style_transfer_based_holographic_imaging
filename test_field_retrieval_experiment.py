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
import scipy.io as sio

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
parser.add_argument('--auto_enc', type=str, default='./experiments/polystyrene_bead/autoencoder_kernel_7_autoencoder/encoder_iter_80000.pth.tar')
parser.add_argument('--autoencoder', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--n_layer', type=int, default=4)
# parser.add_argument('--style_weight', type=float, default=10.0)
parser.add_argument('--unknown_distance', type=int, default=0)  # unknown distance for 1, known distance for 0 
args = parser.parse_args()
# test code #
## experimental paramter for holography ##
args.wavelength = 532e-9

args.data_name = 'polystyrene_bead'
args.decoder_root = 'experiments/polystyrene_bead/single_style_diff_object_field_retrieval'
# args.decoder_root = 'experiments/polystyrene_bead/single_style_disc_vgg_4layer_large_depth_field_retrieval'
# args.data_name = 'red_blood_cell'
# args.decoder_root = './experiments/red_blood_cell/single_style_disc_instance_norm_field_retrieval'

args.exp_name = '250619_diff_object_single'

# args.decoder_root ='./experiments/MNIST/single_style_disc_batch16_field_retrieval'
args.decoder = os.path.join(args.decoder_root, 'decoder_iter_80000.pth.tar')
args.decoder_ph = os.path.join(args.decoder_root, 'decoder_ph_iter_80000.pth.tar')
args.distance_g = os.path.join(args.decoder_root, 'distance_g_iter_80000.pth.tar')
args.style_path = os.path.join('./style_representation', args.data_name, 'style_vector.pt')


device = torch.device(args.device)
args.save_dir = args.save_dir + '/%s/%s_field_retrieval'%(args.data_name, args.exp_name)
output_dir = Path(args.save_dir)
output_dir.mkdir(exist_ok=True, parents=True)

import copy
# call model
if args.n_layer==4:
    decoder = net.decoder
    decoder_ph = copy.deepcopy(net.decoder)
else:
    decoder = net.decoder_large
    decoder_ph = copy.deepcopy(net.decoder_large)
distance_G = net.Distance_G()
# load trained hyperparameter
decoder.load_state_dict(torch.load(args.decoder))
decoder_ph.load_state_dict(torch.load(args.decoder_ph))
distance_G.load_state_dict(torch.load(args.distance_g))

if args.autoencoder:
    vgg = args.auto_enc
else:
    vgg = net.vgg

    state_dict = torch.load(args.vgg)
    state_dict['0.weight'] = state_dict['0.weight'].sum(dim=1, keepdim=True)
    vgg.load_state_dict(state_dict)

    vgg_layer = 31 if args.n_layer==4 else 44
    vgg = nn.Sequential(*list(vgg.children())[:vgg_layer])


args.image_size = 256 if args.data_name == 'polystyrene_bead' or args.data_name == 'red_blood_cell' else 128

network = net.Net(vgg, decoder, decoder_ph, distance_G)

network.eval()
network.to(device)
    
if args.data_name == 'polystyrene_bead':
    output_dir_images = output_dir / 'images'
    output_dir_images.mkdir(exist_ok=True, parents=True)

    output_dir_mat = output_dir / 'mat'
    output_dir_mat.mkdir(exist_ok=True, parents=True)
    if 'single' in args.exp_name:
        train_holo_list_style = [7]
        # train_holo_list_content = [round(float(i), 3) for i in np.arange(9, 25, 2)]
        train_holo_list_content = [round(float(i), 3) for i in np.arange(9, 13, 1)]
    else:
        train_holo_list_style = [round(float(i), 3) for i in np.arange(7, 11, 1)]
        train_holo_list_content = [round(float(i), 3) for i in np.arange(11, 14, 1)]
            
    args.distance_min = min(train_holo_list_style)
    args.distance_max = max(train_holo_list_content)
    args.phase_normalize = 2*pi
    
    transform_img = transforms.Compose([transforms.ToTensor()])
    dataset = Holo_loader(root='/mnt/mooo/CS/style transfer based holographic imaging/data/polystyrene_bead_holo_only', image_set='train', transform=transform_img, holo_list=train_holo_list_style, return_distance=True)    
    dataset_style = iter(DataLoader(dataset, batch_size=args.batch_size, shuffle=False))
    if 'diff_object' in args.exp_name:
        train_holo_list_content = [1]
        args.distance_min = 20
        args.distance_max = 26
        dataset = Holo_loader(root='/mnt/mooo/CS/style transfer based holographic imaging/data/red_blood_cell', image_set='test', transform=transform_img, holo_list=train_holo_list_content, return_distance=True)
        dataset_content = iter(DataLoader(dataset, batch_size=args.batch_size, shuffle=True))
    else:
        dataset = Holo_loader(root='/mnt/mooo/CS/style transfer based holographic imaging/data/polystyrene_bead_test', image_set='test', transform=transform_img, holo_list=train_holo_list_content, return_distance=True)    
        dataset_content = iter(DataLoader(dataset, batch_size=args.batch_size, shuffle=True))
    
    # dataset_content = iter(DataLoader(dataset, batch_size=args.batch_size, shuffle=False))
    args.pixel_size = 6.5e-6
    args.distance_normalize = args.distance_max - args.distance_min
    args.distance_normalize_constant = args.distance_min/args.distance_normalize
    test_num = len(dataset_content)

    
elif args.data_name == 'red_blood_cell':
    if 'single' in args.exp_name:
        train_holo_list_style = [22.2]
        train_holo_list_content = [6]
    rbc_id = "%d"%train_holo_list_content[0]
    output_dir_images = output_dir / rbc_id /'images'
    output_dir_images.mkdir(exist_ok=True, parents=True)

    output_dir_mat = output_dir / rbc_id / 'mat'
    output_dir_mat.mkdir(exist_ok=True, parents=True)
    
    args.distance_min = 20
    args.distance_max = 26
    args.distance_normalize = args.distance_max - args.distance_min
    args.distance_normalize_constant = args.distance_min/args.distance_normalize
    args.phase_normalize = 2*pi
    
    transform_img = transforms.Compose([transforms.ToTensor()])
    dataset = Holo_loader(root='/mnt/mooo/CS/style transfer based holographic imaging/data/red_blood_cell', image_set='train', transform=transform_img, holo_list=train_holo_list_style, return_distance=True)    
    dataset_style = iter(DataLoader(dataset, batch_size=args.batch_size, shuffle=True))
    dataset = Holo_loader(root='/mnt/mooo/CS/style transfer based holographic imaging/data/red_blood_cell', image_set='test', transform=transform_img, holo_list=train_holo_list_content, return_distance=True)    
    dataset_content = iter(DataLoader(dataset, batch_size=args.batch_size, shuffle=False))
    args.pixel_size = 6.5e-6
    test_num = len(dataset_content)
    
model_forward = Holo_Generator(args).to(device)  # ASM, free-space propagator

from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics.image import PeakSignalNoiseRatio as PSNR
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure as MSSSIM
from torchmetrics.regression import MeanAbsoluteError

ssim = SSIM()
msssim = MSSSIM(data_range=1.0, betas=(0.0448, 0.2856, 0.3001, 0.2363, 0.1333))
psnr = PSNR()
mae = MeanAbsoluteError()

ssim_list, psnr_list, mae_list, msssim_list, distance_list = [], [], [], [], []
vis_idx = 0

if os.path.isfile(args.style_path):
    style_vector = torch.load(args.style_path)
else:
    from function import calc_mean_std
    style_mean, style_std = [], []
    for t in range(len(dataset_style)):
        feat = network.encode(next(dataset_style)[0].float().sqrt().to(device))
        mean, std = calc_mean_std(feat)
        style_mean.append(mean)
        style_std.append(std)
    else:
        style_mean = torch.cat(style_mean, dim=0).mean(dim=0, keepdim=True)
        style_std = torch.cat(style_std, dim=0).mean(dim=0, keepdim=True)
        style_vector = torch.cat([style_mean, style_std], dim=0)
        
    torch.save(style_vector.detach().cpu(), args.style_path)
    
    if args.data_name == 'polystyrene_bead':
        dataset = Holo_loader(root='/mnt/mooo/CS/style transfer based holographic imaging/data/polystyrene_bead_holo_only', image_set='train', transform=transform_img, holo_list=train_holo_list_style, return_distance=True)    
        dataset_style = iter(DataLoader(dataset, batch_size=args.batch_size, shuffle=True))
        
    else:
        dataset = Holo_loader(root='/mnt/mooo/CS/style transfer based holographic imaging/data/red_blood_cell', image_set='train', transform=transform_img, holo_list=train_holo_list_style, return_distance=True)    
        dataset_style = iter(DataLoader(dataset, batch_size=args.batch_size, shuffle=True))

style_vector = style_vector.to(device)

for i in tqdm(range(test_num)):
    [content_holo, distance_content, gt_amplitude, gt_phase] = next(dataset_content)
    [style_holo, distance_style] = next(dataset_style)
    # style_holo = style_holo[0].unsqueeze(0).repeat(args.batch_size, 1, 1, 1)
    distance_style = -args.distance_normalize_constant + distance_style.reshape([-1, 1, 1, 1]).to(args.device).float()/args.distance_normalize
    distance_content = -args.distance_normalize_constant + distance_content.reshape([-1, 1, 1, 1]).to(args.device).float()/args.distance_normalize
    
    style_images=torch.sqrt(style_holo).to(device).float().detach() #.repeat(1, 3, 1, 1)
    content_images=torch.sqrt(content_holo).to(device).float().detach() #.repeat(1, 3, 1, 1)

    
    amplitude_pred, phase_pred, distance_content_pred = field_retrieval(network, content_images, style_vector, 1.0, True)
    
    for j in range(content_holo.shape[0]):
        amplitude = amplitude_pred[j:j+1]
        phase = phase_pred[j:j+1]
        vis_idx+=1
        if not args.data_name == 'red_blood_cell' and not 'diff_object' in args.exp_name:
            gt_phase_tmp = gt_phase[j:j+1]
            gt_amp_tmp = gt_amplitude[j:j+1]
        with torch.no_grad():            
            
            amp_foc, ph_foc = model_forward(amplitude, phase*args.phase_normalize, -distance_style[j:j+1]-2*args.distance_normalize_constant, return_field=True, unwrap=True)

            ph_foc -= torch.mean(ph_foc)
            phase -= torch.mean(phase)

            if not args.data_name == 'red_blood_cell' and not 'diff_object' in args.exp_name:
                gt_phase_tmp -= torch.mean(gt_phase_tmp)
       
                psnr_list.append(psnr(ph_foc, gt_phase_tmp).to(device).item())
                mae_list.append(mae(ph_foc, gt_phase_tmp).to(device).item())
                ssim_list.append(ssim(ph_foc, gt_phase_tmp).to(device).item())
                msssim_list.append(msssim(ph_foc, gt_phase_tmp).to(device).item())
                gt_phase_tmp = gt_phase_tmp.detach().cpu()
                
            phase = phase.detach().cpu()*args.phase_normalize
            ph_foc = ph_foc.detach().cpu()
            
            
                
        if vis_idx%1 == 0:
            inputs = torch.cat([content_images[j:j+1], style_images[j:j+1]], dim=2).detach().cpu()
            recon_field = torch.cat([amplitude.detach().cpu(), phase], dim=2)
            recon_foc_field = torch.cat([amp_foc.detach().cpu(), ph_foc], dim=2)
            if not args.data_name == 'red_blood_cell' and not 'diff_object' in args.exp_name:
                gt_field =torch.cat([gt_amp_tmp, gt_phase_tmp], dim=2).detach().cpu()
                total = torch.cat([inputs, recon_field, gt_field, recon_foc_field, torch.abs(gt_field-recon_foc_field)], dim=3)
                plt.imsave(output_dir_images / '{:d}_gt_phase.png'.format(vis_idx), gt_phase_tmp.detach().numpy().squeeze(), vmin=-0.5, vmax=2.5, cmap='gray')
                plt.imsave(output_dir_images / '{:d}_gt_amplitude.png'.format(vis_idx), gt_amp_tmp.detach().numpy().squeeze(), vmin=0, vmax=1.0, cmap='gray')
                plt.imsave(output_dir_images / '{:d}_re_phase.png'.format(vis_idx), ph_foc.detach().numpy().squeeze(), vmin=-0.5, vmax=2.5, cmap='gray')
            else:
                total = torch.cat([inputs, recon_field, recon_foc_field], dim=3)
                plt.imsave(output_dir_images / '{:d}_re_phase.png'.format(vis_idx), ph_foc.detach().numpy().squeeze(), vmin=-0.0, vmax=2, cmap='hot')
                
            output_name = output_dir / 'result_field{:d}{:s}'.format(vis_idx, args.save_ext)
            
            plt.imsave(output_dir_images / '{:d}_re_field_amplitude.png'.format(vis_idx), amplitude.detach().cpu().numpy().squeeze(), vmin=0, vmax=1.0, cmap='gray')
            plt.imsave(output_dir_images / '{:d}_re_field_phase.png'.format(vis_idx), phase.detach().cpu().numpy().squeeze(), vmin=-0.5, vmax=0.5, cmap='gray')
            
            plt.imsave(output_dir_images / '{:d}_re_amplitude.png'.format(vis_idx), amp_foc.cpu().detach().numpy().squeeze(), vmin=0, vmax=1.0, cmap='gray')
            plt.imsave(output_dir_images / '{:d}_style.png'.format(vis_idx), style_images[j:j+1].cpu().detach().numpy().squeeze(), vmin=0, vmax=1.0, cmap='gray')
            plt.imsave(output_dir_images / '{:d}_content.png'.format(vis_idx), content_images[j:j+1].cpu().detach().numpy().squeeze(), vmin=0, vmax=1.0, cmap='gray')
            
            if args.data_name == 'polystyrene_bead' not 'diff_object' in args.exp_name:
                fov_list = [8, 7, 9, 4, 14, 10, 5, 11, 13, 6, 1, 15, 3, 12, 2, 16]
                fov_name = 'fov%d'%(fov_list[vis_idx%16-1])
                data = {'amplitude': amp_foc.cpu().detach().numpy().squeeze(), 'phase': ph_foc.cpu().detach().numpy().squeeze()}
                xxxx = (distance_content.squeeze()[j].item()+args.distance_normalize_constant)*args.distance_normalize
                sio.savemat(output_dir_mat / '{:s}_recon_{:d}.mat'.format(fov_name, int(round(xxxx))), data)
                
            elif args.data_name == 'red_blood_cell' or 'diff_object' in args.exp_name:
                data = {'amplitude': amp_foc.cpu().detach().numpy().squeeze(), 'phase': ph_foc.detach().numpy().squeeze()}
                sio.savemat(output_dir_mat / 'recon_{:d}.mat'.format(vis_idx), data)
    else:
        # _, _, distance_content_pred = field_retrieval(network, content_images, style_images, 1.0, True)
        # print(distance_content_pred, content_images.shape, style_images.shape)
        tmp = [[distance_content.squeeze()[i].item(), distance_content_pred.squeeze()[i].item()] for i in range(content_holo.shape[0])]
        distance_list.extend(tmp) 
else:
    if args.data_name == 'polystyrene_bead' and not 'diff_object' in args.exp_name::
        print("Mean PSNR: %1.3f, %1.4f"%(np.mean(psnr_list), np.std(psnr_list)))
        print("Mean MAE: %1.3f, %1.4f"%(np.mean(mae_list), np.std(mae_list)))
        print("Mean MS-SSIM: %1.3f, %1.4f"%(np.mean(msssim_list), np.std(msssim_list)))
        print("Mean SSIM: %1.3f, %1.4f"%(np.mean(ssim_list), np.std(ssim_list)))
    
        distance_list = (np.array(distance_list)+args.distance_normalize_constant)*args.distance_normalize
        distance_true, distance_pred = distance_list[:, 0], distance_list[:, 1]
        from sklearn.metrics import r2_score
        print("R2 score: ", r2_score(distance_true, distance_pred))
        
        # f = open(output_dir / "distance_prediction.csv", "w")
        # writer = csv.writer(f)
        # writer.writerows(distance_list)
        # f.close()
        
        distance = {round(n, 2):[] for n in train_holo_list_content}
        for n, m in zip(distance_true, distance_pred):
            distance[round(n, 2)].append(m)
            
        mean_list, std_list = [], []
        for key in sorted(list(distance.keys())):
            mean_list.append(np.mean(distance[key]))
            std_list.append(np.std(distance[key]))
        print(sorted(list(distance.keys())), mean_list, std_list)
        plt.figure(figsize=(12, 3))
        plt.boxplot(list(distance.values()), vert=0)
        # plt.errorbar(sorted(list(distance.keys())), mean_list, yerr=std_list, fmt='.')
        plt.savefig(output_dir / 'distance_prediction.png', bbox_inches='tight')
        plt.savefig(output_dir / 'distance_prediction.eps', bbox_inches='tight')
        plt.close()
        