import torch.utils.data as data
from torch.utils.data import Dataset
import numpy as np
import torch
import os
from typing import Any, Callable, List, Optional, Tuple
from torchvision import transforms
import torch.nn.functional as F

def mnist_loader(args, dataset, train_holo_list_style, train_holo_list_content, model_forward, device):
    transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
                                    transforms.RandomAffine(degrees=0, translate=[0.1, 0.1], fill=0.0)])

    style_images = next(iter(data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)))[0].to(device)
    distance_style = torch.Tensor([train_holo_list_style[np.random.randint(low=0, high=len(train_holo_list_style),
                                                                        size=1)[0]] for _ in range(args.batch_size)]).reshape([-1, 1, 1, 1]).to(device).float()
    distance_content = torch.Tensor([train_holo_list_content[np.random.randint(low=0, high=len(train_holo_list_content),
                                                                        size=1)[0]] for _ in range(args.batch_size)]).reshape([-1, 1, 1, 1]).to(device).float()

    distance_style = -args.distance_normalize_constant + distance_style/args.distance_normalize
    distance_content = -args.distance_normalize_constant + distance_content/args.distance_normalize
    
    phase_style = (F.pad(style_images, (32, 32, 32, 32), 'constant', 0)).to(device).float()
    amplitude = torch.ones_like(phase_style).to(device).float() * 0.6
    phase_content = transform(phase_style)
    phase_style = transform(phase_style)

    style_holo = model_forward(amplitude, phase_style, distance_style).to(device).float().detach()
    content_holo = model_forward(amplitude, phase_content, distance_content).to(device).float().detach()
    
    return style_holo, content_holo, distance_style, distance_content

def mnist_loader_test(args, dataset, holo_list_style, holo_list_content, model_forward, device, test_interpolation=False):
    transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
                                    transforms.RandomAffine(degrees=0, translate=[0.1, 0.1], fill=0.0)])

    style_images = next(iter(data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)))[0].to(device)
    if test_interpolation:
        distance_style = torch.Tensor(holo_list_style).reshape([-1, 1, 1, 1]).to(device).float()
    else:
        distance_style = torch.Tensor([holo_list_style[np.random.randint(low=0, high=len(holo_list_style),
                                                                               size=1)[0]] for _ in range(args.batch_size)]).reshape([-1, 1, 1, 1]).to(device).float()
    distance_content = torch.Tensor([holo_list_content[np.random.randint(low=0, high=len(holo_list_content),
                                                                        size=1)[0]] for _ in range(args.batch_size)]).reshape([-1, 1, 1, 1]).to(device).float()

    distance_style = -args.distance_normalize_constant + distance_style/args.distance_normalize
    distance_content = -args.distance_normalize_constant + distance_content/args.distance_normalize
    
    phase_style = (F.pad(style_images, (32, 32, 32, 32), 'constant', 0)).to(device).float()
    amplitude = torch.ones_like(phase_style).to(device).float() * 0.6
    phase_content = transform(phase_style)
    phase_style = transform(phase_style)
    
    if test_interpolation:
        b = distance_style.shape[0]
        style_holo = model_forward(amplitude, phase_style.repeat(b, 1, 1, 1), distance_style).to(device).float().detach()
        content_holo = model_forward(amplitude, phase_content.repeat(b, 1, 1, 1), distance_content.repeat(b, 1, 1, 1)).to(device).float().detach()
    else:
        style_holo = model_forward(amplitude, phase_style, distance_style).to(device).float().detach()
        content_holo = model_forward(amplitude, phase_content, distance_content).to(device).float().detach()
        
    return style_holo, content_holo, distance_style, distance_content


class Holo_loader(Dataset):
    
    def __init__(self,
                 root: str,
                 image_set: str="train",
                 transform: Optional[Callable] = None,
                 holo_list: list=None,
                 return_distance: bool=None,
    ) -> None:

        self.transform = transform
        self.return_distance = return_distance
        self.data_list=[]
        self.root_list=[]

        self.data_root = os.path.join(root, image_set, 'holography')

        for d in holo_list:
            tmp = [os.path.join('%d'%d, j) for j in os.listdir(os.path.join(self.data_root, '%d'%d))]
            print('the # of diffraction patterns measured at %dmm: %d'%(d, len(tmp)))
            self.data_list.extend(tmp)

        self.data_list = np.array(self.data_list)
        self.data_num = len(self.data_list)

    def __len__(self) -> int:
        return self.data_num

    def __getitem__(self, index: int):

        pth = os.path.join(self.data_root, self.data_list[index])
        distance =self.data_list[index].split("/")[0]
        distance = float(distance) if '.' in distance else int(distance)
        holo = self.load_matfile(pth)['holography']

        if self.return_distance:
            if self.transform is not None:
                holo = self.transform(holo)
                distance = torch.Tensor([distance])

            return holo, distance
        else:
            if self.transform is not None:
                holo = self.transform(holo)

            return holo


    def load_matfile(self, path):
        import scipy.io as sio
        data = sio.loadmat(path)
        return data