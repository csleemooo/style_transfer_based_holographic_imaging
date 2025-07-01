from .Angular_Spectrum_Method import ASM
from torch import nn
import torch.nn.functional as F
import torch
from utils.functions import unwrap as uw
class Holo_Generator(nn.Module):
    def __init__(self, args):
        super(Holo_Generator, self).__init__()
        self.wavelength = args.wavelength
        self.pixel_size = args.pixel_size
        self.distance_normalize = args.distance_normalize
        self.distance_normalize_constant = args.distance_normalize_constant
        self.phase_normalize = args.phase_normalize


    def forward(self, amplitude, phase, d, return_field=False, complex_number=False, unwrap=False):

        d = ((d+self.distance_normalize_constant)*self.distance_normalize)*1e-3

        phase = phase*self.phase_normalize

        O_low = amplitude*torch.exp(1j*phase)

        O_holo = ASM(O_low, self.wavelength, d, self.pixel_size, zero_padding=True)
        # O_holo= center_crop(O_holo, h)

        if return_field:
            ph_prop = torch.angle(O_holo).float()
            amp_prop = torch.abs(O_holo).float()
            if unwrap:
                ph_prop = uw(ph_prop)
                return amp_prop, ph_prop
            else:
                return amp_prop, ph_prop
        
        elif complex_number:
            return O_holo
        else:
            return torch.pow(torch.abs(O_holo), 2).float()


class Back_prop(nn.Module):
    def __init__(self, args):
        super(Back_prop, self).__init__()
        self.amplitude_normalize = args.amplitude_normalize
        self.wavelength = args.wavelength
        self.pixel_size = args.pixel_size
        self.distance_normalize = args.distance_normalize
        self.distance_normalize_constant = args.distance_normalize_constant
        self.input_type = args.Holo_G_input

    def forward(self, holo, d):
        d = ((d + self.distance_normalize_constant) * self.distance_normalize) * 0.001

        holo = torch.sqrt(holo)
        O_back_prop = ASM(holo, self.wavelength, d, self.pixel_size) * self.amplitude_normalize

        if self.input_type == 'amp_pha':
            r = torch.abs(O_back_prop)
            i = torch.angle(O_back_prop)
        else:
            r = torch.real(O_back_prop)
            i = torch.imag(O_back_prop)

        return torch.cat([r, i], dim=1)
    
    
def center_crop(H, size):
    batch, channel, Nh, Nw = H.size()

    return H[:, :, (Nh - size)//2 : (Nh+size)//2, (Nw - size)//2 : (Nw+size)//2]