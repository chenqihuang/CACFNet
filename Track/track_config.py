# -*- coding: utf-8 -*-
import numpy as np
import torch
from util import *


class CACFNetTrackConfig(object):

    feature_path = ''
    crop_sz = 125

    lambda0 = 1e-4
    lambda1 = 0.1
    lambda2 = 0.5

    padding = 2
    output_sigma_factor = 0.1
    interp_factor = 0.01
    num_scale = 3
    scale_step = 1.0275
    scale_factor = scale_step ** (np.arange(num_scale) - num_scale / 2)
    min_scale_factor = 0.2
    max_scale_factor = 5
    scale_penalty = 0.9925
    scale_penalties = scale_penalty ** (
        np.abs((np.arange(num_scale) - num_scale / 2))) 

    net_input_size = [crop_sz, crop_sz]
    net_average_image = np.array([104, 117, 123]).reshape(-1, 1, 1).astype(np.float32)
    output_sigma = crop_sz / (1 + padding) * output_sigma_factor
    y = gaussian_shaped_labels(output_sigma, net_input_size)
    yf = torch.rfft(torch.Tensor(y).view(1, 1, crop_sz, crop_sz).cuda(), signal_ndim=2)
    cos_window = torch.Tensor(np.outer(np.hanning(crop_sz), np.hanning(crop_sz))).cuda()
  
    display = False
    savefig = False
    visualization = False
    use_gpu = True

