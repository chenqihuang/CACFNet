# -*- coding: utf-8 -*-
import numpy as np
import cv2
import torch
import os

def cxy_wh_2_rect1(pos, sz):
    return np.array([pos[0]-sz[0]/2+1, pos[1]-sz[1]/2+1, sz[0], sz[1]])  # 1-index


def rect1_2_cxy_wh(rect):
    return np.array([rect[0]+rect[2]/2-1, rect[1]+rect[3]/2-1]), np.array([rect[2], rect[3]])  # 0-index


def cxy_wh_2_bbox(cxy, wh):
    return np.array([cxy[0]-wh[0]/2, cxy[1]-wh[1]/2, cxy[0]+wh[0]/2, cxy[1]+wh[1]/2])  # 0-index


def gaussian_shaped_labels(sigma, sz):
    x, y = np.meshgrid(np.arange(1, sz[0]+1) - np.floor(float(sz[0]) / 2), np.arange(1, sz[1]+1) - np.floor(float(sz[1]) / 2))
    d = x ** 2 + y ** 2
    g = np.exp(-0.5 / (sigma ** 2) * d)

    g = np.roll(g, int(-np.floor(float(sz[0]) / 2.) + 1), axis=0)
    g = np.roll(g, int(-np.floor(float(sz[1]) / 2.) + 1), axis=1)
    return g


def crop_chw(image, bbox, out_sz, padding=(0, 0, 0)):
    a = (out_sz-1) / (bbox[2]-bbox[0])
    b = (out_sz-1) / (bbox[3]-bbox[1])
    c = -a * bbox[0]
    d = -b * bbox[1]
    mapping = np.array([[a, 0, c],
                        [0, b, d]]).astype(np.float)
    crop = cv2.warpAffine(image, mapping, (out_sz, out_sz), borderMode=cv2.BORDER_CONSTANT, borderValue=padding)
    return np.transpose(crop, (2, 0, 1)) 


def get_patchs(im, target_pos, target_sz, config):
    window_sz = target_sz * (1 + config.padding)
    bbox = cxy_wh_2_bbox(target_pos, window_sz)
    patch = crop_chw(im, bbox, config.crop_sz)
    bbox1 = np.array([bbox[0], bbox[1] - window_sz[1], bbox[2], bbox[3] - window_sz[1]])  # up
    patch1 = crop_chw(im, bbox1, config.crop_sz)
    bbox2 = np.array([bbox[0], bbox[1] + window_sz[1], bbox[2], bbox[3] + window_sz[1]])  # down
    patch2 = crop_chw(im, bbox2, config.crop_sz)
    bbox3 = np.array([bbox[0] - window_sz[0], bbox[1], bbox[2] - window_sz[0], bbox[3]])  # left
    patch3 = crop_chw(im, bbox3, config.crop_sz)
    bbox4 = np.array([bbox[0] + window_sz[0], bbox[1], bbox[2] + window_sz[0], bbox[3]])  # right
    patch4 = crop_chw(im, bbox4, config.crop_sz)
    input = []
    target = patch - config.net_average_image
    target = torch.Tensor(np.expand_dims(target, axis=0)).cuda()
    input.append(target)
    target1 = patch1 - config.net_average_image
    target1 = torch.Tensor(np.expand_dims(target1, axis=0)).cuda()
    input.append(target1)
    target2 = patch2 - config.net_average_image
    target2 = torch.Tensor(np.expand_dims(target2, axis=0)).cuda()
    input.append(target2)
    target3 = patch3 - config.net_average_image
    target3 = torch.Tensor(np.expand_dims(target3, axis=0)).cuda()
    input.append(target3)
    target4 = patch4 - config.net_average_image
    target4 = torch.Tensor(np.expand_dims(target4, axis=0)).cuda()
    input.append(target4)

    return input



if __name__ == '__main__':
    pass
