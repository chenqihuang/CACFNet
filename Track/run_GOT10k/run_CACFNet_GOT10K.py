# -*- coding: utf-8 -*-

import argparse
from got10k.experiments import *
from Track.track_config import *
from Track.util import *
from Track.models.CACFNet import *
from got10k.trackers import Tracker
import torch


# config = CACFNetTrackConfig()


def args_():
    parser = argparse.ArgumentParser(description='Test DCFNet on OTB')
    parser.add_argument('--dataset', metavar='SET', default='OTB2015',
                        choices=['OTB2013', 'OTB2015'], help='tune on which dataset')
    parser.add_argument('--model', metavar='PATH', default='param.pth')
    args = parser.parse_args()
    return args


class CACFNetTraker(Tracker):
    # 跟踪器定义
    # 第一帧初始化跟踪器
        # 输入图片和初始目标框
        # 初始化目标
    # 第二帧开始跟踪
        # 输入图片
        # 输出预测
    def __init__(self, config=CACFNetTrackConfig(), gpu=True):

        super(CACFNetTraker, self).__init__(name='CACFNet_no_cv2')

        self.gpu = gpu
        self.config = config
        self.net = CACFNet(self.config)
        self.net.load_param(self.config.feature_path)
                                      weight_decay=5e-5)
        self.net.eval()
        if gpu:
            self.net.cuda()

    def init(self, image, box):
        image = np.asarray(image)
        # image = cv2.cvtColor((image), cv2.COLOR_RGB2BGR)

        self.target_pos, self.target_sz = rect1_2_cxy_wh(box)
        self.min_sz = np.maximum(self.config.min_scale_factor * self.target_sz, 4)
        self.max_sz = np.minimum(image.shape[:2], self.config.max_scale_factor * self.target_sz)
        inputs = get_patchs(image, self.target_pos, self.target_sz, self.config)
        self.net.CACF_update(inputs)

    def update(self, image):
        image = np.asarray(image)
        # image = cv2.cvtColor((image), cv2.COLOR_RGB2BGR)

        patch_crop = np.zeros((self.config.num_scale, 3, 125, 125),
                              np.float32)
        for i in range(self.config.num_scale):  # crop multi-scale search region
            window_sz = self.target_sz * (self.config.scale_factor[i] * (1 + self.config.padding))
            bbox = cxy_wh_2_bbox(self.target_pos, window_sz)
            patch_crop[i, :] = crop_chw(image, bbox, self.config.crop_sz)

        search = patch_crop - self.config.net_average_image

        if self.gpu:
            response = self.net(torch.Tensor(search).cuda())
        else:
            response = self.net(torch.Tensor(search))
        peak, idx = torch.max(response.view(self.config.num_scale, -1), 1)
        peak = peak.data.cpu().numpy() * self.config.scale_penalties
        best_scale = np.argmax(peak)
        r_max, c_max = np.unravel_index(idx[best_scale].cpu().numpy(), self.config.net_input_size)

        if r_max > self.config.net_input_size[0] / 2:
            r_max = r_max - self.config.net_input_size[0]
        if c_max > self.config.net_input_size[1] / 2:
            c_max = c_max - self.config.net_input_size[1]
        window_sz = self.target_sz * (self.config.scale_factor[best_scale] * (1 + self.config.padding))

        self.target_pos = self.target_pos + np.array([c_max, r_max]) * window_sz / self.config.net_input_size
        self.target_sz = np.minimum(np.maximum(window_sz / (1 + self.config.padding), self.min_sz), self.max_sz)

        inputs = get_patchs(image, self.target_pos, self.target_sz, self.config)
        self.net.CACF_update(inputs, lr=self.config.interp_factor)

        return cxy_wh_2_rect1(self.target_pos, self.target_sz)  # 1-index


if __name__ == '__main__':
    # setup tracker

    tracker = CACFNetTraker()

    # setup experiments
    experiments = [
        # ExperimentGOT10k('data/GOT-10k', subset='test'),
        # ExperimentOTB('data/OT2013', version=2013),
        ExperimentOTB('data/OTB100', version=2015),
        # ExperimentVOT('data/VOT2015', version=2015),
        # ExperimentDTB70('data/DTB70'),
        # ExperimentTColor128('data/Temple-color-128'),
        # ExperimentUAV123('data/UAV123', version='UAV123'),
        #
        # ExperimentUAV123('data/UAV123', version='UAV20L'),
        # ExperimentNfS('data/nfs', fps=30),
        # ExperimentNfS('data/nfs', fps=240)
    ]
    # run tracking experiments and report performance
    for e in experiments:
        e.run(tracker, visualize=False)
        print tracker.name
        e.report([tracker.name])
 
