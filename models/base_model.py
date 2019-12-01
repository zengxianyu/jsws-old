import os
import torch
import torch.nn as nn
from datetime import datetime
from .logger import Logger
import pdb


class _BaseModel:
    def __init__(self):
        pass

    def initialize(self, opt):
        self.opt = opt
        self.isTrain = opt.isTrain
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.performance = {}

        # log
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        self.writer = Logger(self.save_dir, clear=True)

    def set_input(self, input):
        self.input = input

    def write_log(self, num_iter, num_show=4):
        raise NotImplementedError

    def forward(self):
        pass

    # used in test time, no backprop
    def test(self, **kwargs):
        pass

    def get_image_paths(self):
        pass

    def optimize_parameters(self, **kwargs):
        pass

    def get_current_visuals(self):
        return self.input

    def get_current_errors(self):
        return {}

    def save(self, label):
        pass

    def load(self, label):
        pass

    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch_label):
        save_filename = '_%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)

        device = next(network.parameters()).device
        torch.save(network.cpu().state_dict(), save_path)
        network.to(device)

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label):
        save_filename = '_%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        device = next(network.parameters()).device
        network.load_state_dict(torch.load(save_path, map_location={'cuda:%d' % device.index: 'cpu'}))

    def update_learning_rate(**kwargs):
        pass
