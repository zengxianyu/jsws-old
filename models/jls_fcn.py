import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.variable import Variable

from .densenet import *
from .resnet import *
from .vgg import *

from .funcs import *

import numpy as np
import sys
thismodule = sys.modules[__name__]
import pdb


def proc_resnet(model):
    def hook(module, input, output):
        model.feats[output.device.index] += [output]
    model.layer3[-1].bn3.register_forward_hook(hook)
    model.layer2[-1].bn3.register_forward_hook(hook)

    # model.layer3[0].conv2.stride=(1, 1)
    # model.layer3[0].downsample[0].stride=(1, 1)
    # for m in model.layer3[1:].modules():
    #     if isinstance(m, nn.Conv2d) and m.kernel_size==(3, 3):
    #         m.dilation = (2, 2)
    #         m.padding = (2, 2)

    model.layer4[0].conv2.stride=(1, 1)
    model.layer4[0].downsample[0].stride=(1, 1)

    model.layer4[1].conv2.dilation = (4, 4)
    model.layer4[1].conv2.padding = (4, 4)

    model.layer4[2].conv2.dilation = (4, 4)
    model.layer4[2].conv2.padding = (4, 4)
    model.classifier = None
    return model


def proc_densenet(model):
    def hook(module, input, output):
        model.feats[output.device.index] += [output]
    model.features.transition3[-2].register_forward_hook(hook)
    model.features.transition2[-2].register_forward_hook(hook)

    model.features.transition3[-1].kernel_size = 1
    model.features.transition3[-1].stride = 1
    def remove_sequential(all_layers, network):
        for layer in network.children():
            if isinstance(layer, nn.Sequential):  # if sequential layer, apply recursively to layers in sequential layer
                remove_sequential(all_layers, layer)
            if list(layer.children()) == []:  # if leaf node, add it to list
                all_layers.append(layer)
    all_layers = []
    remove_sequential(all_layers, model.features.denseblock4)
    for m in all_layers:
        if isinstance(m, nn.Conv2d) and m.kernel_size==(3, 3):
            m.dilation = (2, 2)
            m.padding = (2, 2)
    model.classifier = None
    return model


def proc_vgg(model):
    def hook(module, input, output):
        model.feats[output.device.index] += [output]
    model.features[3][-2].register_forward_hook(hook)
    model.features[2][-2].register_forward_hook(hook)
    model.classifier = None
    return model


procs = {'densenet169': proc_densenet,
         'vgg16': proc_vgg,
         'resnet101': proc_resnet}

class Pass(nn.Module):
    def forward(self, x):
        return x


class CLSFCN(nn.Module):
    def __init__(self, pretrained=True, c_output=20, base='densenet169'):
        super(CLSFCN, self).__init__()
        dims = dim_dict[base][::-1]
        self.fc = nn.Linear(dims[0], c_output)
        self.apply(weight_init)
        self.feature = getattr(thismodule, base)(pretrained=pretrained)
        self.apply(fraze_bn)

    def forward(self, x, boxes=None, ids=None):
        x = self.feature(x)
        cls = self.fc(x.mean(3).mean(2))
        return cls


class JLSFCN(nn.Module):
    def __init__(self, pretrained=True, c_output=21, base='densenet169'):
        super(JLSFCN, self).__init__()
        dims = dim_dict[base][::-1]
        self.preds = nn.ModuleList([nn.Conv2d(d, c_output, kernel_size=1) for d in dims])
        self.fc_cls = nn.Linear(dims[0], c_output-1)
        self.cls_sal = nn.Conv2d(dims[0], c_output, kernel_size=16)
        # if 'vgg' in base:
        #     raise NotImplementedError
        #     # self.upscales = nn.ModuleList([
        #     #     nn.ConvTranspose2d(c_output, c_output, 4, 2, 1),
        #     #     nn.ConvTranspose2d(c_output, c_output, 4, 2, 1),
        #     #     nn.ConvTranspose2d(c_output, c_output, 8, 4, 2),
        #     # ])
        if 'densenet' in base or 'resnet' in base:
            self.upscales = nn.ModuleList([
                Pass(),
                # Pass(),
                nn.ConvTranspose2d(c_output, c_output, 4, 2, 1),
                # nn.ConvTranspose2d(c_output, c_output, 4, 2, 1),
                nn.ConvTranspose2d(c_output, c_output, 16, 8, 4),
            ])
        self.apply(weight_init)
        self.feature = getattr(thismodule, base)(pretrained=pretrained)
        self.feature.feats = {}
        self.feature = procs[base](self.feature)
        self.apply(fraze_bn)

    def forward(self, x, boxes=None, ids=None):
        self.feature.feats[x.device.index] = []
        x = self.feature(x)
        feats = self.feature.feats[x.device.index]
        feats += [x]
        feats = feats[::-1]

        pred_cls_fc = self.fc_cls(x.mean(3).mean(2))

        pred_sal = self.cls_sal(x)
        pred_sal = F.sigmoid(pred_sal)

        pred = 0
        for i, feat in enumerate(feats):
            pred = self.preds[i](feat) + pred
            if i == 0:
                # pred_cls = F.avg_pool2d(pred, kernel_size=16).squeeze(3).squeeze(2)
                pred_cls = pred.mean(3).mean(2)
            pred = self.upscales[i](pred)
        pred_cls_big = pred.mean(3).mean(2)
        # pred_cls_big = F.avg_pool2d(pred, kernel_size=256).squeeze(3).squeeze(2)
        return pred, pred_cls[:, 1:], pred_cls_big[:, 1:], pred_sal, pred_cls_fc
        # return pred, pred_cls[:, 1:], pred_cls_big[:, 1:], pred_sal


if __name__ == "__main__":
    fcn = WSFCN2(base='densenet169').cuda()
    x = torch.Tensor(2, 3, 256, 256).cuda()
    sb = fcn(Variable(x))
