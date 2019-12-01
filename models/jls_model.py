# coding=utf-8
import pdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import os
from PIL import Image
from .base_model import _BaseModel
import sys
from .jls_fcn import JLSFCN, CLSFCN
from .jls_deeplab import JLSDeepLab
from datasets.voc import index2color, palette

thismodule = sys.modules[__name__]


class JLSModel(_BaseModel):
    def __init__(self, opt, c_output, ignored_idx):
        _BaseModel.initialize(self, opt)
        self.name = opt.model + '_' + opt.base
        self.c_output = c_output
        self.v_mean = torch.cuda.FloatTensor(opt.mean)[None, ..., None, None]
        self.v_std = torch.cuda.FloatTensor(opt.std)[None, ..., None, None]
        _pretrained = opt.isTrain and (not opt.from_scratch)
        print('pretrained={}'.format(_pretrained))
        net = getattr(thismodule, opt.model)(pretrained=_pretrained,
                                             c_output=c_output,
                                             base=opt.base)

        net = torch.nn.parallel.DataParallel(net)
        self.net = net.cuda()
        net_cls = CLSFCN(pretrained=_pretrained, c_output=c_output-1, base=opt.base)
        net_cls = torch.nn.parallel.DataParallel(net_cls)
        self.net_cls = net_cls.cuda()
        self.loss = {}

        self.input = None  # input image
        self.targets = None  # image-level categrory
        self.seg_targets = None  # saliency groundtruth

        self.pred_cls2 = None  # catetory predicted by the auxiliary network
        self.pred_seg = None  # segmentation results
        self.pred_cls_big = None  # category from big segmentation results
        self.pred_cls = None  # category from small segmentation results
        self.pred_sal = None  # saliency map

        if opt.phase is 'test1':
            self.forward = self.forward_fcn
        elif opt.phase is 'test2':
            self.forward = self.forward_deeplab
        elif opt.phase is 'train1':
            self.forward = self.forward_fcn
            self.backward_cls = self.backward_cls_fcn
            self.criterion_cls = nn.BCEWithLogitsLoss()
            # self.criterion_cls = nn.MultiLabelSoftMarginLoss()
            self.criterion_seg = None
            self.criterion_sal = nn.NLLLoss2d()
            self.optimizer = torch.optim.Adam([
                {'params': filter(lambda p: p.requires_grad, self.net.parameters()), 'lr': opt.lr, 'betas':(0.95, 0.999)},
                {'params': filter(lambda p: p.requires_grad, self.net_cls.parameters()), 'lr': opt.lr, 'betas':(0.95, 0.999)}
            ])
        elif opt.phase is 'train2':
            self.forward = self.forward_deeplab
            self.backward_cls = self.backward_cls_deeplab
            self.criterion_seg = nn.CrossEntropyLoss(ignore_index=ignored_idx)
            self.criterion_cls = nn.MultiLabelSoftMarginLoss()
            self.criterion_sal = nn.NLLLoss2d()
            self.optimizer = torch.optim.Adam([
                {'params': filter(lambda p: p.requires_grad, self.net.parameters()), 'lr': opt.lr, 'betas':(0.95, 0.999)},
                {'params': filter(lambda p: p.requires_grad, self.net_cls.parameters()), 'lr': opt.lr, 'betas':(0.95, 0.999)}
            ])
            # self.optimizer = torch.optim.Adam(self.net.parameters(),
            #                                   lr=opt.lr)
        else:
            raise NotImplementedError

    def save(self, label):
        self.save_network(self.net, self.name, label)

    def load(self, label):
        print('loading %s' % label)
        self.load_network(self.net, self.name, label)

    def write_log_eval(self, num_iter):
        for k, v in self.performance.items():
            self.writer.add_scalar(k, v, num_iter)
        self.writer.write_html()

    def write_log(self, num_iter, num_show=4):
        loss = 0
        for k, v in self.loss.items():
            self.writer.add_scalar(k, v, num_iter)
            loss += v
        self.writer.add_scalar('total loss', loss, num_iter)
        num_show = num_show if self.input.size(0) > num_show else self.input.size(0)

        _, pred_index = self.pred_seg[:num_show].detach().cpu().max(1)
        msk = torch.Tensor(num_show, 256, 256, 3)
        for j, color in enumerate(index2color):
            if (pred_index == j).sum() > 0:
                msk[pred_index == j, :] = torch.Tensor(color)
        msk = torch.transpose(msk, 2, 3)
        msk = torch.transpose(msk, 1, 2)
        self.writer.add_image('segmentation', torchvision.utils.make_grid(msk / 255).detach(), num_iter)

        pred = self.pred_sal[:num_show]
        pred = pred[:, 1:, ...]
        self.writer.add_image('saliency', torchvision.utils.make_grid(pred.expand(-1, 3, -1, -1)).detach(), num_iter)

        img = self.input[:num_show] * self.v_std + self.v_mean
        self.writer.add_image('image', torchvision.utils.make_grid(img).detach(), num_iter)
        self.writer.write_html()

    def set_input(self, input, targets=None, seg_targets=None):
        self.input = input.cuda()
        self.targets = targets
        self.seg_targets = seg_targets
        if targets is not None:
            self.targets = self.targets.cuda()
        if seg_targets is not None:
            self.seg_targets = self.seg_targets.cuda()

    def forward_fcn(self):
        # print("We are Forwarding !!")
        pred_seg, pred_cls, pred_cls_big, pred_sal, self.pred_cls_fc = self.net.forward(self.input)
        self.pred_seg = pred_seg
        self.pred_cls_big = pred_cls_big
        self.pred_cls = pred_cls
        pred_sal = F.softmax(pred_seg, 1) * pred_sal
        self.pred_sal = torch.cat((pred_sal[:, :1], pred_sal[:, 1:].sum(1, keepdim=True)), 1)

    def forward_deeplab(self):
        # print("We are Forwarding !!")
        pred_seg, pred_sal, self.pred_cls_fc = self.net.forward(self.input)
        self.pred_seg = pred_seg
        pred_sal = F.softmax(pred_seg, 1) * pred_sal
        self.pred_sal = torch.cat((pred_sal[:, :1], pred_sal[:, 1:].sum(1, keepdim=True)), 1)

    def test(self, input, name, WW, HH, gt=None, save_prob=False):
        if save_prob:
            if not os.path.exists(self.opt.results_dir + '_prob'):
                os.mkdir(self.opt.results_dir + '_prob')
        self.set_input(input)
        with torch.no_grad():
            self.forward()
            outputs = self.pred_seg
            if gt is not None:
                bsize = input.size(0)
                gt = torch.cat((torch.ones(bsize, 1, 1, 1).cuda(), gt[..., None, None].cuda()), 1)
                outputs *= gt.cuda()
            # else:
            #     pred_cls_big = self.pred_seg.mean(3, keepdim=True).mean(2, keepdim=True)[:, 1:]
            #     # pred_cls_big = F.avg_pool2d(self.pred_seg, kernel_size=256).squeeze(3).squeeze(2)[:, 1:]
            #     pred_cls_big = F.sigmoid(pred_cls_big)
            #     outputs[:, 1:] *= pred_cls_big
            _, outputs_seg = outputs.max(1)
        outputs = F.softmax(outputs, 1).detach().cpu().numpy()
        outputs_seg = outputs_seg.detach().cpu().numpy()
        for ii, msk in enumerate(outputs_seg):
            msk = Image.fromarray(msk.astype(np.uint8))
            msk = msk.convert('P')
            msk.putpalette(palette)
            msk = msk.resize((WW[ii], HH[ii]))
            msk.save('{}/{}.png'.format(self.opt.results_dir, name[ii]), 'PNG')
            if save_prob:
                np.save('{}/{}'.format(self.opt.results_dir + '_prob', name[ii]), outputs[ii])

    def backward_cls_fcn(self):
        # Combined loss
        loss_var = self.criterion_cls(self.pred_cls_big, self.targets)
        loss_var += self.criterion_cls(self.pred_cls, self.targets)
        loss_var += self.criterion_cls(self.pred_cls_fc, self.targets)

        pred_cls2 = self.net_cls(self.input)
        """
        Zhang, Ying, et al. "Deep mutual learning." Proceedings of the IEEE Conference on Computer Vision and Pattern
        Recognition. 2018.
        """
        loss_aux = self.criterion_cls(pred_cls2, self.targets)
        loss_aux += F.kl_div(F.logsigmoid(self.pred_cls_big), F.sigmoid(pred_cls2).detach())
        loss_aux += F.kl_div(F.logsigmoid(pred_cls2), F.sigmoid(self.pred_cls_big).detach())
        loss = loss_var + loss_aux
        loss.backward()
        self.loss['cls'] = loss_var.data.item()
        # self.loss['aux'] = loss_aux.data.item()

    def backward_cls_deeplab(self):
        # Combined loss
        loss_var = 0.9 * self.criterion_seg(self.pred_seg, self.seg_targets)
        bsize = self.targets.size(0)
        _clsm = torch.cat((torch.ones(bsize, 1, 1, 1).cuda(), self.targets[..., None, None]), 1)
        _, self_gt = (self.pred_seg * _clsm).max(1)
        loss_var += 0.1 * self.criterion_seg(self.pred_seg, self_gt.detach())

        pred_cls_big = F.avg_pool2d(self.pred_seg, kernel_size=256).squeeze(3).squeeze(2)[:, 1:]
        loss_var += self.criterion_cls(pred_cls_big,
                                            self.targets)
        loss_var += self.criterion_cls(self.pred_cls_fc, self.targets)

        pred_cls2 = self.net_cls(self.input)
        """
        Zhang, Ying, et al. "Deep mutual learning." Proceedings of the IEEE Conference on Computer Vision and Pattern
        Recognition. 2018.
        """
        loss_aux = self.criterion_cls(pred_cls2, self.targets)
        loss_aux += F.kl_div(F.logsigmoid(pred_cls_big), F.sigmoid(pred_cls2).detach())
        loss_aux += F.kl_div(F.logsigmoid(pred_cls2), F.sigmoid(pred_cls_big).detach())
        loss = loss_var + loss_aux
        loss.backward()
        self.loss['cls'] = loss_var.data.item()

    def backward_sal(self):
        # Combined loss
        # assert self.pred_sal.min()>=0
        # assert self.pred_sal.max()<=1
        # assert self.targets.min()>=0
        # assert self.targets.max()<=1
        # self.pred_sal[self.pred_sal>1] = 1
        # self.pred_sal[self.pred_sal<0] = 0
        loss_var = self.criterion_sal(self.pred_sal, self.targets)
        loss_var.backward()
        self.loss['sal'] = loss_var.item()

    def optimize_parameters_cls(self):
        self.forward()
        self.optimizer.zero_grad()
        self.backward_cls()
        self.optimizer.step()

    def optimize_parameters_sal(self):
        self.forward()
        self.optimizer.zero_grad()
        self.backward_sal()
        self.optimizer.step()

    def switch_to_train(self):
        self.net.train()

    def switch_to_eval(self):
        self.net.eval()
