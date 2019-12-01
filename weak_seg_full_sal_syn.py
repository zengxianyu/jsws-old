# coding=utf-8

import pdb
import time
import torch
import sys
from tqdm import tqdm
from models import JLSModel
from datasets import VOC, Folder
from evaluate_seg import evaluate_iou
import json
import os


from options.test_options import TestOptions
opt = TestOptions()  # set CUDA_VISIBLE_DEVICES before import torch
opt.parser.set_defaults(model='JLSFCN')
opt.parser.set_defaults(name='jlsfcn_dense169')
opt.parser.set_defaults(phase='test1')
opt = opt.parse()

#home = os.path.expanduser("~")
home = "."

voc_train_img_dir = '%s/data/datasets/segmentation_Dataset/VOCdevkit/VOC2012/JPEGImages'%home
voc_train_gt_dir = '%s/data/datasets/segmentation_Dataset/VOCdevkit/VOC2012/SegmentationClassAug'%home

voc_val_img_dir = '%s/data/datasets/segmentation_Dataset/VOCdevkit/VOC2012/JPEGImages'%home
voc_val_gt_dir = '%s/data/datasets/segmentation_Dataset/VOCdevkit/VOC2012/SegmentationClass'%home

voc_train_split = '%s/data/datasets/segmentation_Dataset/VOCdevkit/VOC2012/ImageSets/Segmentation/argtrain.txt'%home
voc_val_split = '%s/data/datasets/segmentation_Dataset/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt'%home

label = "" # label of model parameters to load

c_output = 21


voc_train_loader = torch.utils.data.DataLoader(
    VOC(voc_train_img_dir, voc_train_gt_dir, voc_train_split,
           crop=None, flip=False, rotate=None, size=opt.imageSize,
           mean=opt.mean, std=opt.std, training=-1),
    batch_size=opt.batchSize, shuffle=False, num_workers=4, pin_memory=True)

voc_val_loader = torch.utils.data.DataLoader(
    VOC(voc_val_img_dir, voc_val_gt_dir, voc_val_split,
           crop=None, flip=False, rotate=None, size=opt.imageSize,
           mean=opt.mean, std=opt.std, training=False),
    batch_size=opt.batchSize, shuffle=True, num_workers=4, pin_memory=True)


def test(model):
    print("============================= TEST ============================")
    model.switch_to_eval()
    for i, (img, name, WW, HH) in tqdm(enumerate(voc_val_loader), desc='testing'):
        model.test(img, name, WW, HH)
    model.switch_to_train()
    miou = evaluate_iou(opt.results_dir, voc_val_gt_dir, c_output)
    model.performance = {'miou': miou}
    return miou


def syn(model):
    print("============================= TEST ============================")
    model.switch_to_eval()
    _rdir = model.opt.results_dir
    model.opt.results_dir = '/'.join(model.opt.results_dir.split('/')[:-1])+'/syn_train'
    if not os.path.exists(model.opt.results_dir):
        os.mkdir(model.opt.results_dir)
    for i, (img, gt, name, WW, HH) in tqdm(enumerate(voc_train_loader), desc='testing'):
        bsize = img.size(0)
        gt[gt == -1] = c_output
        gt = torch.zeros(bsize, c_output + 1, opt.imageSize, opt.imageSize).scatter(1, gt.unsqueeze(1), 1)
        gt = gt.sum(3).sum(2)
        gt = gt[:, :c_output]
        gt = gt[:, 1:]
        gt[gt>0] = 1
        model.test(img, name, WW, HH, gt, save_prob=True)
    model.switch_to_train()
    miou = evaluate_iou(model.opt.results_dir, voc_train_gt_dir, c_output)
    model.opt.results_dir = _rdir
    model.performance = {'miou': miou}
    print(miou)
    with open(model.save_dir+'/'+'syn-log.json', 'w') as outfile:
        json.dump(model.performance, outfile)
    return miou


model = JLSModel(opt, c_output, voc_train_loader.dataset.ignored_idx)
model.load(label)


syn(model)

print("We are done")
