# coding=utf-8

import pdb
import time
import torch
import sys
from tqdm import tqdm
from models import JLSModel
from datasets import VOC, Folder, ImageFiles
from evaluate_seg import evaluate_iou
from evaluate_sal import fm_and_mae
import json
import os

from options.test_options import TestOptions
opt = TestOptions()  # set CUDA_VISIBLE_DEVICES before import torch
opt.parser.set_defaults(model='JLSDeepLab')
opt.parser.set_defaults(phase='test2')
opt = opt.parse()

#home = os.path.expanduser("~")
home = "."

voc_train_img_dir = '%s/data/datasets/segmentation_Dataset/VOCdevkit/VOC2012/JPEGImages'%home
voc_train_gt_dir = '%s/data/datasets/segmentation_Dataset/VOCdevkit/VOC2012/SegmentationClassAug'%home

voc_val_img_dir = '%s/data/datasets/segmentation_Dataset/VOCdevkit/VOC2012/JPEGImages'%home
voc_val_gt_dir = '%s/data/datasets/segmentation_Dataset/VOCdevkit/VOC2012/SegmentationClass'%home

voc_train_split = '%s/data/datasets/segmentation_Dataset/VOCdevkit/VOC2012/ImageSets/Segmentation/argtrain.txt'%home
voc_val_split = '%s/data/datasets/segmentation_Dataset/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt'%home
voc_test_split = '%s/data/datasets/segmentation_Dataset/VOCdevkit/VOC2012/ImageSets/Segmentation/test.txt'%home

sal_val_img_dir = '%s/data/datasets/saliency_Dataset/DUT-test/images'%home
sal_val_gt_dir = '%s/data/datasets/saliency_Dataset/DUT-test/masks'%home

c_output = 21

voc_test_loader = torch.utils.data.DataLoader(
    VOC(voc_val_img_dir, voc_val_gt_dir, voc_val_split,
        crop=None, flip=False, rotate=None, size=opt.imageSize,
        mean=opt.mean, std=opt.std, training=False),
    batch_size=opt.batchSize, shuffle=True, num_workers=4, pin_memory=True)

voc_train_loader = torch.utils.data.DataLoader(
    VOC(voc_train_img_dir, voc_train_gt_dir, voc_train_split,
        crop=None, flip=False, rotate=None, size=opt.imageSize,
        mean=opt.mean, std=opt.std, training=-1),
    batch_size=opt.batchSize, shuffle=False, num_workers=4, pin_memory=True)

sal_val_loader = torch.utils.data.DataLoader(
    Folder(sal_val_img_dir, sal_val_gt_dir,
           crop=None, flip=False, rotate=None, size=opt.imageSize,
           mean=opt.mean, std=opt.std, training=False),
    batch_size=opt.batchSize, shuffle=True, num_workers=4, pin_memory=True)


def test(model):
    print("============================= TEST ============================")
    model.switch_to_eval()
    for i, (img, name, WW, HH) in tqdm(enumerate(voc_test_loader), desc='testing'):
        model.test(img, name, WW, HH)
    model.switch_to_train()
    miou = evaluate_iou(opt.results_dir, voc_val_gt_dir, c_output)
    model.performance = {'miou': miou}
    print(miou)
    with open('val_voc.json', 'w') as f:
        json.dump(model.performance, f)
    return model.performance


def test_sal(model):
    print("============================= TEST ============================")
    model.switch_to_eval()
    model.opt.sal_results_dir = model.opt.results_dir + '_sal'
    if not os.path.exists(model.opt.sal_results_dir):
        os.mkdir(model.opt.sal_results_dir)
    for i, (img, name, WW, HH) in tqdm(enumerate(sal_val_loader), desc='testing'):
        model.test_sal(img, name, WW, HH)
    model.switch_to_train()
    maxfm, mae, _, _ = fm_and_mae(opt.sal_results_dir, sal_val_gt_dir)
    model.performance = {'maxfm': maxfm, 'mae': mae}
    print(maxfm)
    print(mae)
    with open('val_sal.json', 'w') as f:
        json.dump(model.performance, f)
    return model.performance


model = JLSModel(opt, c_output, voc_train_loader.dataset.ignored_idx)
model.load('best')


miou = test(model)

print("We are done")
