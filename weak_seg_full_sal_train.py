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


from options.train_options import TrainOptions
opt = TrainOptions()  # set CUDA_VISIBLE_DEVICES before import torch
opt.parser.set_defaults(model='JLSFCN')
opt.parser.set_defaults(name='jlsfcn_dense169')
opt.parser.set_defaults(phase='train1')
opt = opt.parse()

#home = os.path.expanduser("~")
home = "."

voc_train_img_dir = '%s/data/datasets/segmentation_Dataset/VOCdevkit/VOC2012/JPEGImages'%home
voc_train_gt_dir = '%s/data/datasets/segmentation_Dataset/VOCdevkit/VOC2012/SegmentationClassAug'%home

voc_val_img_dir = '%s/data/datasets/segmentation_Dataset/VOCdevkit/VOC2012/JPEGImages'%home
voc_val_gt_dir = '%s/data/datasets/segmentation_Dataset/VOCdevkit/VOC2012/SegmentationClass'%home

voc_train_split = '%s/data/datasets/segmentation_Dataset/VOCdevkit/VOC2012/ImageSets/Segmentation/argtrain.txt'%home
voc_val_split = '%s/data/datasets/segmentation_Dataset/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt'%home

sal_train_img_dir = '%s/data/datasets/saliency_Dataset/DUT-train/images'%home
sal_train_gt_dir = '%s/data/datasets/saliency_Dataset/DUT-train/masks'%home

sal_val_img_dir = '%s/data/datasets/saliency_Dataset/ECSSD/images'%home
sal_val_gt_dir = '%s/data/datasets/saliency_Dataset/ECSSD/masks'%home


sal_train_loader = torch.utils.data.DataLoader(
    Folder(sal_train_img_dir, sal_train_gt_dir,
           crop=0.9, flip=True, rotate=None, size=opt.imageSize,
           mean=opt.mean, std=opt.std, training=True),
    batch_size=opt.batchSize, shuffle=True, num_workers=4, pin_memory=True)

sal_val_loader = torch.utils.data.DataLoader(
    Folder(sal_val_img_dir, sal_val_gt_dir,
           crop=None, flip=False, rotate=None, size=opt.imageSize,
           mean=opt.mean, std=opt.std, training=False),
    batch_size=opt.batchSize, shuffle=True, num_workers=4, pin_memory=True)

c_output = 21


voc_train_loader = torch.utils.data.DataLoader(
    VOC(voc_train_img_dir, voc_train_gt_dir, voc_train_split,
           crop=0.9, flip=True, rotate=None, size=opt.imageSize,
           mean=opt.mean, std=opt.std, training=True),
    batch_size=opt.batchSize, shuffle=True, num_workers=4, pin_memory=True)

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


model = JLSModel(opt, c_output, voc_train_loader.dataset.ignored_idx)


def train(model):
    print("============================= TRAIN ============================")
    # model.load('best')
    model.switch_to_train()

    voc_train_iter = iter(voc_train_loader)
    voc_it = 0
    sal_train_iter = iter(sal_train_loader)
    sal_it = 0
    log = {'best': 0, 'best_it': 0}

    for i in tqdm(range(opt.train_iters), desc='train'):
        # if i!=0 and i % 1000 == 0:
        #     for g in model.optimizer.param_groups: g['lr'] *= 0.5
        # train on saliency
        if sal_it >= len(sal_train_loader):
            sal_train_iter = iter(sal_train_loader)
            sal_it = 0
        img, gt = sal_train_iter.next()
        sal_it += 1

        model.set_input(img, gt.long())
        model.forward()
        model.optimizer.zero_grad()
        model.backward_sal()

        if voc_it >= len(voc_train_loader):
            voc_train_iter = iter(voc_train_loader)
            voc_it = 0
        img, gt = voc_train_iter.next()
        voc_it += 1

        bsize = img.size(0)
        gt[gt == -1] = c_output
        gt = torch.zeros(bsize, c_output + 1, opt.imageSize, opt.imageSize).scatter(1, gt.unsqueeze(1), 1)
        gt = gt.sum(3).sum(2)
        gt = gt[:, :c_output]
        gt = gt[:, 1:]
        gt[gt>0] = 1

        model.set_input(img, gt)
        model.forward()
        model.backward_cls()
        model.optimizer.step()

        if i % opt.display_freq == 0:
            model.write_log(i)

        if i != 0 and i % opt.save_latest_freq == 0:
            model.save(i)
            miou = test(model)
            model.write_log_eval(i)
            log[i] = {'miou': miou}
            if miou > log['best']:
                log['best'] = miou
                log['best_it'] = i
                model.save('best')
            print(u'max miou: %.4f, current miou: %.4f'%(log['best'], miou))
            with open(model.save_dir+'/'+'train-log.json', 'w') as outfile:
                json.dump(log, outfile)


train(model)

print("We are done")
