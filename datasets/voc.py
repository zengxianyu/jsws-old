import numpy as np
import PIL.Image as Image
import torch
from torchvision import transforms
import pdb
from .base_data import _BaseData

palette = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128,
           128, 128, 128, 64, 0, 0, 192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128,
           64, 128, 128, 192, 128, 128, 0, 64, 0, 128, 64, 0, 0, 192, 0, 128, 192, 0, 0, 64, 128]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])


def labelcolormap(N):
    cmap = np.zeros((N, 3), dtype=np.uint8)
    for i in range(N):
        r = 0
        g = 0
        b = 0
        id = i
        for j in range(7):
            str_id = uint82bin(id)
            r = r ^ (np.uint8(str_id[-1]) << (7 - j))
            g = g ^ (np.uint8(str_id[-2]) << (7 - j))
            b = b ^ (np.uint8(str_id[-3]) << (7 - j))
            id = id >> 3
        cmap[i, 0] = r
        cmap[i, 1] = g
        cmap[i, 2] = b
    return cmap


index2color = labelcolormap(21)
index2color = [list(hh) for hh in index2color]
index2name = ('background', 'airplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
              'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant',
              'sheep', 'sofa', 'train', 'tvmonitor')


class _BaseVOC(_BaseData):
    # def __init__(self, root, split='train', crop=None, flip=False, rotate=None,
    #              mean=None, std=None):
    #     super(BaseVOC, self).__init__()
    def __init__(self, img_dir, split_file, img_format='jpg', size=256, training=True,
                 crop=None, rotate=None, flip=False, mean=None, std=None):
        super(_BaseVOC, self).__init__(crop=crop, rotate=rotate, flip=flip, mean=mean, std=std)
        self.ignored_idx = -1
        self.training = training
        self.size = size
        self.mean, self.std = mean, std
        with open(split_file, 'r') as f:
            names = f.read().split('\n')[:-1]
        img_filenames = ['{}/{}.{}'.format(img_dir, name, img_format) for name in names]
        self.img_filenames = img_filenames
        self.names = names

    def __len__(self):
        return len(self.names)


class VOC(_BaseVOC):
    def __init__(self, img_dir, gt_dir, split_file, img_format='jpg', gt_format='png', size=256, training=True,
                 crop=None, rotate=None, flip=False, mean=None, std=None):
        super(VOC, self).__init__(img_dir, split_file, img_format, size, training, crop, rotate,
                                  flip, mean, std)
        gt_filenames = ['{}/{}.{}'.format(gt_dir, _name, gt_format) for _name in self.names]
        self.gt_filenames = gt_filenames

    def __getitem__(self, index):
        # load image
        img_file = self.img_filenames[index]
        img = Image.open(img_file)
        gt_file = self.gt_filenames[index]
        name = self.names[index]
        w, h = img.size
        if self.training:
            gt = Image.open(gt_file).convert("P")
            gt = gt.resize(img.size)
            data = (img, gt)
        else:
            data = (img,)
        if self.crop is not None:
            data = self.random_crop(*data)
        if self.rotate is not None:
            data = self.random_rotate(*data)
        if self.flip:
            data = self.random_flip(*data)
        img = data[0]
        img = img.resize((self.size, self.size))

        img = np.array(img, dtype=np.float64) / 255.0
        if len(img.shape) < 3:
            img = np.stack((img, img, img), 2)
        if img.shape[2] > 3:
            img = img[:, :, :3]
        if self.mean is not None:
            img -= self.mean
        if self.std is not None:
            img /= self.std
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        if self.training:
            gt = data[1]
            gt = gt.resize((self.size, self.size))
            gt = np.array(gt, dtype=np.int64)
            gt[gt == 255] = self.ignored_idx
            gt = torch.from_numpy(gt)
            if self.training == -1:
                return img, gt, name, w, h
            else:
                return img, gt
        else:
            return img, name, w, h


class VOCMS(_BaseVOC):
    def __init__(self, img_dir, gt_dirs, split_file, img_format='jpg', gt_format='png', size=256, training=True,
                 crop=None, rotate=None, flip=False, mean=None, std=None):
        super(VOCMS, self).__init__(img_dir, split_file, img_format, size, training, crop, rotate,
                                    flip, mean, std)
        list_gt_filenames = [['{}/{}.{}'.format(gt_dir, _name, gt_format) for _name in self.names] for gt_dir in gt_dirs]
        self.list_gt_filenames = list_gt_filenames

    def __getitem__(self, index):
        # load image
        img_file = self.img_filenames[index]
        img = Image.open(img_file)
        list_gt_file = [gt_filenames[index] for gt_filenames in self.list_gt_filenames]
        name = self.names[index]
        w, h = img.size
        if self.training:
            gts = [Image.open(gt_file).convert("P") for gt_file in list_gt_file]
            gts = [gt.resize(img.size) for gt in gts]
            data = (img,) + tuple(gts)
        else:
            data = (img,)
        if self.crop is not None:
            data = self.random_crop(*data)
        if self.rotate is not None:
            data = self.random_rotate(*data)
        if self.flip:
            data = self.random_flip(*data)
        img = data[0]
        img = img.resize((self.size, self.size))
        img = np.array(img, dtype=np.float64) / 255.0
        if len(img.shape) < 3:
            img = np.stack((img, img, img), 2)
        if img.shape[2] > 3:
            img = img[:, :, :3]
        if self.mean is not None:
            img -= self.mean
        if self.std is not None:
            img /= self.std
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        if self.training:
            gts = data[1:]
            gts = [gt.resize((self.size, self.size)) for gt in gts]
            gts = [np.array(gt, dtype=np.int64) for gt in gts]
            for gt in gts:
                gt[gt == 255] = self.ignored_idx
            gts = [torch.from_numpy(gt) for gt in gts]
            if self.training == -1:
                return img, gts, name, w, h
            else:
                return img, gts
        else:
            return img, name, w, h


if __name__ == "__main__":
    sb = WSVOCSemi('/home/zeng/data/datasets/segmentation_Dataset/VOCdevkit/VOC2012',
                   '/home/zeng/WSL/size256', split='train', img_root='/home/zeng/data/datasets/ILSVRC14VOC/images',
                   msk_root='/home/zeng/WSLfiles/STsoftmax_densenet169/semi_results',
                   source_transform=transforms.Compose([transforms.Resize((256, 256))]),
                   target_transform=transforms.Compose([transforms.Resize((256, 256))]))
    img, gt, syn, name = sb.__getitem__(0)
    pdb.set_trace()
