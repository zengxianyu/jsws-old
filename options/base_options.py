import argparse
import os


class _BaseOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def initialize(self):
        # self.parser.add_argument('--dataroot', required=True, help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        self.parser.add_argument('--batchSize', type=int, default=24, help='input batch size')
        self.parser.add_argument('--imageSize', type=int, default=256, help='input image size')
        self.parser.add_argument('--input_nc', type=int, default=3, help='input image channel')
        self.parser.add_argument('--gpu_ids', type=str, default='0,1', help='gpu ids: e.g. 0  0,1,2, 0,2')
        self.parser.add_argument('--name', type=str, help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--model', type=str,
                                 help='chooses which model to use. FCN, DeepLab, etc')
        self.parser.add_argument('--base', type=str, default='densenet169',
                                 help='chooses which backbone network to use. densenet169, vgg16, etc')
        self.parser.add_argument('--checkpoints_dir', type=str, default='savefiles', help='path to save params and tensorboard files')
        # self.parser.add_argument('--results_dir', type=str, default='../segggFiles/results', help='saves prediction results here.')


    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt, _ = self.parser.parse_known_args()
        self.opt.isTrain = self.isTrain   # train or test
        self.opt.mean = self.mean
        self.opt.std = self.std


        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        # save to the disk
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        self.opt.results_dir = '{}/results'.format(expr_dir)
        if not os.path.exists(expr_dir):
            os.makedirs(expr_dir)
        if not os.path.exists(self.opt.results_dir):
            os.makedirs(self.opt.results_dir)

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        file_name = os.path.join(expr_dir, 'opt-{}.txt'.format(self.opt.phase))
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        return self.opt
