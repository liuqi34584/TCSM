from __future__ import print_function
import argparse
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import cv2
from PIL import Image
import shutil
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import utils.semantic_seg as transform
import torch.nn.functional as F
from lib import transforms_for_rot, transforms_back_rot, transforms_for_noise, transforms_for_scale, transforms_back_scale, postprocess_scale
from torchvision import transforms
import models.network as models
from mean_teacher import losses, ramps

from utils import Bar, mkdir_p
from tensorboardX import SummaryWriter
from utils.utils import multi_validate, update_ema_variables

parser = argparse.ArgumentParser(description='PyTorch MixMatch Training')
# Optimization options
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', default=8, type=int, metavar='N',
                    help='train batchsize')

# Checkpoints
parser.add_argument('--resume', default='',
                    type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

# Miscs
parser.add_argument('--manualSeed', type=int, default=0, help='manual seed')
# Device options
parser.add_argument('--gpu', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
# Method options
parser.add_argument('--n-labeled', type=int, default=50,
                    help='Number of labeled data')
parser.add_argument('--val-iteration', type=int, default=10,
                    help='Number of labeled data')
parser.add_argument('--data', default='',
                    help='input data path')
parser.add_argument('--out', default='exp/skin/skin50_tcsm',
                    help='Directory to output the result')
parser.add_argument('--alpha', default=0.75, type=float)
parser.add_argument('--lambda-u', default=75, type=float)
parser.add_argument('--T', default=0.5, type=float)
parser.add_argument('--ema-decay', default=0.999, type=float)
parser.add_argument('--num-class', default=2, type=int)

parser.add_argument('--evaluate', action="store_false") # 用于预测时store_flase 
parser.add_argument('--wlabeled', action="store_true")
 
parser.add_argument('--scale', action="store_false") # 用于预测时store_flase

parser.add_argument('--presdo', action="store_true")
parser.add_argument('--tcsm', action="store_true")
parser.add_argument('--tcsm2', action="store_true")
parser.add_argument('--autotcsm', action="store_true")
parser.add_argument('--multitcsm', action="store_true")
parser.add_argument('--baseline', action="store_true")

parser.add_argument('--test_mode', action="store_true")
parser.add_argument('--retina', action="store_true")
# lr
parser.add_argument("--lr_mode", default="cosine", type=str)
parser.add_argument("--lr", default=1e-4, type=float)
parser.add_argument("--warmup_epochs", default=0, type=int)
parser.add_argument("--warmup_lr", default=0.0, type=float)
parser.add_argument("--targetlr", default=0.0, type=float)

#
parser.add_argument('--consistency_type', type=str, default="mse")
parser.add_argument('--consistency', type=float,  default=1.0, help='consistency')
parser.add_argument('--consistency_rampup', type=float,  default=600.0, help='consistency_rampup')

#
parser.add_argument('--initial-lr', default=0.0, type=float,
                    metavar='LR', help='initial learning rate when using linear rampup')
parser.add_argument('--lr-rampup', default=0, type=int, metavar='EPOCHS',
                    help='length of learning rate rampup in the beginning')
parser.add_argument('--lr-rampdown-epochs', default=None, type=int, metavar='EPOCHS',
                    help='length of learning rate cosine rampdown (>= length of training)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}


# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
use_cuda = torch.cuda.is_available()  # print(use_cuda) True


# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
np.random.seed(args.manualSeed)

best_acc = 0  # best test accuracy
NUM_CLASS = args.num_class

from shutil import copyfile


def main():
    global best_acc

    if not os.path.isdir(args.out):
        mkdir_p(args.out)
    copyfile("train_tcsm_mean.py", args.out+"/train_tcsm_mean.py")

    # args.retina参数默认生效为false
    if args.retina:
        mean = [22, 47, 82]
    else:
        mean = [140, 150, 180]
    std = None

    # Data augmentation
    # print(f'==> Preparing skinlesion dataset')
    transform_train = transform.Compose([
        transform.RandomRotationScale(),
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)])

    transform_val = transform.Compose([
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)])

    import dataset.skinlesion as dataset
    train_labeled_set, train_unlabeled_set, val_set, test_set = dataset.get_skinlesion_dataset("./data/skinlesion/",
                                                num_labels=args.n_labeled, 
                                                        transform_train=transform_train,
                                                        transform_val=transform_val,
                                                        transform_forsemi=None)

    # 输出：main:166 2 6 说明这里都有数据
    print("\n main 168:",len(train_labeled_set), len(train_unlabeled_set) ,len(val_set)) 

    # drop_last为True会将多出来不足一个batch的数据丢弃
    labeled_trainloader = data.DataLoader(train_labeled_set, batch_size=args.batch_size, shuffle=True,
                                          num_workers=0, drop_last=False)
    
    if args.baseline:
        unlabeled_trainloader = None
    else:
        unlabeled_trainloader = data.DataLoader(train_unlabeled_set, batch_size=args.batch_size, shuffle=True,
                                                num_workers=0, drop_last=False)

    val_loader = data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    # test_loader = data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=2)

    print("\n main 183:", len(list(labeled_trainloader)))
    print("\n main 184:", len(list(unlabeled_trainloader)))
    print("\n main 185:", len(list(val_loader)))

    # Model
    print("==> creating model")

    def create_model(ema=False):
        model = models.DenseUnet_2d()
        model = model.cuda()

        if ema:
            for param in model.parameters():
                param.detach_()

        return model

    model = create_model()
    ema_model = create_model(ema=True)

    # cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([1.93, 8.06]).cuda())
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.evaluate:
        val_loss, val_result = multi_validate(val_loader, ema_model, criterion, 0, use_cuda, args)
        print("val_loss", val_loss)
        print("Val ema_model : JA, AC, DI, SE, SP \n")
        print(", ".join("%.4f" % f for f in val_result))
        val_loss, val_result = multi_validate(val_loader, model, criterion, 0, use_cuda, args)
        print("val_loss", val_loss)
        print("Val model: JA, AC, DI, SE, SP \n")
        print(", ".join("%.4f" % f for f in val_result))
        return

if __name__ == '__main__':
    main()
