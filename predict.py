import argparse
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
from PIL import Image
import torch.utils.data as data
import numpy as np
import torch
import utils.semantic_seg as transform
import torch.nn.functional as F
import models.network as models
from mean_teacher import losses, ramps
from utils.utils import multi_validate, update_ema_variables
import dataset.skinlesion as dataset
import torch.nn as nn
import torch.optim as optim

parser = argparse.ArgumentParser(description='PyTorch MixMatch Training')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', default=8, type=int, metavar='N',help='train batchsize')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--manualSeed', type=int, default=0, help='manual seed')
parser.add_argument('--gpu', default='0', type=str,help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--n-labeled', type=int, default=50,help='Number of labeled data')
parser.add_argument('--val-iteration', type=int, default=10,help='Number of labeled data')
parser.add_argument('--data', default='', help='input data path')
parser.add_argument('--out', default='exp/skin/skin50_tcsm',help='Directory to output the result')
parser.add_argument('--alpha', default=0.75, type=float)
parser.add_argument('--lambda-u', default=75, type=float)
parser.add_argument('--T', default=0.5, type=float)
parser.add_argument('--ema-decay', default=0.999, type=float)
parser.add_argument('--num-class', default=2, type=int)
parser.add_argument('--evaluate', action="store_false") # 用于预测时store_flase 
parser.add_argument('--wlabeled', action="store_true")
parser.add_argument('--scale', action="store_false") # 用于预测时store_false
parser.add_argument('--presdo', action="store_true")
parser.add_argument('--tcsm', action="store_true")
parser.add_argument('--tcsm2', action="store_true")
parser.add_argument('--autotcsm', action="store_true")
parser.add_argument('--multitcsm', action="store_true")
parser.add_argument('--baseline', action="store_true")
parser.add_argument('--test_mode', action="store_true")
parser.add_argument('--retina', action="store_true")
parser.add_argument("--lr_mode", default="cosine", type=str)
parser.add_argument("--lr", default=1e-4, type=float)
parser.add_argument("--warmup_epochs", default=0, type=int)
parser.add_argument("--warmup_lr", default=0.0, type=float)
parser.add_argument("--targetlr", default=0.0, type=float)
parser.add_argument('--consistency_type', type=str, default="mse")
parser.add_argument('--consistency', type=float,  default=1.0, help='consistency')
parser.add_argument('--consistency_rampup', type=float,  default=600.0, help='consistency_rampup')
parser.add_argument('--initial-lr', default=0.0, type=float,metavar='LR', help='initial learning rate when using linear rampup')
parser.add_argument('--lr-rampup', default=0, type=int, metavar='EPOCHS',help='length of learning rate rampup in the beginning')
parser.add_argument('--lr-rampdown-epochs', default=None, type=int, metavar='EPOCHS',help='length of learning rate cosine rampdown (>= length of training)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',help='momentum')
args = parser.parse_args()



def create_model(ema=False):
    model = models.DenseUnet_2d()
    model = model.cuda()
    if ema:
        for param in model.parameters():
            param.detach_()
    return model


def main():
    val_data_img = []
    val_label_img = []
    mean = [140, 150, 180]
    std = None

    transform_val = transform.Compose([
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)])


    img = cv2.imread("./data/skinlesion/myValid_Data248/ISIC_0000009.png", -1)
    val_data_img.append(img)

    img = cv2.imread("./data/skinlesion/myValid_Label248/ISIC_0000009_segmentation.png", 0)
    val_label_img.append(img)

    val_dataset = dataset.skinlesion_labeled(val_data_img, val_label_img, name="009.png", indexs=None, transform=transform_val)
    val_loader = data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = create_model()
    ema_model = create_model(ema=True)
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([1.93, 8.06]).cuda())
    use_cuda = torch.cuda.is_available()

    val_loss, val_result = multi_validate(val_loader, ema_model, criterion, 0, use_cuda, args)
    print("val_loss", val_loss)
    print("val_result: JA, AC, DI, SE, SP:", val_result)
    
    val_loss, val_result = multi_validate(val_loader, model, criterion, 0, use_cuda, args)
    print("val_loss", val_loss)
    print("Val model: JA, AC, DI, SE, SP:", val_result)



if __name__ == '__main__':
    main()
