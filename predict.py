import os
os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"
import cv2
import torch.utils.data as data
import numpy as np
import torch
import utils.semantic_seg as transform
import models.network as models
import dataset.skinlesion as dataset
from PIL import Image


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
    val_loader = data.DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)

    model = create_model()
    ema_model = create_model(ema=True)

    ema_model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets, name) in enumerate(val_loader):
            inputs = inputs.cuda()

            # init 新建空的原图大小numpy数组 
            score_box = np.zeros((inputs.shape[0], 248, 248), dtype='float32')
            num_box = np.zeros((inputs.shape[0], 248, 248), dtype='uint8')
            # compute output
            outputs = ema_model(inputs[:,:, 0:224,0:224], dropout=False)
            outputs = torch.softmax(outputs, dim=1)
            score_box[:, 0:224, 0:224] = outputs[:,1,:,:].cpu().detach().numpy()
            num_box[:, 0:224, 0:224] = 1

            outputs = ema_model(inputs[:, :, 24:248, 24:248], dropout=False)
            outputs = torch.softmax(outputs, dim=1)
            score_box[:, 24:248, 24:248] += outputs[:,1,:,:].cpu().detach().numpy()
            num_box[:, 24:248, 24:248] += 1

            outputs = ema_model(inputs[:, :, 0:224,24:248], dropout=False)
            outputs = torch.softmax(outputs, dim=1)
            score_box[:, 0:224, 24:248] += outputs[:,1,:,:].cpu().detach().numpy()
            num_box[:, 0:224, 24:248] += 1

            outputs = ema_model(inputs[:, :, 24:248,0:224], dropout=False)
            outputs = torch.softmax(outputs, dim=1)
            score_box[:,24:248,0:224] += outputs[:,1,:,:].cpu().detach().numpy()
            num_box[:,24:248,0:224] += 1

            score = score_box / (num_box + 1e-5)
            print(score.shape)
            img = score[0]
            img[img >= 0.5] = 255
            img[img < 0.5] = 0
            img = Image.fromarray(img.astype(np.uint8))
            img.save('./result/result_ema_model.png')


    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets, name) in enumerate(val_loader):
            inputs = inputs.cuda()

            # init
            score_box = np.zeros((inputs.shape[0], 248, 248), dtype='float32')
            num_box = np.zeros((inputs.shape[0], 248, 248), dtype='uint8')

            # compute output
            outputs = model(inputs[:,:, 0:224,0:224], dropout=False)
            # Lx1 = criterion(outputs, targets[:,0:224,0:224].long())
            outputs = torch.softmax(outputs, dim=1)
            score_box[:, 0:224, 0:224] = outputs[:,1,:,:].cpu().detach().numpy()
            num_box[:, 0:224, 0:224] = 1

            outputs = model(inputs[:, :, 24:248, 24:248], dropout=False)
            # Lx2 = criterion(outputs, targets[:, 24:248, 24:248].long())
            outputs = torch.softmax(outputs, dim=1)
            score_box[:, 24:248, 24:248] += outputs[:,1,:,:].cpu().detach().numpy()
            num_box[:, 24:248, 24:248] += 1

            outputs = model(inputs[:, :, 0:224,24:248], dropout=False)
            # Lx3 = criterion(outputs, targets[:, 0:224,24:248].long())
            outputs = torch.softmax(outputs, dim=1)
            score_box[:, 0:224, 24:248] += outputs[:,1,:,:].cpu().detach().numpy()
            num_box[:, 0:224, 24:248] += 1

            outputs = model(inputs[:, :, 24:248,0:224], dropout=False)
            # Lx4 = criterion(outputs, targets[:, 24:248,0:224].long())
            outputs = torch.softmax(outputs, dim=1)
            score_box[:,24:248,0:224] += outputs[:,1,:,:].cpu().detach().numpy()
            num_box[:,24:248,0:224] += 1

            score = score_box / (num_box + 1e-5)
            print(score.shape)
            img = score[0]
            img[img >= 0.5] = 255
            img[img < 0.5] = 0
            img = Image.fromarray(img.astype(np.uint8))
            img.save('./result/result_model.png')


if __name__ == '__main__':
    main()
