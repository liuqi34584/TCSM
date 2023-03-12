import os
os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"
import cv2
import torch.utils.data as data
import numpy as np
import torch
import utils.semantic_seg as transform
import models.network as models
import dataset.skinlesion as dataset
import matplotlib.pyplot as plt


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

            # 新建空的原图大小numpy数组 
            score_box = np.zeros((inputs.shape[0], 248, 248), dtype='float32') 

            # 将三通道原图[1, 3, 248, 248]输入模型,然后softmax处理
            outputs = ema_model(inputs[:,:, 0:224,0:224], dropout=False)
            outputs = torch.softmax(outputs, dim=1)

            # 将概率矩阵传递给score_box
            score_box[:, 0:224, 0:224] = outputs[:,1,:,:].cpu().detach().numpy()


    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets, name) in enumerate(val_loader):
            inputs = inputs.cuda()

            # 新建空的原图大小numpy数组 
            score_box = np.zeros((inputs.shape[0], 248, 248), dtype='float32') 

            # 将三通道原图[1, 2, 224, 224]输入模型,然后softmax处理
            outputs = model(inputs[:,:, 0:224,0:224], dropout=False)
            outputs = torch.softmax(outputs, dim=1)

            # 将概率矩阵传递给score_box
            score_box[:, 0:224, 0:224] = outputs[:,1,:,:].cpu().detach().numpy()


if __name__ == '__main__':
    main()
