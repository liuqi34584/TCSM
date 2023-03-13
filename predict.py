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
from scipy import ndimage
from skimage import measure

# 评价指标计算
def post_process_evaluate(x, target):

    JA_sum, AC_sum, DI_sum, SE_sum, SP_sum = [],[],[],[],[]

    for i in range(x.shape[0]):
        x_tmp = x[i]
        target_tmp = target[i]

        x_tmp[x_tmp >= 0.5] = 1
        x_tmp[x_tmp <= 0.5] = 0
        x_tmp = np.array(x_tmp, dtype='uint8')
        x_tmp = ndimage.binary_fill_holes(x_tmp).astype(int)

        # only reserve largest connected component.
        box = []
        [lesion, num] = measure.label(x_tmp, return_num=True)
        if num == 0:
            JA_sum.append(0)
            AC_sum.append(0)
            DI_sum.append(0)
            SE_sum.append(0)
            SP_sum.append(0)
        else:
            region = measure.regionprops(lesion)
            for i in range(num):
                box.append(region[i].area)
            label_num = box.index(max(box)) + 1
            lesion[lesion != label_num] = 0
            lesion[lesion == label_num] = 1

            #  calculate TP,TN,FP,FN
            TP = float(np.sum(np.logical_and(lesion == 1, target_tmp == 1)))
            # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
            TN = float(np.sum(np.logical_and(lesion == 0, target_tmp == 0)))

            # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
            FP = float(np.sum(np.logical_and(lesion == 1, target_tmp == 0)))

            # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
            FN = float(np.sum(np.logical_and(lesion == 0, target_tmp == 1)))

            #  calculate JA, Dice, SE, SP
            JA = TP / ((TP + FN + FP + 1e-7))
            AC = (TP + TN) / ((TP + FP + TN + FN + 1e-7))
            DI = 2 * TP / ((2 * TP + FN + FP + 1e-7))
            SE = TP / (TP + FN+1e-7)
            SP = TN / ((TN + FP+1e-7))

            JA_sum.append(JA); AC_sum.append(AC); DI_sum.append(DI); SE_sum.append(SE); SP_sum.append(SP)

    return sum(JA_sum), sum(AC_sum), sum(DI_sum), sum(SE_sum), sum(SP_sum)


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
            
            results = post_process_evaluate(score, targets.cpu().detach().numpy())
            print("\nJA:",results[0],"\nAC:",results[1],"\nDI:",results[2],"\nSE:",results[3],"\nSP:",results[4])

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

            results = post_process_evaluate(score, targets.cpu().detach().numpy())
            print("\nJA:",results[0],"\nAC:",results[1],"\nDI:",results[2],"\nSE:",results[3],"\nSP:",results[4])

            print(score.shape)
            img = score[0]
            img[img >= 0.5] = 255
            img[img < 0.5] = 0
            img = Image.fromarray(img.astype(np.uint8))
            img.save('./result/result_model.png')


if __name__ == '__main__':
    main()
