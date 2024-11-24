# -*- coding: utf-8 -*-

# from PIL import Image
import os
import cv2
import sys
import numpy as np
from tqdm import tqdm


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        dirflag = True
    else:
        dirflag = False
    return dirflag


def read_GZ(root_path, in_name):
    img1 = cv2.imread(root_path + 'T1/' + in_name)
    img2 = cv2.imread(root_path + 'T2/' + in_name)
    label = cv2.imread(root_path + '/labels_change/' + in_name)
    return img1, img2, label


# imageDir="./example/images/"  #./Original/Images/Labels     #原大图像数据
# saveDir="./example/" + str(crop_w) + "x" + str(crop_h) + "/image/"    ##裁剪小图像数据
def gen_data(train, names, root_path, saveDir, crop_w,stride):

    num = 0
    crop_h = crop_w
    tbar = tqdm(range(len(names)))

    for i in tbar:
    # for in_name in names:
        in_name=names[i]
        if in_name =='':
            continue
        # print(root_path, in_name)
        I1, I2, cm = read_GZ(root_path , in_name)
        # if train:
        #     I1, I2, cm = read_GZ(root_path+'train/', in_name)
        # else:
        #     I1, I2, cm = read_GZ(root_path + 'val/', in_name)
        I1_ori = I1.astype(np.uint8)
        I2_ori = I2.astype(np.uint8)
        cm_ori = cm.astype(np.uint8)
        # cv2.imwrite('I1_ori.png', I1_ori)
        # cv2.imwrite('I2_ori.png', I2_ori)
        # cv2.imwrite('cm_ori.png', cm_ori)
        h_ori, w_ori, _ = I1_ori.shape
        for ratio in range(1, 999):
            if (w_ori / ratio < 260 or h_ori / ratio < 260) and ratio>1:
                break
            if ratio != 1:
                # print(in_name, 'resize:', ratio)
                h_new = int(h_ori / ratio)
                w_new = int(w_ori / ratio)
                I1 = cv2.resize(I1_ori, (h_new, w_new), interpolation=cv2.INTER_NEAREST)
                I2 = cv2.resize(I2_ori, (h_new, w_new), interpolation=cv2.INTER_NEAREST)
                cm = cv2.resize(cm_ori, (h_new, w_new), interpolation=cv2.INTER_NEAREST)
                I1 = I1.astype(np.uint8)
                I2 = I2.astype(np.uint8)
                cm = cm.astype(np.uint8)

                # sys.exit(0)

            else:
                I1 = I1_ori
                I2 = I2_ori
                cm = cm_ori

            h, w, _ = I1.shape
            padding_h = (h // stride + 1) * stride
            padding_h=padding_h+padding_h%crop_h
            padding_w = (w // stride + 1) * stride
            padding_w = padding_w + padding_w % crop_h
            padding_img_T1 = np.zeros((padding_h, padding_w, 3), dtype=np.uint8)
            padding_img_T2 = np.zeros((padding_h, padding_w, 3), dtype=np.uint8)
            padding_img_cm = np.zeros((padding_h, padding_w), dtype=np.uint8)
            padding_img_T1[0:h, 0:w, :] = I1[:, :, :]
            padding_img_T2[0:h, 0:w, :] = I2[:, :, :]
            padding_img_cm[0:h, 0:w] = cm[:, :, 0]

            h_leave = h % crop_h
            w_leave = w % crop_w
            mask_whole = np.zeros((padding_h, padding_w), dtype=np.uint8)
            for i in range(padding_h // stride):
                for j in range(padding_w // stride):

                    crop_T1 = padding_img_T1[i * stride:i * stride + crop_h, j * stride:j * stride + crop_w, :3]
                    crop_T2 = padding_img_T2[i * stride:i * stride + crop_h, j * stride:j * stride + crop_w, :3]
                    crop_cm = padding_img_cm[i * stride:i * stride + crop_h, j * stride:j * stride + crop_w]
                    _, ch, cw = crop_T1.shape
                    # print(crop_T1.shape,i * stride,i * stride + crop_h, j * stride,j * stride + crop_w)
                    # if (len(crop_cm[crop_cm != 0]) ) / (crop_w * crop_h) > (0.1/ratio) or not train:
                    if train:
                        saveName = in_name.split('.')[0] + '_' + str(i) + '_' + str(j) + ".png"  # 小图像名称，内含小图像的顺序
                        cv2.imwrite(root_path + saveDir[0] + saveName, crop_T1)
                        cv2.imwrite(root_path + saveDir[1] + saveName, crop_T2)
                        cv2.imwrite(root_path + saveDir[2] + saveName, crop_cm)
                        # print(root_path , saveDir[2] , saveName)
                        num = num + 1
                        # print('train num generated: ', num)
                        # break
                        # cv2.imwrite('crop_T1.png', crop_T1)
                        # cv2.imwrite('crop_T2.png', crop_T2)
                        # cv2.imwrite('crop_cm.png', crop_cm)
                    else:
                        saveName = in_name.split('.')[0] + '_' + str(i) + '_' + str(j) + ".png"  # 小图像名称，内含小图像的顺序
                        cv2.imwrite(root_path + saveDir[3] + saveName, crop_T1)
                        cv2.imwrite(root_path + saveDir[4] + saveName, crop_T2)
                        cv2.imwrite(root_path + saveDir[5] + saveName, crop_cm)
                        num = num + 1
                        # print('val num generated: ', num)
    print('data num generated: ', num)

crop_w = 256  # 裁剪图像宽度
crop_h = 256  # 裁剪图像高度
root_path = '/data/project_frb/dataset/CD_Data_GZ/'
# root_path = './'

dataset_name = 'GZ_CDPatch'
# dataset_name = 'GZ_aug2'

saveDir = ['%s/train/T1/' % dataset_name, '%s/train/T2/' % dataset_name, '%s/train/label/' % dataset_name,
           '%s/val/T1/' % dataset_name, '%s/val/T2/' % dataset_name, '%s/val/label/' % dataset_name]
for dirr in saveDir:
    dirflag = mkdir(root_path + dirr)
    if not dirflag:
        break



# filename = open(root_path+'trainnew.txt', 'w')
# dirpath_lable=root_path+'/train/T1'
# imglist = os.listdir(dirpath_lable)
# for i in imglist:
#     # print(i)
#     filename.write(i + '\n')
# filename.close()

fnametrain = 'trainnew.txt'
with open(root_path + fnametrain, "r") as f:  # 打开文件
    f.seek(0)  # 加入这一行代码
    datatrain = f.read()  # 读取文件
datatrain = datatrain.split('\n')
namestrain = datatrain
print('train names list:', namestrain)

#
# filename = open(root_path+'val.txt', 'w')
# dirpath_lable=root_path+'/val/T1'
# imglist = os.listdir(dirpath_lable)
# for i in imglist:
#     # print(i)
#     filename.write(i + '\n')
# filename.close()



fnametest = 'testtnew.txt'
with open(root_path + fnametest, "r") as f:  # 打开文件
    # f.seek(0)  # 加入这一行代码
    datatest = f.read()  # 读取文件

datatest = datatest.split('\n')
namestest = datatest
print('test names list:', namestest)

# if not dirflag:
#     print("The Floder Have Existed!")
#     sys.exit(0)
print('##############train dataset##################')
train = True
stride = 200
gen_data(train, namestrain, root_path, saveDir, crop_w,stride)
print('##############test dataset##################')
train = False
# stride = 200
stride = crop_w
gen_data(train, namestest, root_path, saveDir, crop_w,stride)
