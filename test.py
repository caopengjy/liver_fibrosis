import os
import copy
import collections
from time import time

import torch
import numpy as np
import pandas as pd
import scipy.ndimage as ndimage
import SimpleITK as sitk
import skimage.measure as measure
import skimage.morphology as morphology

from net.ResUNet import ResUNet
from utilities.calculate_metrics import Metirc

import parameter as para

os.environ['CUDA_VISIBLE_DEVICES'] = para.gpu

# 为了计算dice_global定义的两个变量
dice_intersection = 0.0
dice_union = 0.0

file_name = []  # 文件名称
time_pre_case = []  # 单例数据消耗时间

# 定义网络并加载参数
net = torch.nn.DataParallel(ResUNet(training=False)).cuda()
net.load_state_dict(torch.load(para.module_path))
net.eval()

for file_index, file in enumerate(os.listdir(para.test_ct_path)):

    start = time()

    file_name.append(file)

    # 将MRI读入内存
    ct = sitk.ReadImage(os.path.join(para.test_ct_path, file), sitk.sitkInt16)
    ct_array = sitk.GetArrayFromImage(ct)

    origin_shape = ct_array.shape
    # print('originshape' , origin_shape)


    # 将灰度值在阈值之外的截断掉
    # ct_array[ct_array > para.upper] = para.upper
    # ct_array[ct_array < para.lower] = para.lower

    # min max 归一化
    ct_array = ct_array.astype(np.float32)
    ct_array = ct_array / 200
    print('ctarray', ct_array.shape)


    # 对磁共振图像使用双三次算法进行插值，插值之后的array依然是int16
    ct_array = ndimage.zoom(ct_array, (1, para.down_scale, para.down_scale), order=3)
    # print(ct_array.shape)

    # 对slice过少的数据使用padding
    too_small = False
    if ct_array.shape[0] < para.size:
        depth = ct_array.shape[0]
        temp = np.ones((para.size, int(ct_array.shape[1]*2 * para.down_scale), int(ct_array.shape[2]*2 * para.down_scale))) * para.lower
        temp[0: depth] = ct_array
        ct_array = temp
        too_small = True*2

    # 滑动窗口取样预测
    start_slice = 0
    end_slice = start_slice + para.size - 1
    count = np.zeros((ct_array.shape[0], ct_array.shape[1]*2, ct_array.shape[2]*2), dtype=np.int16)
    probability_map = np.zeros((ct_array.shape[0], ct_array.shape[1]*2, ct_array.shape[2]*2), dtype=np.float32)

    with torch.no_grad():
        while end_slice < ct_array.shape[0]:
            ct_tensor = torch.FloatTensor(ct_array[start_slice: end_slice + 1]).cuda()
            ct_tensor = ct_tensor.unsqueeze(dim=0).unsqueeze(dim=0)

            outputs = net(ct_tensor)

            count[start_slice: end_slice + 1] += 1
            # print('promap', probability_map.shape, outputs.cpu().detach().numpy().shape)
            probability_map[start_slice: end_slice + 1] += np.squeeze(outputs.cpu().detach().numpy())

            # 由于显存不足，这里直接保留ndarray数据，并在保存之后直接销毁计算图
            del outputs

            start_slice += para.stride
            end_slice = start_slice + para.size - 1

        if end_slice != ct_array.shape[0] - 1:
            end_slice = ct_array.shape[0] - 1
            start_slice = end_slice - para.size + 1

            ct_tensor = torch.FloatTensor(ct_array[start_slice: end_slice + 1]).cuda()
            ct_tensor = ct_tensor.unsqueeze(dim=0).unsqueeze(dim=0)
            outputs = net(ct_tensor)

            count[start_slice: end_slice + 1] += 1
            probability_map[start_slice: end_slice + 1] += np.squeeze(outputs.cpu().detach().numpy())

            del outputs

        pred_seg = np.zeros_like(probability_map)
        pred_seg[probability_map >= (para.threshold * count)] = 1
        #
        # if too_small:
        #     temp = np.zeros((depth, 512, 512), dtype=np.float32)
        #     temp += pred_seg[0: depth]
        #     pred_seg = temp



    # 对肝脏进行最大连通域提取,移除细小区域,并进行内部的空洞填充
    pred_seg = pred_seg.astype(np.uint8)
    liver_seg = copy.deepcopy(pred_seg)
    liver_seg = measure.label(liver_seg, 4)
    props = measure.regionprops(liver_seg)

    max_area = 0
    max_index = 0
    for index, prop in enumerate(props, start=1):
        if prop.area > max_area:
            max_area = prop.area
            max_index = index

    liver_seg[liver_seg != max_index] = 0
    liver_seg[liver_seg == max_index] = 1

    liver_seg = liver_seg.astype(np.bool)
    morphology.remove_small_holes(liver_seg, para.maximum_hole, connectivity=2, in_place=True)
    liver_seg = liver_seg.astype(np.uint8)

    # 将预测的结果保存为nii数据
    pred_seg = sitk.GetImageFromArray(liver_seg)

    pred_seg.SetDirection(ct.GetDirection())
    pred_seg.SetOrigin(ct.GetOrigin())
    pred_seg.SetSpacing(ct.GetSpacing())

    sitk.WriteImage(pred_seg, os.path.join(para.pred_path, file.replace('volume', 'pred')))

    speed = time() - start
    time_pre_case.append(speed)

    print(file_index, 'this case use {:.3f} s'.format(speed))
    # print(file_name)
    print('-----------------------')
