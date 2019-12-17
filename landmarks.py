#!/usr/bin/env python
# encoding: utf-8
"""
@author: Shanda Lau 刘祥德
@license: (C) Copyright 2019-now, Node Supply Chain Manager Corporation Limited.
@contact: shandalaulv@gmail.com
@software: 
@file: landmarks.py
@time: 12/17/19 4:40 PM
@version 1.0
@desc:
"""
# 导入系统库并定义辅助函数
from pprint import pformat
import numpy as np

# import PythonSDK
from PythonSDK.facepp import API, File
from pathlib import Path
import os

# 导入图片处理类
import PythonSDK.ImagePro


# 此方法专用来打印api返回的信息
def print_result(hit, result):
    print(hit)
    print('\n'.join("  " + i for i in pformat(result, width=75).split('\n')))


def printFuctionTitle(title):
    return "\n" + "-" * 60 + title + "-" * 60;


def generate_landmarks(file_lists, output: Path):
    for f in file_lists:
        if f.is_dir():
            output_dir = output / f.name
            output_dir.mkdir(parents=True, exist_ok=True)
            imgs = list(f.glob('*.png'))
            for img in imgs:
                print(img)
                res = api.thousandlandmark(image_file=File(img),
                                           return_landmark="face,left_eyebrow,right_eyebrow,left_eye_eyelid,right_eye_eyelid,nose,mouth")
                if 'face' not in res:
                    continue
                if 'landmark' not in res['face']:
                    continue
                landmarks = res['face']['landmark']
                landmarks_list = []
                # print_result(printFuctionTitle("人脸关键点检测"), landmarks)
                for region, landmarks_dict in landmarks.items():
                    for k, landmark in landmarks_dict.items():
                        landmarks_list.append([landmark['x'], landmark['y']])
                landmarks_list = np.array(landmarks_list)
                txt_name = os.path.splitext(os.path.basename(img))[0] + '.txt'
                np.savetxt(str(output_dir / txt_name), landmarks_list, fmt="%d")
                # return


# 初始化对象，进行api的调用工作
api = API()

dataset_name = 'AF_dataset'
output_name = 'landmarks'

dataset = Path('AF_dataset')
output = Path(output_name)
output.mkdir(parents=True, exist_ok=True)

file_lists = list(dataset.glob('*'))

generate_landmarks(file_lists, output)
