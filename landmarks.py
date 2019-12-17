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
import threading
import multiprocessing

# import PythonSDK
from PythonSDK.facepp import API, File
from pathlib import Path
import os
import json
# 导入图片处理类
import PythonSDK.ImagePro


# 此方法专用来打印api返回的信息
def print_result(hit, result):
    print(hit)
    print('\n'.join("  " + i for i in pformat(result, width=75).split('\n')))


def printFuctionTitle(title):
    return "\n" + "-" * 60 + title + "-" * 60;


def generate_landmarks(file_lists, output: Path):
    with multiprocessing.Pool() as p:
        for f in file_lists:
            if f.is_dir():
                output_dir = output / f.name
                output_dir.mkdir(parents=True, exist_ok=True)
                imgs = list(f.glob('*.png'))
                for img in imgs:
                    # handle_img(img, output_dir)
                    p.apply_async(handle_img, (img, output_dir,))
                    # return
                    # task = threading.Thread(target=handle_img, args=(img, output_dir,))
                    # task.start()
                    # threads.append(task)
            elif f.is_file():
                p.apply_async(handle_img, (f, output,))
        p.close()
        p.join()

        # return


def handle_img(img, output_dir):
    not_fetch = True
    res = None
    output_landmarks_dir = output_dir / 'landmarks'
    output_json_dir = output_dir / 'results'
    output_landmarks_dir.mkdir(exist_ok=True, parents=True)
    output_json_dir.mkdir(exist_ok=True, parents=True)

    while not_fetch:
        try:
            res = api.thousandlandmark(image_file=File(img),
                                       return_landmark="face,left_eyebrow,right_eyebrow,left_eye_eyelid,right_eye_eyelid,nose,mouth")
            not_fetch = False
        except Exception as e:
            print(e)
    if 'face' not in res:
        return None
    if 'landmark' not in res['face']:
        return None
    landmarks = res['face']['landmark']
    landmarks_list = []
    print(img)
    # print_result(printFuctionTitle("人脸关键点检测"), landmarks)
    for region, landmarks_dict in landmarks.items():
        for k, landmark in landmarks_dict.items():
            landmarks_list.append([landmark['x'], landmark['y']])
    landmarks_list = np.array(landmarks_list)
    img_name = os.path.splitext(os.path.basename(img))[0]
    txt_name = img_name + '.txt'
    np.savetxt(str(output_landmarks_dir / txt_name), landmarks_list, fmt="%d")

    output_json = output_json_dir / (img_name + '.json')
    fw = open(output_json, 'w')
    fw.write(json.dumps(res, indent=4))
    fw.close()


# 初始化对象，进行api的调用工作
api = API()

dataset_name = 'CelebA'
output_name = 'CelebA-landmarks'
dataset = Path(dataset_name)

output = Path(output_name)
output.mkdir(parents=True, exist_ok=True)

file_lists = list(dataset.glob('*'))

generate_landmarks(file_lists, output)
