#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : train.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/11/16 上午11:08
# @ Software   : PyCharm
#-------------------------------------------------------
import os
import json

import torch

coco_dataset = '/media/alex/AC6A2BDB6A2BA0D6/alex_dataset/COCO_2017'

train_instance_path = os.path.join(coco_dataset, 'annotations', 'instances_train2017.json')


def main():
    # ++++++++++++++++++++++++++++display coco dataset info++++++++++++++++++++++++++++++++++++++++++++

    # train_instance = json.load(open(train_instance_path, 'r'))
    # print('Done')

    l = [4, 1, 2, 3]
    l.sort(key=lambda x: abs(x-2))
    print(l)
if __name__ == "__main__":
    main()