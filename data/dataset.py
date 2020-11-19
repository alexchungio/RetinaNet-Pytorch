#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : dataset.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/11/18 下午2:15
# @ Software   : PyCharm
#-------------------------------------------------------
import os
import sys
import numpy as np
from PIL import Image
from pycocotools.coco import COCO

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader


from configs.cfgs import args
from utils.tools import read_class_names, draw_boxes

class CocoDataset(Dataset):
    """Coco dataset."""

    def __init__(self, root_dir, set_name='train2017', transform=None):
        """
        Args:
            root_dir (string): COCO directory.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.set_name = set_name
        self.transform = transform

        self.coco      = COCO(os.path.join(self.root_dir, 'annotations', 'instances_' + self.set_name + '.json'))
        self.image_ids = self.coco.getImgIds()

        self.load_classes()

    def load_classes(self):
        # load class names (name -> label)
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])

        self.classes             = {}
        self.coco_labels         = {}
        self.coco_labels_inverse = {}
        for c in categories:
            self.coco_labels[len(self.classes)] = c['id']
            self.coco_labels_inverse[c['id']] = len(self.classes)
            self.classes[c['name']] = len(self.classes)

        # also load the reverse (label -> name)
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):

        img = self.load_image(idx)
        annot = self.load_annotations(idx)
        sample = {'img': img, 'annot': annot}
        if self.transform:
            sample = self.transform(sample)

        return sample

    def load_image(self, image_index):
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        img_path   = os.path.join(self.root_dir, self.set_name, image_info['file_name'])

        img = Image.open(img_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return img

    def load_annotations(self, image_index):
        # get ground truth annotations
        annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_index], iscrowd=False)
        annotations     = np.zeros((0, 5))
        # some images appear to miss annotations (like image with id 257034)
        if len(annotations_ids) == 0:
            return annotations

        # parse annotations
        coco_annotations = self.coco.loadAnns(annotations_ids)
        for idx, a in enumerate(coco_annotations):

            # some annotations have basically no width / height, skip them
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue
            annotation        = np.zeros((1, 5))
            annotation[0, :4] = a['bbox']
            annotation[0, 4]  = self.coco_label_to_label(a['category_id'])
            annotations       = np.append(annotations, annotation, axis=0)

        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations

    def coco_label_to_label(self, coco_label):
        return self.coco_labels_inverse[coco_label]

    def label_to_coco_label(self, label):
        return self.coco_labels[label]

    def image_aspect_ratio(self, image_index):
        image = self.coco.loadImgs(self.image_ids[image_index])[0]
        return float(image['width']) / float(image['height'])

    def num_classes(self):
        return 80


class VOCDataset(Dataset):
    def __init__(self, annotation_file, num_classes, transform=None, is_training=False):
        """
        :param dataset: dataset annotation file
        :param transform:
        """
        self.annotation_file = annotation_file
        self.num_classes = num_classes
        self.transform = transform
        self.is_training = is_training

        try:
            with self._open_file(self.annotation_file) as file:
                self.names, self.annotations = self._read_annotations(file)
        except ValueError as e:
            raise (ValueError('invalid txt annotations file: {}: {}'.format(self.annotation_file, e)))

    def _parse(self, value, function, fmt):
        """
        Parse a string into a value, and format a nice ValueError if it fails.
        Returns `function(value)`.
        Any `ValueError` raised is catched and a new `ValueError` is raised
        with message `fmt.format(e)`, where `e` is the caught `ValueError`.
        """
        try:
            return function(value)
        except ValueError as e:
            raise ValueError(fmt.format(e))

    def _open_file(self, path):
        """
        Open a file with flags suitable for csv.reader.
        This is different for python2 it means with mode 'rb',
        for python3 this means 'r' with "universal newlines".
        """
        if sys.version_info[0] < 3:
            return open(path, 'rb')
        else:
            return open(path, 'r', newline='')

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):

        img = self.load_image(self.names[idx])
        annot = self.annotations[idx]
        sample = {'img': img, 'annot': annot}
        if self.transform:
            img, bbox, label = self.transform(sample, self.is_training)

        return sample

    def load_image(self, image_path):

        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')

        return img


    def _read_annotations(self, file_reader):

        images, annotations = [], []
        for index, line in enumerate(file_reader):
            try:
                img_path, height, width, annotation_info = line.strip().split(' ')
            except ValueError:
                raise ValueError(
                    'line {}: format should be \'img_file, height, width, bbox_label\' or \'img_file,,,,,\''.format(line))

            # update img_path
            images.append(img_path)

            img_ann = np.zeros((0, 5))
            # some images appear to miss annotations (like image with id 257034)
            if len(annotation_info) == 0:
                pass
            else:
                # update bbox and label
                bbox_labels = annotation_info.split(',')
                num_target = len(bbox_labels) // 5

                for i in range(num_target):
                    x_min = bbox_labels[5 * i]
                    y_min = bbox_labels[5 * i + 1]
                    x_max = bbox_labels[5 * i + 2]
                    y_max = bbox_labels[5 * i + 3]
                    label = bbox_labels[5 * i + 4]

                    x_min = self._parse(x_min, float, 'line {}: malformed x1: {{}}'.format(line))
                    y_min = self._parse(y_min, float, 'line {}: malformed y1: {{}}'.format(line))
                    x_max = self._parse(x_max, float, 'line {}: malformed x2: {{}}'.format(line))
                    y_max = self._parse(y_max, float, 'line {}: malformed y2: {{}}'.format(line))
                    label = self._parse(label, int, 'line {}: malformed label: {{}}'.format(line))


                    # Check that the bounding box is valid.
                    if x_max <= x_min:
                        raise ValueError('line {}: x2 ({}) must be higher than x1 ({})'.format(line, x_max, x_min))
                    if y_max <= y_min:
                        raise ValueError('line {}: y2 ({}) must be higher than y1 ({})'.format(line, y_max, y_min))

                    # check if the current class name is correctly present
                    if label >= self.num_classes:
                        raise ValueError('line {}: class out of range: {}'.format(line, label))

                    annot = np.zeros((1, 5))

                    annot[0, 0] = x_min
                    annot[0, 1] = y_min
                    annot[0, 2] = x_max
                    annot[0, 3] = y_max
                    annot[0, 4] = label

                    img_ann = np.append(img_ann, annot, axis=0)

                annotations.append(img_ann)

        return images, annotations


def collate_fn(data):
    """
    ensure images remains the same shape in one batch
    :param data:
    :return:
    """
    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    scales = [s['scale'] for s in data]

    widths = [int(s.shape[0]) for s in imgs]
    heights = [int(s.shape[1]) for s in imgs]
    batch_size = len(imgs)

    max_width = max(widths)
    max_height = max(heights)

    padded_imgs = torch.zeros(batch_size, max_width, max_height, 3)

    for i in range(batch_size):
        img = imgs[i]
        padded_imgs[i, :int(img.shape[0]), :int(img.shape[1]), :] = img

    max_num_annots = max(annot.shape[0] for annot in annots)

    if max_num_annots > 0:

        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1

        if max_num_annots > 0:
            for idx, annot in enumerate(annots):
                # print(annot.shape)
                if annot.shape[0] > 0:
                    annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1

    padded_imgs = padded_imgs.permute(0, 3, 1, 2)

    return {'img': padded_imgs, 'annot': annot_padded, 'scale': scales}


if __name__ == "__main__":

    # coco_dataset = '/media/alex/AC6A2BDB6A2BA0D6/alex_dataset/COCO_2017'
    # coco_dataset = CocoDataset(coco_dataset)
    # sample = coco_dataset[0]

    voc_dataset = VOCDataset(args.train_data, num_classes=20)

    sample = voc_dataset[5]
    img = draw_boxes(sample['img'], sample['annot'][:, :4])
    img.show()

    print('Done')

