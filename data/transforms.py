#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : transforms.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/11/18 下午2:15
# @ Software   : PyCharm
#-------------------------------------------------------

import math
import random
import numpy as np
import skimage
import torch
import torchvision
from PIL import Image, ImageDraw
import torchvision.transforms as transforms


class Resizer(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, min_side, max_side=1000, interpolation=Image.BILINEAR):
        """

        :param sample:
        :param min_side:
        :param max_side:(int) when size is int, limit the image longer size to max_size.
                        This is essential to limit the usage of GPU memory.
        :param interpolation:
        :return:
        """
        assert isinstance(min_side, int)
        self.min_side = min_side
        self.max_side = max_side
        self.interpolation = interpolation

    def __call__(self, sample):

        img, annots = sample['img'], sample['annot']
        width, height = img.size

        # step 1 aspect ratio resize
        size_min = min(width, height)
        size_max = max(width, height)

        # rescale the image so the smallest side is min_side
        scale = float(self.min_side) / size_min

        if scale * size_max > self.max_side:
            scale = float(size_max) / size_max

        w_size = int(width * scale)
        h_size = int(height * scale)
        # resize the image with the computed scale
        img = img.resize((w_size, h_size), self.interpolation)


        # step 2 padding ensure the image size to be a multiple of 32
        w_pad = 0 if w_size % 32 == 0 else 32 - w_size % 32
        h_pad =  0 if h_size % 32 == 0 else 32 - h_size % 32

        img = img.crop((0, 0, w_size + w_pad, h_size + h_pad))  # padding width
        annots[:, :4] *= scale

        return {'img': img, 'annot': torch.as_tensor(annots), 'scale': scale}


class RandomCrop(object):
    """A crop of random size of (0.08 to 1.0) of the original size and a random
    aspect ratio of 3/4 to 4/3 of the original aspect ratio is made.."""

    def __init__(self):
        """

        :param sample:
        :param min_side:
        :param max_side:(int) when size is int, limit the image longer size to max_size.
                        This is essential to limit the usage of GPU memory.
        :param interpolation:
        :return:
        """
        pass

    def __call__(self, sample):

        img, annots = sample['img'], sample['annot']

        success = False
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(0.56, 1.0) * area
            aspect_ratio = random.uniform(3. / 4, 4. / 3)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                x = random.randint(0, img.size[0] - w)
                y = random.randint(0, img.size[1] - h)
                success = True
                break

        # Fallback
        if not success:
            w = h = min(img.size[0], img.size[1])
            x = (img.size[0] - w) // 2
            y = (img.size[1] - h) // 2


        print(x)
        img = img.crop((x, y, x + w, y + h))
        annots -= [x, y, x, y, 0]

        annots[:, 0] = annots[:, 0].clip(min=0, max=w - 1)
        annots[:, 2] = annots[:, 2].clip(min=0, max=w - 1)
        annots[:, 1::2] = annots[:, 1::2].clip(min=0, max=h - 1)

        return {'img': img, 'annot': torch.tensor(annots)}


class CenterCrop(object):
    """Crops the given PIL Image at the center."""

    def __init__(self, size):
        """

        :param size: size (tuple): desired output size of (w,h).
        """

        self.size = size

    def __call__(self, sample):

        img, annots = sample['img'], sample['annot']

        w, h = img.size
        ow, oh = self.size
        i = int(round((h - oh) / 2.))
        j = int(round((w - ow) / 2.))
        img = img.crop((j, i, j + ow, i + oh))
        annots -= [j, i, j, i, 0]
        annots[:, 0] = annots[:, 0].clip(min=0, max=ow - 1)
        annots[:, 2] = annots[:, 2].clip(min=0, max=ow - 1)
        annots[:, 1::2] = annots[:, 1::2].clip(min=0, max=oh - 1)

        return {'img': img, 'annot': torch.tensor(annots)}


class RandomHorizonFlip(object):
    """Randomly flip the given PIL Image."""
    def __init__(self):

        pass

    def __call__(self, sample):

        img, annots = sample['img'], sample['annot']

        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            w = img.width
            xmin = w - annots[:, 2]
            xmax = w - annots[:, 0]
            annots[:, 0] = xmin
            annots[:, 2] = xmax

        return {'img': img, 'annot': torch.as_tensor(annots)}


class Normalizer(object):

    def __init__(self, mean, std):
        if mean == None:
            self.mean = torch.tensor([[[0.485, 0.456, 0.406]]])
        else:
            self.mean = torch.tensor([[mean]])

        if std == None:
            self.std = torch.tensor([[[0.229, 0.224, 0.225]]])
        else:
            self.std = torch.tensor([[std]])

    def __call__(self, sample):

        image, annots, scale = sample['img'], sample['annot'], sample['scale']

        # convert to tensor
        image = torch.as_tensor(np.array(image), dtype=torch.float32)
        annots = torch.as_tensor(annots)
        # convert to (0., 1.)
        image = image / 255.

        image = (image - self.mean) / self.std

        return {'img':image, 'annot': annots, 'scale': scale}

def test():
    from data.dataset import VOCDataset
    from configs.cfgs import args
    from utils.tools import draw_boxes

    # img = Image.open('/media/alex/AC6A2BDB6A2BA0D6/alex_dataset/pascal_voc/test/VOC2007/JPEGImages/000001.jpg')

    voc_dataset = VOCDataset(args.train_data, num_classes=20)
    sample = voc_dataset[7]

    random_crop = RandomCrop()
    random_flip  = RandomHorizonFlip()
    resize = Resizer(min_side=608)
    normalizer = Normalizer()
    crop_sample = random_crop(sample)
    flip_sample = random_flip(crop_sample)
    resize_sample = resize(flip_sample)
    normalize_sample = normalizer(resize_sample)

    resize_img = draw_boxes(resize_sample['img'], resize_sample['annot'][:, :4])
    print(resize_sample['img'].size, resize_sample['annot'][:, :4])
    resize_img.show()

    print('Done')





if __name__ == "__main__":
    test()
