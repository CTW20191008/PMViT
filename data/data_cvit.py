# --------------------------------------------------------
# SimMIM
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Zhenda Xie
# --------------------------------------------------------

import math
import random
import numpy as np
import cv2
import time

import torch
import torch.distributed as dist
import torchvision.transforms as T
from torch.utils.data import DistributedSampler
from torch.utils.data._utils.collate import default_collate
from torchvision.datasets import ImageFolder
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from .data_load_x import DataLoaderX
import torch.nn as nn
import numpy as np
from PIL import Image


class CannyGenerator:
    def __init__(self, img_size=224, patch_size=16, logger=None, 
                 mask_ratio=0.6):
        self._img_size = img_size
        self._patch_size = patch_size
        self._stride = int(self._img_size/self._patch_size)

        self._logger = logger

        self.mask_ratio = mask_ratio
        self.token_count = patch_size ** 2
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))
        
    def __call__(self, img):
        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1
        
        mask = mask.reshape((self._patch_size, self._patch_size))
        # mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)
        # print(f"[INFO]: mask is {mask}, size is {mask.shape}")

        # print(f"[INFO]: mask type is {type(mask)}")
        mask_view = mask.reshape(-1)

        # print(f"[TMP] - [CViT]: img type is {type(img)}, size is {img.size}")
        # 将 Image 对象转换为 NumPy 数组
        img_array = np.array(img)
        # print(f"[TMP] - [CViT]: img_array type is {type(img_array)}, size is {img_array.shape}")

        # 切块操作
        image_blocks = [
            Image.fromarray(img_array[i:i+self._stride, j:j+self._stride, :])
            for i in range(0, self._img_size, self._stride)
            for j in range(0, self._img_size, self._stride)
        ]

        # print(f"[TMP]: image_blocks number is {len(image_blocks)}")
        # print(f"[TMP]: mask_view number is {len(mask_view)}")
        
        # 存储需要Canny Detection的图片块
        blocks_to_canny = []
        blocks_no_canny = []
        for block, flag in zip(image_blocks, mask_view):
            if flag == 1:
                # print(f"[TMP] - [CViT]: block type is {type(block)}, size is {block.size}")
                gray_img = cv2.cvtColor(np.array(block), cv2.COLOR_RGB2GRAY)
                # print(f"[TMP] - [CViT]: gray_img type is {type(gray_img)}, size is {gray_img.shape}")
                edge_img = cv2.Canny(gray_img, 100, 200)
                edges_img = cv2.merge((edge_img, edge_img, edge_img))
                # print(f"[TMP] - [CViT]: gray_img type is {type(edges_img)}, size is {edges_img.shape}")
                # cv2.imshow('gray Image', gray_img)
                # cv2.imshow('edges Image', edges_img)
                # cv2.waitKey(0)
                edges_img = Image.fromarray(edges_img)
                # img_array = np.array(edges_img)
                # print(f"[TMP] - [CViT]: edges_img type is {type(edges_img)}, size is {edges_img.size}, mode is {edges_img.mode}")
                # print(f"[TMP] - [CViT]: edges_img max is {np.max(img_array)}, min is {np.min(img_array)}")
                # print(f"[TMP]: edges_img pixel_value is {type(edges_img.getpixel((2, 2))[0])}")
                blocks_to_canny.append(edges_img)
            else:
                # img_array = np.array(block)
                # # cv2.imshow('block Image', img_array)
                # # cv2.waitKey(0)
                # # print(f"[TMP]: block pixel_value is {type(block.getpixel((2, 2))[0])}")
                # print(f"[TMP] - [CViT]: block type is {type(block)}, size is {block.size}, mode is {block.mode}")
                # print(f"[TMP] - [CViT]: block max is {np.max(img_array)}, min is {np.min(img_array)}")
                blocks_no_canny.append(block)

        # print(f"[TMP] - [CViT]: blocks_to_canny length is {len(blocks_to_canny)}")
        # print(f"[TMP] - [CViT]: blocks_no_canny length is {len(blocks_no_canny)}")

        # 更新原始图片块列表
        canny_image_blocks = []
        for flag in mask_view:
            if flag == 1:
                canny_image_blocks.append(blocks_to_canny.pop(0))
            else:
                canny_image_blocks.append(blocks_no_canny.pop(0))
        # print(f"[TMP] - [CViT]: canny_image_blocks length is {len(canny_image_blocks)}")

        scramble_image = np.zeros((self._img_size, self._img_size, 3), dtype=np.uint8)
        index = 0
        for i in range(0, self._img_size, self._stride):
            for j in range(0, self._img_size, self._stride):
                scramble_image[i:i+self._stride, j:j+self._stride, :] = canny_image_blocks[index]
                index += 1

        # show_image = np.array(img)
        # cv2.imshow('Origina Image', show_image)
        # show_scramble_image = np.array(scramble_image)
        # cv2.imshow('Scramble Image', show_scramble_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return scramble_image, mask

class CannyVisionTransform():
    def __init__(self, config, logger):
        self.crop_img = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.RandomResizedCrop(config.DATA.IMG_SIZE, scale=(0.67, 1.), ratio=(3. / 4., 4. / 3.)),
            T.RandomHorizontalFlip(),
        ])

        self.norm_img = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=torch.tensor(IMAGENET_DEFAULT_MEAN),std=torch.tensor(IMAGENET_DEFAULT_STD)),
        ])
 
        self.canny_generator = CannyGenerator(
            img_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.VIT.PATCH_SIZE,
            logger=logger,
            mask_ratio=config.DATA.MASK_RATIO,
        )

    def __call__(self, img):
        # print(f"[INFO]: haha: img size is {img.size}")
        # print(f"[INFO]: haha: befor img element is {img.getpixel((20,20))}")
        # print(f"[INFO]: haha: IMAGENET_DEFAULT_MEAN is {IMAGENET_DEFAULT_MEAN}, IMAGENET_DEFAULT_STD is {IMAGENET_DEFAULT_STD}")
        # time1 = time.time()
        img = self.crop_img(img)
        # time2 = time.time()
        # print(f"[INFO]: transform_img cost time is {time2-time1} ms")

        # print(f"[INFO]: haha:after img size is {img.size()}, element is {img[:,20,20]}")
        scramble_img, mask = self.canny_generator(img)
        # time3 = time.time()
        # print(f"[INFO]: scramble_img cost time is {time3-time2} ms")
        img = self.norm_img(img)
        scramble_img = self.norm_img(scramble_img)

        return img, scramble_img, mask


def collate_fn(batch):
    if not isinstance(batch[0][0], tuple):
        print("[INFO]: batch is not instance")
        return default_collate(batch)
    else:
        batch_num = len(batch)
        # print(f"[INFO]: batch num is {batch_num}")
        ret = []
        for item_idx in range(len(batch[0][0])):
            if batch[0][0][item_idx] is None:
                ret.append(None)
            else:
                ret.append(default_collate([batch[i][0][item_idx] for i in range(batch_num)]))
        ret.append(default_collate([batch[i][1] for i in range(batch_num)]))
        return ret


def build_load_cvit(config, logger):
    transform = CannyVisionTransform(config, logger)
    logger.info(f'Pre-train data transform:\n{transform}')

    dataset = ImageFolder(config.DATA.DATA_PATH, transform)
    logger.info(f'Build dataset: train images = {len(dataset)}')
    
    sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True)

    dataloader = DataLoaderX(
        dataset=dataset, batch_size=config.DATA.BATCH_SIZE, 
        sampler=sampler, num_workers=config.DATA.NUM_WORKERS, 
        pin_memory=config.DATA.PIN_MEMORY, drop_last=True, collate_fn=collate_fn)
    
    return dataloader