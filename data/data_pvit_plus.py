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
from torch.utils.data import RandomSampler
from torch.utils.data._utils.collate import default_collate
from torchvision.datasets import ImageFolder
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from .data_load_x import DataLoaderX
import torch.nn as nn


class PuzzleGenerator:
    def __init__(self, img_size=224, patch_size=16, logger=None, hold_first_patch = False,
                 patch_flip=True, unorder_ratio=0.6,
                 random_unorder=False, min_unorder_ratio=0.3, max_unorder_ratio=0.6,
                 do_mask=False, mask_pixel=2,
                 do_shuffle=True):
        self._img_size = img_size
        self._patch_size = patch_size
        self._patch_num = int(self._img_size/self._patch_size)
        self.token_count = self._patch_num ** 2

        self._logger = logger

        self._hold_first_patch = hold_first_patch
        # print(f"[INFO]: hold first patch is {self._hold_first_patch}")

        self._patch_flip = patch_flip
        # print(f"[INFO]: patch_flip is {self._patch_flip}")

        self.random_unorder = random_unorder
        self.min_unorder_ratio = min_unorder_ratio
        self.max_unorder_ratio = max_unorder_ratio
        
        self.unorder_ratio = unorder_ratio
        self.unorder_count = int(np.ceil(self.token_count * self.unorder_ratio))

        self.do_mask = do_mask
        self.begin_mask_pixel = mask_pixel
        self.end_mask_pixel = self._patch_size - self.begin_mask_pixel

        self.do_shuffle = do_shuffle

    def __call__(self, scramble_image):
        if self.random_unorder:
            unorder_ratio = random.uniform(self.min_unorder_ratio, self.max_unorder_ratio)
            # print(f"[TMP]: unorder ratio is {unorder_ratio}")
            unorder_count = int(np.ceil(self.token_count * unorder_ratio))
            unorder_idx = np.random.permutation(self.token_count)[:unorder_count]
        else:
            unorder_idx = np.random.permutation(self.token_count)[:self.unorder_count]
        unorder = np.zeros(self.token_count, dtype=int)
        unorder[unorder_idx] = 1
        
        unorder = unorder.reshape((self._patch_num, self._patch_num))
        # unorder = unorder.repeat(self.scale, axis=0).repeat(self.scale, axis=1)
        # print(f"[INFO]: unorder is {unorder}, size is {unorder.shape}")

        # print(f"[INFO]: unorder type is {type(unorder)}")
        unorder_view = unorder.reshape(-1)

        # self._logger.info(f"img type is {type(img)}, size is {img.size()}")
        image_blocks = [
            scramble_image[:, i:i+self._patch_size, j:j+self._patch_size]
              for i in range(0, self._img_size, self._patch_size)
              for j in range(0, self._img_size, self._patch_size)]
        # print(f"[TMP]: image_blocks number is {len(image_blocks)}")
        # print(f"[TMP]: unorder_view number is {len(unorder_view)}")
        
        blocks_to_shuffle, blocks_no_shuffle = [], []
        # 存储需要乱序的图片块
        if self._patch_flip:
            for block, flag in zip(image_blocks, unorder_view):
                if flag == 1:
                    random_number = random.randint(0, 3)
                    if random_number == 1:
                        block = torch.flip(block, dims=(1, 2))
                    elif random_number == 2:
                        block = torch.rot90(block, k=1, dims=(1,2))
                    elif random_number == 3:
                        block = torch.rot90(block, k=-1, dims=(1,2))

                    if self.do_mask:
                        block[:, self.begin_mask_pixel:self.end_mask_pixel, self.begin_mask_pixel:self.end_mask_pixel] = 0

                    blocks_to_shuffle.append(block)
                else:
                    blocks_no_shuffle.append(block)
        else:
            for block, flag in zip(image_blocks, unorder_view):
                if flag == 1:
                    if self.do_mask:
                        block[:, self.begin_mask_pixel:self.end_mask_pixel, self.begin_mask_pixel:self.end_mask_pixel] = 0
                    blocks_to_shuffle.append(block)
                else:
                    blocks_no_shuffle.append(block)
            # blocks_to_shuffle = [block for block, flag in zip(image_blocks, unorder_view) if flag == 1]
            # blocks_no_shuffle = [block for block, flag in zip(image_blocks, unorder_view) if flag == 0]

        # 随机乱序需要乱序的图片块
        if self.do_shuffle:
            random.shuffle(blocks_to_shuffle)

        # 根据乱序结果更新原始图片块列表
        shuffled_image_blocks = []
        for flag in unorder_view:
            if flag == 1:
                shuffled_image_blocks.append(blocks_to_shuffle.pop(0))
            else:
                shuffled_image_blocks.append(blocks_no_shuffle.pop(0))

        # scramble_image = np.zeros((3, self._img_size, self._img_size))
        index = 0
        for i in range(0, self._img_size, self._patch_size):
            for j in range(0, self._img_size, self._patch_size):
                scramble_image[:, i:i+self._patch_size, j:j+self._patch_size] = shuffled_image_blocks[index]
                index += 1
        scramble_image = torch.Tensor(scramble_image)

        # show_image = img.permute(1,2,0)
        # show_image = show_image.cpu().numpy()[:,:,::-1]
        # cv2.imshow('Origina Image', show_image)

        # show_scramble_image = scramble_image.permute(1,2,0)
        # show_scramble_image = show_scramble_image.cpu().numpy()[:,:,::-1]
        # cv2.imshow('Scramble Image', show_scramble_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return scramble_image, unorder

class PuzzleVisionTransform():
    def __init__(self, config, logger):
        self.transform_img = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.RandomResizedCrop(config.DATA.IMG_SIZE, scale=(0.67, 1.), ratio=(3. / 4., 4. / 3.)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=torch.tensor(IMAGENET_DEFAULT_MEAN),std=torch.tensor(IMAGENET_DEFAULT_STD)),
        ])
 
        self.puzzle_generator = PuzzleGenerator(
            img_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.VIT.PATCH_SIZE,
            logger=logger,
            hold_first_patch=config.PRETRAIN.HOLD_FIRST_PATCH,
            patch_flip = config.PRETRAIN.PATCH_FLIP,
            unorder_ratio=config.DATA.UNORDER_RATIO,
            random_unorder=config.DATA.RANDOM_UNORDER,
            min_unorder_ratio=config.DATA.MIN_UNORDER_RATIO,
            max_unorder_ratio=config.DATA.MAX_UNORDER_RATIO,
            do_mask=config.DATA.DO_MASK,
            mask_pixel=config.DATA.MASK_PIXEL,
            do_shuffle=config.DATA.DO_SHUFFLE
        )

    def __call__(self, img):
        # print(f"[INFO]: haha: img size is {img.size}")
        # print(f"[INFO]: haha: befor img element is {img.getpixel((20,20))}")
        # print(f"[INFO]: haha: IMAGENET_DEFAULT_MEAN is {IMAGENET_DEFAULT_MEAN}, IMAGENET_DEFAULT_STD is {IMAGENET_DEFAULT_STD}")
        # time1 = time.time()
        img = self.transform_img(img)
        # time2 = time.time()
        # print(f"[INFO]: transform_img cost time is {time2-time1} ms")

        # print(f"[INFO]: haha:after img size is {img.size()}, element is {img[:,20,20]}")
        scramble_img = torch.from_numpy(np.copy(img.numpy()))
        scramble_img, unorder = self.puzzle_generator(scramble_img)
        # time3 = time.time()
        # print(f"[INFO]: scramble_img cost time is {time3-time2} ms")

        return img, scramble_img, unorder


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
    

def build_load_pvit_single(config, logger):
    transform = PuzzleVisionTransform(config, logger)
    logger.info(f'Pre-train data transform:\n{transform}')

    dataset = ImageFolder(config.DATA.DATA_PATH, transform)
    logger.info(f'Build dataset: train images = {len(dataset)}')
    
    # sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True)
    sampler = RandomSampler(dataset, replacement=False)

    dataloader = DataLoaderX(
        dataset=dataset, batch_size=config.DATA.BATCH_SIZE, 
        sampler=sampler, num_workers=config.DATA.NUM_WORKERS, 
        pin_memory=config.DATA.PIN_MEMORY, drop_last=True, collate_fn=collate_fn)
    
    return dataloader


def build_load_pvit(config, logger):
    transform = PuzzleVisionTransform(config, logger)
    logger.info(f'Pre-train data transform:\n{transform}')

    dataset = ImageFolder(config.DATA.DATA_PATH, transform)
    logger.info(f'Build dataset: train images = {len(dataset)}')
    
    sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True)
    # sampler = RandomSampler(dataset, replacement=False)

    dataloader = DataLoaderX(
        dataset=dataset, batch_size=config.DATA.BATCH_SIZE, 
        sampler=sampler, num_workers=config.DATA.NUM_WORKERS, 
        pin_memory=config.DATA.PIN_MEMORY, drop_last=True, collate_fn=collate_fn)
    
    return dataloader