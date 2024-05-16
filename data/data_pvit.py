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
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data._utils.collate import default_collate
from torchvision.datasets import ImageFolder
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from .data_load_x import DataLoaderX


class PuzzleGenerator:
    def __init__(self, img_size=224, patch_size=16, logger=None, hold_first_patch = False,
                 patch_flip=True):
        self._img_size = img_size
        self._patch_size = patch_size
        self._stride = int(self._img_size/self._patch_size)

        self._logger = logger

        self._hold_first_patch = hold_first_patch
        print(f"[INFO]: hold first patch is {self._hold_first_patch}")

        self._patch_flip = patch_flip
        print(f"[INFO]: patch_flip is {self._patch_flip}")
        
    def __call__(self, img):
        # self._logger.info(f"img type is {type(img)}, size is {img.size()}")
        image_blocks = [
            img[:, i:i+self._stride, j:j+self._stride]
              for i in range(0, self._img_size, self._stride)
              for j in range(0, self._img_size, self._stride)]

        # 随机翻转图片块
        if self._hold_first_patch:
            first_block = image_blocks[0]
            image_blocks_new = []
            for i in range(1, len(image_blocks)):
                block = image_blocks[i]

                if self._patch_flip:
                    random_number = random.randint(0, 3)
                    if random_number == 1:
                        block = torch.flip(block, dims=(1, 2))
                    elif random_number == 2:
                        block = torch.rot90(block, k=1, dims=(1,2))
                    elif random_number == 3:
                        block = torch.rot90(block, k=-1, dims=(1,2))

                image_blocks_new.append(block)

                # self._logger.info(f"block type is {type(block)}, size is {block.size()}")
                # show_block_image = block.permute(1,2,0)
                # show_block_image = show_block_image.cpu().numpy()[:,:,::-1]
                # cv2.imshow('Block Image', show_block_image)
                # cv2.waitKey(0)

            random.shuffle(image_blocks_new)
            image_blocks_new.insert(0, first_block)

        else:
            image_blocks_new = []
            for block in image_blocks:
                random_number = random.randint(0, 3)

                if random_number == 1:
                    block = torch.flip(block, dims=(1, 2))
                elif random_number == 2:
                    block = torch.rot90(block, k=1, dims=(1,2))
                elif random_number == 3:
                    block = torch.rot90(block, k=-1, dims=(1,2))

                image_blocks_new.append(block)

                # self._logger.info(f"block type is {type(block)}, size is {block.size()}")
                # show_block_image = block.permute(1,2,0)
                # show_block_image = show_block_image.cpu().numpy()[:,:,::-1]
                # cv2.imshow('Block Image', show_block_image)
                # cv2.waitKey(0)

            random.shuffle(image_blocks_new)

        scramble_image = np.zeros((3, self._img_size, self._img_size))
        index = 0
        for i in range(0, self._img_size, self._stride):
            for j in range(0, self._img_size, self._stride):
                scramble_image[:, i:i+self._stride, j:j+self._stride] = image_blocks_new[index]
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

        return scramble_image


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
            patch_flip = config.PRETRAIN.PATCH_FLIP
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
        scramble_img = self.puzzle_generator(img)
        # time3 = time.time()
        # print(f"[INFO]: scramble_img cost time is {time3-time2} ms")

        return img, scramble_img


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