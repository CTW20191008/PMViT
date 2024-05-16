# --------------------------------------------------------
# SimMIM
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# Modified by Zhenda Xie
# --------------------------------------------------------

import os
import time
import argparse
import datetime
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import cv2
import torchvision.transforms as transforms

from matplotlib import pyplot as plt
from config import get_config
from models import build_model
from data import build_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import load_checkpoint, auto_resume_helper
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from skimage.metrics import structural_similarity as compare_ssim


def parse_option():
    parser = argparse.ArgumentParser('ViT pretrain model inderence script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')

    # distributed training
    parser.add_argument("--local-rank", type=int, default=0, help='local rank for DistributedDataParallel')

    args = parser.parse_args()

    config = get_config(args)

    config.DATA.BATCH_SIZE = 1
    config.MODEL.MODE = 'inference'

    config.freeze()

    return args, config


def main(config):
    time1 = time.time()
    data_loader_train = build_loader(config, logger, is_pretrain=True)
    time2 = time.time()
    print(f"build_loader cost time is {time2-time1} ms")

    print(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config, logger, is_pretrain=True)
    model.cuda()
    print(str(model))
    time3 = time.time()
    print(f"build_loader cost time is {time3-time2} ms")

    optimizer = build_optimizer(config, model, logger, is_pretrain=True)
    time4 = time.time()
    print(f"build_loader cost time is {time4-time3} ms")

    # model = torch.nn.parallel.DataParallel(model, device_ids=[config.LOCAL_RANK])
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False, find_unused_parameters=True)
    model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"number of params: {n_parameters}")
    if hasattr(model_without_ddp, 'flops'):
        flops = model_without_ddp.flops()
        print(f"number of GFLOPs: {flops / 1e9}")

    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))
    time5 = time.time()
    print(f"build_scheduler cost time is {time5-time4} ms")

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT, logger)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            print(f'auto resuming from {resume_file}')
        else:
            print(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    if config.MODEL.RESUME:
        load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, logger)

    print("Start inference")
    start_time = time.time()

    inference(model, data_loader_train)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('inference time {}'.format(total_time_str))


def inference(model, data_loader):
    time1 = time.time()
    model.train()
    time2 = time.time()
    print(f"model train cost time is {time2-time1} ms")

    time3 = time.time()
    print(f"optimizer zero_grad cost time is {time3-time2} ms")

    # transform = transforms.ToPILImage()
    for idx, (img, mask, _) in enumerate(data_loader):
        print("[INFO]: Enter inference")
        time1 = time.time()
        img = img.cuda(non_blocking=True)
        mask = mask.cuda(non_blocking=True)
        # time2 = time.time()
        # print(f"get images cost time is {time2-time1} ms")

        generate_img = model(img, mask)
        # print(f"[TMP]: scramble_img size is {scramble_img.size()}")

        # img = transform(img.squeeze(0))
        # scramble_img = transform(scramble_img.squeeze(0))
        # puzzles_img = transform(puzzles_img.squeeze(0))

        img_1 = img.cpu().squeeze(0).numpy().transpose((1, 2, 0))
        generate_img_1 = generate_img.cpu().squeeze(0).detach().numpy().transpose((1, 2, 0))
        # print(f"[TMP]: img1 type is {type(img_1)}, size is {img_1.shape}")
        # print(f"[TMP]: puzzles_img_1 type is {type(puzzles_img_1)}, size is {puzzles_img_1.shape}")
        ssim = compare_ssim(img_1, generate_img_1, channel_axis=2, data_range=img_1.max()-img_1.min())

        # 定义反归一化的转换
        unnormalize = transforms.Normalize(mean=[-m/s for m, s in zip(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)],
                                        std=[1/s for s in IMAGENET_DEFAULT_STD])
        img = unnormalize(img.cpu().squeeze(0)).numpy().transpose((1, 2, 0))
        generate_img = unnormalize(generate_img.cpu().squeeze(0)).detach().numpy().transpose((1, 2, 0))

        ssim_unnorm = compare_ssim(img, generate_img, channel_axis=2, data_range=img.max() - img.min())
        print(f"[INFO]: ssim is {ssim}, ssim_unnorm is {ssim_unnorm}")

        plt.subplot(1,2,1)
        plt.imshow(img)
        plt.title('original image')
        plt.subplot(1,2,2)
        plt.imshow(generate_img)
        plt.title('generate image')
        plt.show()

        cv2.waitKey()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    _, config = parse_option()

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    torch.cuda.set_device(config.LOCAL_RANK)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()

    # seed = config.SEED + 0
    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")

    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        print(f"Full config saved to {path}")

    # print config
    print(config.dump())

    main(config)
