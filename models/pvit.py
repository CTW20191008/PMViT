# --------------------------------------------------------
# PVIT
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by ZhuHao
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from timm.models.layers import to_2tuple
import random
import cv2
import numpy as np

from .vision_transformer import VisionTransformer


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, logger, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = to_2tuple(img_size)
        self.patch_size = to_2tuple(patch_size)
        self.num_patches = (self.img_size[1] // self.patch_size[1]) * \
            (self.img_size[0] // self.patch_size[0])
        self.patch_shape = (
            self.img_size[0] // self.patch_size[0], self.img_size[1] // self.patch_size[1])

        self._stride = int(img_size/patch_size)

        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)  # This place stride must be patch_size!!!
        self._logger = logger
        self._image_size = img_size

    def forward(self, x, **kwargs):
        # self._logger.info("Enter PatchEmbed forward")
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        
        # self._logger.info(f"[BEFOR]: x type is {type(x)}, size is {x.size()}")
        x = self.proj(x).flatten(2).transpose(1, 2)
        # self._logger.info(f"[AFTER]: x type is {type(x)}, size is {x.size()}")

        return x


class VisionTransformerForPuzzle(VisionTransformer):
    def __init__(self, logger, **kwargs):
        super().__init__(**kwargs)

        assert self.num_classes == 0

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self._trunc_normal_(self.mask_token, std=.02)

        self._logger = logger
        self.patch_embed = PatchEmbed(
            logger=self._logger, img_size=self._img_size, patch_size=self.patch_size,
            in_chans=self.in_chans, embed_dim=self.embed_dim)

    def _trunc_normal_(self, tensor, mean=0., std=1.):
        trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)

    def forward(self, x):
        x = self.patch_embed(x)
        B, L, _ = x.shape

        # stole cls_tokens impl from Phil Wang, thanks
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        # rel_pos_bias = None
        for blk in self.blocks:
            x = blk(x, rel_pos_bias=rel_pos_bias)
        x = self.norm(x)

        x = x[:, 1:]
        B, L, C = x.shape
        H = W = int(L ** 0.5)
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        return x


class PVIT(nn.Module):
    def __init__(self, encoder, encoder_stride, logger, mode, loss):
        super().__init__()
        self.encoder = encoder
        self.encoder_stride = encoder_stride

        self.decoder = nn.Sequential(
            nn.Conv2d(
                in_channels=self.encoder.num_features,
                out_channels=self.encoder_stride ** 2 * 3, kernel_size=1),
            nn.PixelShuffle(self.encoder_stride),
        )

        self.in_chans = self.encoder.in_chans
        self.patch_size = self.encoder.patch_size

        self._logger = logger
        self._mode = mode
        self._loss = loss

    def forward(self, x, scramble_img):
        if self._mode == 'pretrain':
            # print(f"[INFO]: befor scramble_img element is {scramble_img[0][0][10][10]}, {scramble_img[0][0][20][20]}")
            z = self.encoder(scramble_img)
            # print(f"[INFO]: after scramble_img element is {scramble_img[0][0][10][10]}, {scramble_img[0][0][20][20]}")
            # print(f"[INFO]: scramble_img size is {scramble_img.size()}, z sizie is {z.size()}") # [1, 3, 224, 224], [1, 768, 14, 14].
            x_rec = self.decoder(z)

            # mask = mask.repeat_interleave(self.patch_size, 1).repeat_interleave(
            #     self.patch_size, 2).unsqueeze(1).contiguous()

            if self._loss == 'L1':
                loss_recon = F.l1_loss(x, x_rec, reduction='none')
                loss_value = loss_recon.sum() / loss_recon.numel() / self.in_chans
                # self._logger.info(f"loss_recon type is {type(loss_recon)}, size is {loss_recon.size()}")
                # loss = (loss_recon * mask).sum() / (mask.sum() + 1e-5) / self.in_chans
            elif self._loss == 'L2':
                l2_loss = nn.MSELoss()
                loss_value = l2_loss(x, x_rec)

            return loss_value
        elif self._mode == 'inference':
            z = self.encoder(scramble_img)
            x_rec = self.decoder(z)
            return x_rec
    
    @torch.jit.ignore
    def no_weight_decay(self):
        if hasattr(self.encoder, 'no_weight_decay'):
            return {'encoder.' + i for i in self.encoder.no_weight_decay()}
        return {}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        if hasattr(self.encoder, 'no_weight_decay_keywords'):
            return {'encoder.' + i for i in self.encoder.no_weight_decay_keywords()}
        return {}


def build_pvit(config, logger):
    encoder = VisionTransformerForPuzzle(
        logger,
        img_size=config.DATA.IMG_SIZE,
        patch_size=config.MODEL.VIT.PATCH_SIZE,
        in_chans=config.MODEL.VIT.IN_CHANS,
        num_classes=0,
        embed_dim=config.MODEL.VIT.EMBED_DIM,
        depth=config.MODEL.VIT.DEPTH,
        num_heads=config.MODEL.VIT.NUM_HEADS,
        mlp_ratio=config.MODEL.VIT.MLP_RATIO,
        qkv_bias=config.MODEL.VIT.QKV_BIAS,
        drop_rate=config.MODEL.DROP_RATE,
        drop_path_rate=config.MODEL.DROP_PATH_RATE,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_values=config.MODEL.VIT.INIT_VALUES,
        use_abs_pos_emb=config.MODEL.VIT.USE_APE,
        use_rel_pos_bias=config.MODEL.VIT.USE_RPB,
        use_shared_rel_pos_bias=config.MODEL.VIT.USE_SHARED_RPB,
        use_mean_pooling=config.MODEL.VIT.USE_MEAN_POOLING)
    encoder_stride = 16
    # print(f'[TMP]: config model is {config.MODEL}')
    model = PVIT(encoder=encoder, encoder_stride=encoder_stride, logger=logger, mode=config.MODEL.MODE,
                 loss=config.MODEL.LOSS)

    return model
