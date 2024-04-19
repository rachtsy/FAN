#!/bin/bash
# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/FAN/blob/main/LICENSE
CUDA_VISIBLE_DEVICES=0 /root/FAN/validate_ood.py \
        /root/data/imagenet-a \
	--model fan_tiny_12_p16_224 \
        --img-size 224 \
        -b 128 \
        -j 32 \
	--no-test-pool \
        --imagenet_a \
	--results-file /root/checkpoints/ \
	--checkpoint /root/checkpoints/fan_tiny_robust/train/20240324-173056-fan_tiny_12_p16_224-224/model_best.pth.tar \
        --robust
