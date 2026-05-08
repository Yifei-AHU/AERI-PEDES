#!/bin/bash
DATASET_NAME="AERI-PEDES"    # AERI-PEDES  AGDataAttr

CUDA_VISIBLE_DEVICES=5 \
python finetune.py \
--name finetune \
--img_aug \
--batch_size 64 \
--MLM \
--dataset_name $DATASET_NAME \
--loss_names 'cda' \
--lr 5e-6 \
--lr2 5e-5 \
--num_epoch 60 \
--root_dir /data1/Datasets/ReID/ \
--finetune 'pretrain/HAMbest0.pth'