#!/bin/bash
export CUDA_VISIBLE_DEVICES=0 && nohup python train/train_rcnn.py --gpu 0 --decay_step 800000 --decay_rate 0.5 --batch_size=32 --num_point 512 --learning_rate 0.01 --max_epoch  200 > train_rcnn.log 2>&1 &
