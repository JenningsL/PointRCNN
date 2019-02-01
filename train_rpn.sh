#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=2,3 && nohup python train/train.py --gpu 0 --decay_step 30000 --decay_rate 0.8 --model rpn --batch_size=2 --learning_rate 0.0001 > train.log 2>&1 &
