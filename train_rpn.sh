#!/usr/bin/env bash
#export CUDA_VISIBLE_DEVICES=2,3 && nohup python train/train.py --gpu 0 --decay_step 30000 --decay_rate 0.8 --model rpn --batch_size=4 --learning_rate 0.002 > train.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2,3 && nohup python train/train.py --gpu 0 --decay_step 30000 --decay_rate 0.8 --model rpn --batch_size=4 --learning_rate 0.002 --restore_model_path log/bk/model.ckpt.038 --log_dir log_rpn > train.log 2>&1 &
