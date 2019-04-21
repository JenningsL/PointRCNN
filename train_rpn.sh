#!/usr/bin/env bash
#export CUDA_VISIBLE_DEVICES=0 && nohup python train/train_rpn.py --gpu 0 --decay_step 30000 --decay_rate 0.5 --num_point 16384 --model rpn --batch_size=4 --learning_rate 0.002 --log_dir log_rpn --restore_model_path ./log_rpn/bk/model.ckpt.008 > train.log 2>&1 &
export CUDA_VISIBLE_DEVICES=0 && nohup python train/train_rpn.py --gpu 0 --decay_step 30000 --decay_rate 0.5 --num_point 16384 --model rpn --batch_size=4 --learning_rate 0.002 --log_dir log_rpn > train.log 2>&1 &
