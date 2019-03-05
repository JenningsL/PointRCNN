#!/bin/bash
export CUDA_VISIBLE_DEVICES=3 && nohup python train/test_rcnn.py --gpu 0 --num_point 512 --batch_size 8 --model_path log_rcnn/model.ckpt.059 --split val --output val_results > test_rcnn.log 2>&1
./evaluate_object_3d_offline /data/ssd/public/jlliu/Kitti/object/training/label_2 ./val_results >> test_rcnn.log 2>&1
