#/bin/bash
export CUDA_VISIBLE_DEVICES=0 && nohup python train/train_frustum.py --gpu 0 --model frustum_pointnets_v2 --log_dir log_frustum --num_point 1024 --max_epoch 201 --batch_size 12 --decay_step 800000 --decay_rate 0.5 --pos_ratio 0.5 --use_gt_prop 0 > train_frustum.log 2>&1 &
