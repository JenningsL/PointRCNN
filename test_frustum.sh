#/bin/bash
# test with val set
export CUDA_VISIBLE_DEVICES=0 && python test/test_frustum.py --gpu 0 --model frustum_pointnets_v2 --model_path log_frustum/model.ckpt --output val_results --num_point 1024 --batch_size 12 --dump_result --kitti_path /data/ssd/public/jlliu/Kitti/object --split val
./evaluate_object_3d_offline /data/ssd/public/jlliu/Kitti/object/training/label_2/ val_results
