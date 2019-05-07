#!/bin/bash

IMG_MODEL="frozen_inference_graph.pb"
RPN_MODEL="log_rpn/model.ckpt"
RCNN_MODEL="log_frustum/model.ckpt"

export CUDA_VISIBLE_DEVICES=0 && python test/detect.py --gpu 0 --rpn_model ${RPN_MODEL} --img_seg_model ${IMG_MODEL} --rcnn_model ${RCNN_MODEL} --split test --kitti_path /data/ssd/public/jlliu/Kitti/2011_09_29 --output test_result_2011_09_29 --save_img_seg > detect.log 2>&1
# visualization
python visualize/viz_detection.py --kitti_path /data/ssd/public/jlliu/Kitti/2011_09_29 --kitti_type video --detection_path test_result_2011_09_29/data --output_dir vis_2011_09_29
python visualize/images_to_video.py vis_2011_09_29/result_2d_image 2011_09_29.avi
