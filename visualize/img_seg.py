import cv2
import sys
import numpy as np

colormap = np.zeros((256, 3), dtype=np.uint8)
colormap[0] = [0, 0, 0]
colormap[1] = [244, 35, 232]
colormap[2] = [20, 230, 230]
colormap[3] = [20, 230, 20]

def add_mask_to_img(img, mask):
    mask = colormap[mask]
    return cv2.addWeighted(img, 1, mask, 0.5, 0)

def display_seg_mask(img, mask):
    img = add_mask_to_img(img, mask)
    cv2.imshow('img', img)
    cv2.waitKey(0)

if __name__ == '__main__':
    img_path = sys.argv[1]
    mask_path = sys.argv[2]
    img = cv2.imread(img_path)
    pred = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    # pred = cv2.imread(mask_path)
    display_seg_mask(img, pred, colormap)
