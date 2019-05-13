import cv2
import os
import sys
import numpy as np
from PIL import Image

def assemble_one_frame(img_2d, img_3d, img_seg):
    target = Image.new('RGB', (1920, 720))

    width, height = img_3d.size
    left = (width - 720)/2
    top = (height - 720)/2 + 70
    right = (width + 720)/2
    bottom = (height + 720)/2 + 70
    img_3d = img_3d.crop((left, top, right, bottom))

    img_3d = img_3d.resize((720, 720))
    img_2d = img_2d.resize((1200, 360))
    img_seg = img_seg.resize((1200, 360))
    target.paste(img_3d, (0, 0, 720, 720))
    target.paste(img_2d, (720, 0, 1920, 360))
    target.paste(img_seg, (720, 360, 1920, 720))
    return target

def images_to_video(images, video_name, fps=20):
    width, height = images[0].size

    video = cv2.VideoWriter(video_name, 0, fps, (width,height))

    for image in images:
        image = np.array(image)
        video.write(image[:, :, ::-1])

    cv2.destroyAllWindows()
    video.release()

if __name__ == '__main__':
    vis_folder = sys.argv[1]
    video_name = sys.argv[2]

    folder_2d = os.path.join(vis_folder, 'result_2d_image')
    folder_3d = os.path.join(vis_folder, 'result_3d_image')
    folder_seg = os.path.join(vis_folder, 'result_seg_image')

    images_2d = [Image.open(os.path.join(folder_2d, img)) for img in sorted(os.listdir(folder_2d)) if img.endswith(".png")]
    images_3d = [Image.open(os.path.join(folder_3d, img)) for img in sorted(os.listdir(folder_3d)) if img.endswith(".png")]
    images_seg = [Image.open(os.path.join(folder_seg, img)) for img in sorted(os.listdir(folder_seg)) if img.endswith(".png")]

    frames = map(lambda f:assemble_one_frame(f[0], f[1], f[2]), zip(images_2d, images_3d, images_seg))

    images_to_video(frames, video_name)
