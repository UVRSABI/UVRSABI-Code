import re
import cv2
import argparse
import os
import matplotlib.pyplot as plt
import time


def SaveResult(image_path, roof_mask_path, save_path):

    image = cv2.imread(os.path.join(opt.image_folder, image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    roof_mask = cv2.imread(os.path.join(opt.roof_mask_folder, roof_mask_path))
    roof_mask = cv2.cvtColor(roof_mask, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(image)
    ax[0].set_title("Image:"+str(image_path),fontdict={'fontsize': 10})
    ax[0].axis('off')
    ax[1].imshow(roof_mask)
    ax[1].set_title("Roof Mask" ,fontdict={'fontsize': 10})
    ax[1].axis('off')
    plt.savefig(os.path.join(save_path, image_path.replace('.png', '_roofresult.png')), bbox_inches='tight', dpi = 300)
    time.sleep(1)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generate Intermediate Results')
    parser.add_argument('-i', '--image_folder', help='Image folder', required=True)
    parser.add_argument('-r', '--roof_mask_folder', help='Roof Mask folder', required=True)
    parser.add_argument('-s', '--save_folder', help='Save Folder', required=True)
    opt = parser.parse_args()

    images = os.listdir(opt.image_folder)
    print(images)
    roof_masks = os.listdir(opt.roof_mask_folder)

    images.sort(key=lambda f:int(re.sub('\D', '', f)))
    roof_masks.sort(key=lambda f:int(re.sub('\D', '', f)))

    for i in range(0,len(images),10):
        SaveResult(images[i], roof_masks[i], opt.save_folder)