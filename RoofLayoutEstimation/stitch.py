import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
import argparse
import os

def extract_features(img):
    
    #Using Clahe for better contrast, thus increasing the number of features detected
    #clahe = cv2.createCLAHE(clipLimit=25.0)
    img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #img=clahe.apply(img)
    
    #Using FAST
    fast= cv2.FastFeatureDetector_create(threshold = 25, nonmaxSuppression = True)
    kp = fast.detect(img)
    kp = np.array([kp[idx].pt for idx in range(len(kp))], dtype = np.float32)


def track_features(image_ref, image_cur,ref):
    #Initializing LK parameters
    lk_params = dict(winSize=(21, 21), criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    kp2, st, err = cv2.calcOpticalFlowPyrLK(image_ref, image_cur, ref, None, **lk_params)
    
    kp1, st, err = cv2.calcOpticalFlowPyrLK(image_cur, image_ref, kp2, None, **lk_params)
#     distance=abs(ref-kp1).max(-1)

    return kp1, kp2

def stitch(img1, img2):

    img_gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    img_gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    descriptor = cv2.ORB_create()
    kps1, features1 = descriptor.detectAndCompute(img_gray1, None)
    kps2, features2 = descriptor.detectAndCompute(img_gray2, None)

    kps1_ref = np.array([kps1[idx].pt for idx in range(len(kps1))], dtype = np.float32)
    kps1_of, kps2_of = track_features(img1, img2, kps1_ref)
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = False)

    best_match = bf.knnMatch(features1, features2, 2)
    good_match = []
    for m, n in best_match:
        if m.distance < n.distance * 0.7:
            good_match.append(m)
    img3 = cv2.drawMatches(img1, kps1, img2, kps2, good_match, None, flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    kps1 = np.float32([kp.pt for kp in kps1])
    kps2 = np.float32([kp.pt for kp in kps2])
    H =np.eye(3)
    A = np.eye(3)
    if len(good_match) > 4:
        pts1 = np.float32([kps1[m.queryIdx] for m in good_match])
        pts2 = np.float32([kps2[m.trainIdx] for m in good_match])
        A, inliner = cv2.estimateAffine2D(pts1, pts2)
    else:
        print("oops homo esti failed")
    A_,inl = cv2.estimateAffine2D(kps1_of, kps2_of)
    # print("A_", A_)
    # print("A = ", A)
    return A_

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Stitching Images')
    parser.add_argument('-i', '--image_folder', help='Image folder', required=True)
    parser.add_argument('-r', '--roof_mask_folder', help='Roof Mask folder', required=True)
    parser.add_argument('-o', '--object_mask_folder', help='Object Mask folder', required=True)
    parser.add_argument('-s', '--save_folder', help='Save Folder',required=True)
    opt = parser.parse_args()
    count = 0
    H = np.eye(3)
    H_old = H
    num_img = len(os.listdir('images'))
    for i in range(0, num_img):
        j = i+1
        if(count == 0):
            prev_frame = cv2.imread(os.path.join(opt.image_folder, str(j) + '.png'))
            prev_mask = cv2.imread(os.path.join(opt.object_mask_folder, str(j) + '.png'))
            prev_roof_mask = cv2.imread(os.path.join(opt.roof_mask_folder, str(j) + '.png'))
            prev_frame = cv2.resize(prev_frame, (0,0), fx = 0.2, fy = 0.2)
            prev_mask = cv2.resize(prev_mask, (0, 0), fx = 0.2, fy = 0.2)
            prev_roof_mask = cv2.resize(prev_roof_mask, (384, 216))

            count += 1
            warp_naive = prev_frame 
            warp_mask = prev_mask
            warp_roof_mask = prev_roof_mask
            continue

        cur_frame = cv2.imread(os.path.join(opt.image_folder, str(j) + '.png'))
        cur_mask = cv2.imread(os.path.join(opt.object_mask_folder, str(j) + '.png'))
        cur_roof_mask = cv2.imread(os.path.join(opt.roof_mask_folder, str(j) + '.png'))
        cur_frame = cv2.resize(cur_frame, (0,0), fx = 0.2, fy = 0.2)
        cur_mask = cv2.resize(cur_mask, (0, 0), fx = 0.2, fy = 0.2)
        cur_roof_mask = cv2.resize(cur_roof_mask, (384, 216))
        A = stitch(prev_frame, cur_frame)
     
        warp_naive = cv2.warpAffine(warp_naive, A, (prev_frame.shape[1] + cur_frame.shape[1], 3000), flags = cv2.INTER_NEAREST)
        warp_mask = cv2.warpAffine(warp_mask, A, (prev_mask.shape[1] + prev_frame.shape[1], 3000), flags = cv2.INTER_NEAREST)
        warp_roof_mask = cv2.warpAffine(warp_roof_mask, A, (prev_mask.shape[1] + prev_frame.shape[1], 3000), flags = cv2.INTER_NEAREST)
        warp_naive[:cur_frame.shape[0], :cur_frame.shape[1]] = cur_frame
        warp_mask[:cur_mask.shape[0], :cur_mask.shape[1]] = np.maximum(warp_mask[:cur_mask.shape[0], :cur_mask.shape[1]],  cur_mask)
        warp_roof_mask[:cur_roof_mask.shape[0], :cur_roof_mask.shape[1]] = np.maximum(warp_roof_mask[:cur_roof_mask.shape[0], :cur_roof_mask.shape[1]],  cur_roof_mask)
        prev_frame = cur_frame
        prev_mask = cur_mask
        prev_roof_mask = cur_roof_mask


    object_stitched_mask = warp_mask
    roof_stitched_mask = warp_roof_mask
    stitched_image = warp_naive
    cv2.imwrite(os.path.join(opt.save_folder,"stitched_image.png"), stitched_image)
    cv2.imwrite(os.path.join(opt.save_folder,"stitched_object_mask.png"), object_stitched_mask)
    cv2.imwrite(os.path.join(opt.save_folder,"stitched_roof_mask.png"), roof_stitched_mask)
    stitched_image = cv2.cvtColor(stitched_image, cv2.COLOR_BGR2RGB)
    roof_stitched_mask = cv2.cvtColor(roof_stitched_mask, cv2.COLOR_BGR2RGB)
    object_stitched_mask = cv2.cvtColor(object_stitched_mask, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(1, 3)
    
    ax[0].imshow(stitched_image)
    ax[0].set_title("Stitched Image",fontdict={'fontsize': 10})
    ax[0].axis('off')
    ax[1].imshow(roof_stitched_mask)
    ax[1].set_title("Roof Mask" ,fontdict={'fontsize': 10})
    ax[1].axis('off')
    ax[2].imshow(object_stitched_mask)
    ax[2].set_title("Object Mask" ,fontdict={'fontsize': 10})
    ax[2].axis('off')
    plt.savefig(os.path.join(opt.save_folder,"stitched.png"), bbox_inches='tight', dpi = 300)
    time.sleep(1)