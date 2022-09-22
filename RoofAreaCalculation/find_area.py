import numpy as np 
import cv2
import matplotlib.pyplot as plt
import shapely
from shapely.geometry import Polygon
import argparse
import pandas
import os


def calcNetArea(contours, hierarchy, depth, focalLength):
    
    sorted_contours = sorted(contours, key=cv2.contourArea)
    maxContour = sorted_contours[len(contours) - 1]
    maxCont = np.squeeze(maxContour, axis=1)
    hierarchy = np.squeeze(hierarchy, axis=0)
    roi_contours = []
    roi_contours.append(maxContour)

    maxCont = maxCont*(depth/focalLength)

    outer_polygon_area = Polygon(maxCont).area
    child_polygon_totalarea = 0.0
    maxContourIndex = -1

    for i in range(len(contours)):
        if contours[i] is maxContour:
            maxContourIndex = i
            break
    index = maxContourIndex

    if hierarchy[index][2] != -1: ## check if child is there or not
        for i in range(len(contours)):
            child_contour = contours[i]
            if  child_contour is maxContour:
                continue
            elif hierarchy[i][3] == maxContourIndex: ## Is a child
                roi_contours.append(contours[i])
                c = contours[i]
                c = np.squeeze(c, axis=1)
                c = c*(depth/focalLength)
                child_polygon_area = Polygon(c).area
                child_polygon_totalarea += child_polygon_area
                
    net_area = outer_polygon_area - child_polygon_totalarea

    return net_area, roi_contours
    
def get_area_from_mask(i, depth, focalLength, thresh = 200):
    '''
    image: input mask of building obtained by semantic segmentation
    depth: depth to the building
    focalLength: focalLength of camera in pixels
    thresh: binary threshold
    '''
    image = cv2.imread(os.path.join(args.roofmasks,i))

    imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # print('Binary threhold set to:', thresh)
    ret, thresh = cv2.threshold(imgray.astype(np.uint8), thresh, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    ## Draw max area contour
    c = max(contours, key = cv2.contourArea)
    cv2.drawContours(image, [c], 0, (0,0,255), 3)
    c = np.squeeze(c, axis=1)
    # print('Final Outer Polygon Area:', Polygon(c).area*(depth/focalLength)**2)    
    # plt.title('Contour with max area')
    # plt.imshow(image)
    # plt.show()

    net_area, roi_contours = calcNetArea(contours, hierarchy, depth, focalLength)


    ## Draw ROI contours
    img_roicontours = image.copy()
    cv2.drawContours(img_roicontours, roi_contours, -1, (0,0,255), 20)

    fig, ax = plt.subplots(1,2, figsize=(10,10))
    display_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ax[0].imshow(display_image)
    ax[0].set_title('Roof Mask')
    ax[0].axis('off')
    ax[1].set_title('Building contour')
    ax[1].imshow(img_roicontours)
    ax[1].axis('off')
    plt.savefig(os.path.join(args.save_dir_intermediate, i), bbox_inches='tight')
    # 
    # plt.show()
    return net_area

def read_logs(log_file):

    df = pandas.read_csv(log_file)
    number_of_images = df.shape[0]
    roof_masks = []
    relative_altitude = []
    focalLengths = []
    for i in range(number_of_images):
        roof_masks.append(df['image'][i])
        relative_altitude.append(df['relative_altitude'][i])
        focalLengths.append(df['focal_length'][i])
    
    return roof_masks, relative_altitude, focalLengths

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--roofmasks', type=str, required=True, help='path to roof masks')
    parser.add_argument('--log_file', type=str, required=True, help='relative altitude and focal length')
    parser.add_argument('--save_dir_intermediate', type=str, required=True, help='path to save the intermediate results')
    parser.add_argument('--save_dir_final', type=str, required=True, help='path to save the final results')

    args = parser.parse_args()
    print(args.log_file)
    roof_masks, relative_altitude, focallength = read_logs(args.log_file)
    
    final_results = {}
    for i,j,k in zip(roof_masks, relative_altitude, focallength):
        net_area = get_area_from_mask(i, j, k, thresh=200)
        final_results[i] = net_area
    
    with open(os.path.join(args.save_dir_final, 'final_results.txt'), 'w') as f:
        for key, value in final_results.items():
            f.write('%s:%s\n' % (key, value))
    