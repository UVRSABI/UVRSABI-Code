import cv2
import numpy as np
import argparse

def CalculateOccupancy(roof_mask, object_mask):

    roof_mask = cv2.cvtColor(roof_mask, cv2.COLOR_BGR2GRAY)
    object_mask = cv2.cvtColor(object_mask, cv2.COLOR_BGR2GRAY)
    roof_mask = np.where(roof_mask > 0, 255, 0)
    object_mask = np.where(object_mask > 0, 255, 0)
    roof = np.where(roof_mask == 255)
    object = np.where(object_mask == 255)
    roof_area = len(roof[0])
    object_area = len(object[0])
    occupancy = object_area/roof_area
    return occupancy


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Calculate Percentage Occupancy')
    parser.add_argument('-r', '--roofmask', help='Roof Mask Path', required=True)
    parser.add_argument('-o', '--objectmask', help='Object Mask Path', required=True)
    parser.add_argument('-t', '--result_text_file', help='Result Text File Path', required=True)
    opt = parser.parse_args()

    roof_mask = cv2.imread(opt.roofmask)
    object_mask = cv2.imread(opt.objectmask)
    occupancy=CalculateOccupancy(roof_mask, object_mask)
    with open(opt.result_text_file, 'w') as f:
        f.write('The percentage of roof area occupied by objects is: ' + str(round(occupancy*100,2))+str('%'))