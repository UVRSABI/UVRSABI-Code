# Used for visualization purpose, not much of use elsewhere so not maintained.
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import random
import cv2
class Visualization:
    def leftbuildingvisualize(self,leftbuild,cloud):
        print(len(cloud))
        cloud[0].paint_uniform_color([0, 0, 0])
        cloud[1].paint_uniform_color([0, 0, 1])
        cloud[2].paint_uniform_color([1, 0, 0])
        cloud[3].paint_uniform_color([0, 1, 1])
        cloud[4].paint_uniform_color([0, 0.75, 0.25])
        cloud[5].paint_uniform_color([0, 0.25, 0.75])
        # cloud[-4+10].paint_uniform_color([1, 0, 0])
        # cloud[-4+11].paint_uniform_color([0, 0.5, 1])
        o3d.visualization.draw_geometries([leftbuild, cloud[-4+4], cloud[-4+5], cloud[-4+6], cloud[-4+7], cloud[-4+8], cloud[-4+9]],
                                        zoom=0.8,
                                        front=[-0.4999, -0.1659, -0.8499],
                                        lookat=[2.1813, 2.0619, 2.0999],
                                        up=[0.1204, -0.9852, 0.1215])
        vis= o3d.visualization.Visualizer()
        vis.create_window(visible = False)
        vis.add_geometry(leftbuild)
        for i in range(0, 6):
            vis.add_geometry(cloud[i])
        img = vis.capture_screen_float_buffer(True)
        img = np.asarray(img)
        img1 = cv2.normalize(img, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
        img1 = img1.astype(np.uint8)
        cv2.imshow("img3", img1)
        print(np.amax(img), np.amin(img1))
        cv2.waitKey(0)
        cv2.imwrite('LeftSegment.png',img1)
    def rightbuildingvisualize(self,rightbuild,cloud):
        cloud[0].paint_uniform_color([0, 0, 1])
        cloud[1].paint_uniform_color([1, 0, 0])
        cloud[2].paint_uniform_color([0, 1, 0])
        cloud[3].paint_uniform_color([0, 1, 1])
        cloud[4].paint_uniform_color([0, 0.75, 0.25])
        cloud[5].paint_uniform_color([0, 0.25, 0.75])
        cloud[6].paint_uniform_color([1, 0, 0])
        cloud[7].paint_uniform_color([0, 0.5, 1])
        o3d.visualization.draw_geometries([rightbuild, cloud[0], cloud[1], cloud[2], cloud[3], cloud[4], cloud[5], cloud[6], cloud[7]],
                                        zoom=0.8,
                                        front=[-0.4999, -0.1659, -0.8499],
                                        lookat=[2.1813, 2.0619, 2.0999],
                                        up=[0.1204, -0.9852, 0.1215])
        vis= o3d.visualization.Visualizer()
        vis.create_window(visible = False)
        vis.add_geometry(rightbuild)
        for i in range(8):
            vis.add_geometry(cloud[i])
        img = vis.capture_screen_float_buffer(True)
        img = np.asarray(img)
        img1 = cv2.normalize(img, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
        img1 = img1.astype(np.uint8)
        cv2.imshow("img1.png", img1)

        cv2.waitKey(0)
        cv2.imwrite('RightSegment.png',img1)