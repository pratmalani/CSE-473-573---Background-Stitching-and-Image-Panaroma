# 1. Only add your code inside the function (including newly improted packages).
#  You can design a new function and call the new function in the given functions.
# 2. For bonus: Give your own picturs. If you have N pictures, name your pictures such as ["t3_1.png", "t3_2.png", ..., "t3_N.png"], and put them inside the folder "images".
# 3. Not following the project guidelines will result in a 10% reduction in grades

import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
from numpy import *


def stitch(imgmark, N,savepath=''):  # For bonus: change your input(N=*) here as default if the number of your input pictures is not 4.
    "The output image should be saved in the savepath."
    "The intermediate overlap relation should be returned as NxN a one-hot(only contains 0 or 1) array."
    "Do NOT modify the code provided."
    imgpath = [f'./images/{imgmark}_{n}.png' for n in range(1, N + 1)]
    imgs = []
    for ipath in imgpath:
        img = cv2.imread(ipath)
        imgs.append(img)
    "Start you code here"
    import numpy
    calc = numpy.zeros((N, N))
    for index, val in np.ndenumerate(calc):
        x = index[0]
        y = index[1]
        if x != y:
            sift = cv2.xfeatures2d.SIFT_create(500)
            descriptors_1 = sift.detectAndCompute(imgs[x], None)[1]
            descriptors_2 = sift.detectAndCompute(imgs[y], None)[1]
            match_counter = []
            for i, d1 in enumerate(descriptors_1):
                kp_list = []
                for j, d2 in enumerate(descriptors_2):
                    kp_list.append(((sum((d1 - d2) ** 2)), j))
                kp_list.sort(key=lambda tup: tup[0])
                if (kp_list[0][0] / kp_list[1][0]) < 0.7:
                    match_counter.append('a')
            if (len(match_counter) <= 20):

                calc[x][y] = 0
                calc[y][x] = 0
            elif (len(match_counter) > 20):
                calc[x][y] = 1
                calc[y][x] = 1
        else:
            calc[x][y] = 1;
    overlap_arr = calc

    sift = cv2.xfeatures2d.SIFT_create(500)
    keypoints_1, descriptors_1 = sift.detectAndCompute(imgs[1], None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(imgs[0], None)
    match_counter = []

    for i, d1 in enumerate(descriptors_1):
        kp_list = []
        for j, d2 in enumerate(descriptors_2):
            kp_list.append(((sum((d1 - d2) ** 2)), j))
        kp_list.sort(key=lambda tup: tup[0])
        if (kp_list[0][0] / kp_list[1][0]) < 0.7:
            match_counter.append((i, kp_list[0][1]))
    kp11 = np.float32([keypoints_1[val[0]].pt for val in match_counter]).reshape(-1, 1, 2)
    kp22 = np.float32([keypoints_2[val[1]].pt for val in match_counter]).reshape(-1, 1, 2)
    homography, matchesMask = cv2.findHomography(kp11, kp22, cv2.RANSAC, 5.0)
    ht1 = imgs[1].shape[0]
    wd1 = imgs[1].shape[1]
    ht2 = imgs[0].shape[0]
    wd2 = imgs[0].shape[1]
    kp1 = float32([[0, 0], [0, ht1], [wd1, ht1], [wd1, 0]]).reshape(-1, 1, 2)
    kp2 = float32([[0, 0], [0, ht2], [wd2, ht2], [wd2, 0]]).reshape(-1, 1, 2)
    kp = concatenate((kp2, cv2.perspectiveTransform(kp1, homography)), axis=0)
    [x_min, y_min] = int32(kp.min(axis=0)[0] - 0.5)
    [x_max, y_max] = int32(kp.max(axis=0)[0] + 0.5)
    overlap_mat = array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
    final = cv2.warpPerspective(imgs[1], overlap_mat.dot(homography), (x_max - x_min, y_max - y_min))
    final[-y_min:ht2 - y_min, -x_min:wd2 - x_min] = imgs[0]

    img = final

    for x in range(1, N - 1):
        sift = cv2.xfeatures2d.SIFT_create(500)
        keypoints_1, descriptors_1 = sift.detectAndCompute(img, None)
        keypoints_2, descriptors_2 = sift.detectAndCompute(imgs[x + 1], None)
        match_counter = []

        for i, d1 in enumerate(descriptors_1):
            kp_list = []
            for j, d2 in enumerate(descriptors_2):
                kp_list.append(((sum((d1 - d2) ** 2)), j))
            kp_list.sort(key=lambda tup: tup[0])
            if (kp_list[0][0] / kp_list[1][0]) < 0.7:
                match_counter.append((i, kp_list[0][1]))
        kp11 = np.float32([keypoints_1[val[0]].pt for val in match_counter]).reshape(-1, 1, 2)
        kp22 = np.float32([keypoints_2[val[1]].pt for val in match_counter]).reshape(-1, 1, 2)
        homography, matchesMask = cv2.findHomography(kp11, kp22, cv2.RANSAC, 5.0)
        ht1 = img.shape[0]
        wd1 = img.shape[1]
        ht2 = imgs[x + 1].shape[0]
        wd2 = imgs[x + 1].shape[1]
        kp1 = float32([[0, 0], [0, ht1], [wd1, ht1], [wd1, 0]]).reshape(-1, 1, 2)
        kp2 = float32([[0, 0], [0, ht2], [wd2, ht2], [wd2, 0]]).reshape(-1, 1, 2)
        kp = concatenate((kp2, cv2.perspectiveTransform(kp1, homography)), axis=0)
        [x_min, y_min] = int32(kp.min(axis=0)[0] - 0.5)
        [x_max, y_max] = int32(kp.max(axis=0)[0] + 0.5)
        overlap_mat = array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
        final = cv2.warpPerspective(img, overlap_mat.dot(homography), (x_max - x_min, y_max - y_min))
        final[-y_min:ht2 - y_min, -x_min:wd2 - x_min] = imgs[x + 1]

        img = final

    cv2.imwrite(savepath, img)
    return overlap_arr

if __name__ == "__main__":
    # task2
    overlap_arr = stitch('t2', N=4, savepath='task2.png')
    with open('t2_overlap.txt', 'w') as outfile:
        json.dump(overlap_arr.tolist(), outfile)
    # bonus
    overlap_arr2 = stitch('t3', N=5, savepath='task3.png')
    with open('t3_overlap.txt', 'w') as outfile:
        json.dump(overlap_arr2.tolist(), outfile)