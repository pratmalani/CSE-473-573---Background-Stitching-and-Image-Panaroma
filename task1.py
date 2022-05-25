#Only add your code inside the function (including newly improted packages)
# You can design a new function and call the new function in the given functions.
# Not following the project guidelines will result in a 10% reduction in grades

import cv2
import numpy as np
import matplotlib.pyplot as plt


def stitch_background(img1, img2, savepath=''):
    "The output image should be saved in the savepath."
    "Do NOT modify the code provided."

    img_1 = np.zeros((img2.shape[0] + 500, img2.shape[1] + 500, 3), dtype=np.uint8)
    img_1[250:250 + img2.shape[0], 250:250 + img2.shape[1]] = img2
    img2 = img_1

    SIFT = cv2.xfeatures2d.SIFT_create(500)
    keypoints_1, descriptors_1 = SIFT.detectAndCompute(img1, None)
    keypoints_2, descriptors_2 = SIFT.detectAndCompute(img2, None)

    val = []
    for i, d_1 in enumerate(descriptors_1):
        kp2_list = []
        for j, d_2 in enumerate(descriptors_2):
            euclidean_distance = np.linalg.norm(d_1 - d_2)
            kp2_list.append([i, j, euclidean_distance])
        kp2_list.sort(key=lambda x: x[2])

        if kp2_list[0][2]/ kp2_list[1][2] < 0.7:
            val.append(kp2_list[0])

    kp11 = []
    kp22 = []
    for i in val:
        kp11.append(np.float32(keypoints_1[i[0]].pt))
        kp22.append(np.float32(keypoints_2[i[1]].pt))
    kp11 = np.array(kp11)
    kp22 = np.array(kp22)

    homography, matchesMask = cv2.findHomography(kp11, kp22, cv2.RANSAC, 5.0)

    buffer_1 = img1.shape[1] + img2.shape[1]
    buffer_2 = img1.shape[0] + img2.shape[0]
    stitched_img = cv2.warpPerspective(img1, homography, (buffer_1, buffer_2))

    reshape = np.zeros((stitched_img.shape))
    reshape[:img2.shape[0], :img2.shape[1]] = img2
    reshape = reshape.astype(np.float32)
    stitched_img = stitched_img.astype(np.float32)
    img_1 = stitched_img
    img_2 = reshape

    result = np.zeros((img_1.shape))
    for index, x in np.ndenumerate(result):
        i = index[0]
        j = index[1]
        k = index[2]
        condition_1 = img_2[i][j][k] == 0 and img_1[i][j][k] == 0
        condition_2 = img_2[i][j][k] - img_1[i][j][k] >= 0
        if condition_1:
            result[i][j][k] = 0
        elif not condition_1 and condition_2:
            result[i][j][k] = img_2[i][j][k]
        elif not condition_1 and not condition_2:
            result[i][j][k] = img_1[i][j][k]
    p1 = []
    p2 = []
    for index, x in np.ndenumerate(result):
        if x != 0:
            p1.append(index[0])
            p2.append(index[1])
    resultant_image = result[min(p1):max(p1), min(p1):max(p2)]

    cv2.imwrite("task1.png", resultant_image)

    return resultant_image


if __name__ == "__main__":
    img1 = cv2.imread('./images/t1_1.png')
    img2 = cv2.imread('./images/t1_2.png')
    savepath = 'task1.png'
    stitch_background(img1, img2, savepath=savepath)
