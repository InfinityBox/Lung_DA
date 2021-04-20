import cv2
import os
import re
import numpy as np
import matplotlib.pyplot as plt

def recover(dir: str, title: str):
    """
    recover the entire heatmap by name of image patch.

    :param dir: the directory of image patches
    :param title:  the common part in name of image patches.
    the name of patches must be like 159-481048-2_10_7_1.png
    :return:
    """
    pattern = title + "_[\d]+_[\d]+_[\d]{1,3}.png"
    width, height = -1, -1
    repo = {}
    for file in os.listdir(dir):
        if re.match(pattern, file):
            x, y, tag = [int(elem) for elem in file.split(".")[0].split("_")[1:]]
            width = max(width, x)
            height = max(height, y)
            repo[tuple([y, x])] = tag

    heatmap = np.zeros([height+1, width+1])
    for loc, tag in repo.items():
        heatmap[loc] = tag

    plt.imshow(heatmap)
    plt.show()


if __name__ == '__main__':

    recover(dir="H://test", title="159-481048-2")

    print("End.")





