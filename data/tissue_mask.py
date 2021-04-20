# public
import os
import numpy as np
import glob
from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu
import javabridge
import bioformats
# private
from data.vsi_reader import VsiReader


javabridge.start_vm(class_path=bioformats.JARS)
level = 6
RGB_min = 50
vsipath = '/media/ains/DataA/An/lung/AC/or/'
outputpath = '/home/ains/An/histologic/TJ_CH/lung/masks/meta/tumor_mask/'
vsis = glob.glob(vsipath + "*.vsi")

for vsi_path in vsis:
    npy_path = outputpath + os.path.basename(vsi_path)[:-4] + '.npy'
    reader = VsiReader(vsi_path)
    img_RGB = np.transpose(reader.getImage(level), axes=[1, 0, 2])
    img_HSV = rgb2hsv(img_RGB)

    background_R = img_RGB[:, :, 0] > threshold_otsu(img_RGB[:, :, 0])
    background_G = img_RGB[:, :, 1] > threshold_otsu(img_RGB[:, :, 1])
    background_B = img_RGB[:, :, 2] > threshold_otsu(img_RGB[:, :, 2])
    tissue_RGB = np.logical_not(background_R & background_G & background_B)
    tissue_S = img_HSV[:, :, 1] > threshold_otsu(img_HSV[:, :, 1])
    min_R = img_RGB[:, :, 0] > RGB_min
    min_G = img_RGB[:, :, 1] > RGB_min
    min_B = img_RGB[:, :, 2] > RGB_min

    tissue_mask = tissue_S & tissue_RGB & min_R & min_G & min_B

    np.save(npy_path, tissue_mask)

javabridge.kill_vm()

