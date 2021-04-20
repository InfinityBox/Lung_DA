# public
import os
import re
import cv2
import json
import time
import random
import numpy as np
from typing import Union, Tuple, List
import win32com.client
from collections import defaultdict

# private
from gist import parse_vsi_anno, parse_vsi_meta
from vsireader import VsiReader


TABLE = {"lung": 63, "ac": 127, "psp": 255}


class Generator(object):
    """
    the generator of low-resolution image of vsi
    :param work_dir: the directory of original vsi file
    """

    def __init__(self, work_dir: str):
        self.work_dir = work_dir
        self.shell = win32com.client.Dispatch("WScript.Shell")
        self.dict_path, num = self.traverse(self.work_dir)
        print(self.dict_path)

    def lnk_parse(self, path: str):
        shortcut = self.shell.CreateShortCut(path)
        path_target = shortcut.Targetpath  # 解析快捷方式，并且输出其指向路径
        return path_target

    def traverse(self, dir: str):
        assert os.path.exists(dir), "PathError: {} doesn't exist.".format(dir)
        file_dist = defaultdict(list)
        val_sum = 0
        for file in os.listdir(dir):
            path = self.lnk_parse(os.path.join(dir, file))
            if file[0] == "8":
                file_dist["lung"].append(path)
            elif file[0] == "5":
                file_dist["ac"].append(path)
            if file[0] == "1":
                file_dist["psp"].append(path)
            val_sum += 1
        return file_dist, val_sum

    def generate(self, path_out: str, dir_anno: str, layer: int):
        """
        generating method
        :param path_out: the output directory
        :param dir_anno: the directory of annotations
        :param layer: the layer you want to extract, e.g. 0 means 40x, 1 means 20x, and so on.
        :return: bool: return True if all image patch is generated.
        """
        assert os.path.exists(path_out), "PathError: {} doesn't exist.".format(path_out)
        assert os.path.exists(dir_anno), "PathError: {} doesn't exist.".format(dir_anno)

        for key, vsis in self.dict_path.items():
            print(key)
            value = TABLE[key]
            for vsi in vsis:
                reader = VsiReader(vsi)
                img = reader.getImage(layer)
                vsi_name = vsi.split("\\")[-1].split(".")[0]
                print(vsi_name)
                name = vsi_name + ".png"
                cv2.imwrite(os.path.join(path_out, name), img[..., ::-1])

                path_anno = os.path.join(dir_anno, vsi_name + ".vsi - 40x_annotation.json")
                base = np.zeros(img.shape[:2], dtype=np.uint8)
                if os.path.exists(path_anno):
                    anno = parse_vsi_anno(path_anno)
                    for elem in anno:
                        base = cv2.drawContours(base, [elem["contour"]//(2**layer)], 0, value, -1)
                name = vsi_name + "_mask.png"
                cv2.imwrite(os.path.join(path_out, name), base)
        return True


if __name__ == '__main__':
    import bioformats
    import javabridge
    import matplotlib.pyplot as plt

    javabridge.start_vm(class_path=bioformats.JARS)  # I code javabridge here, and maybe it will be proven wrong

    gen = Generator("G:\medical\TJU-PSP-try\Join")
    gen.generate(path_out="G:\\medical\\Out_5X", dir_anno="G:\\medical\\anno_meta", layer=3)
    javabridge.kill_vm()

    print("End.")
