# public
import os
import cv2
import numpy as np
from typing import List
# import win32com.client
from collections import defaultdict
import glob

# private
from .gist import parse_vsi_anno
from data.vsi_reader import VsiReader


TABLE = {"lung": 63, "ac": 127, "psp": 255}


class Generator(object):
    """
    the generator of low-resolution image of vsi
    :param work_dir: the directory of original vsi file
    """

    def __init__(self, work_dir: str):
        self.work_dir = work_dir
        # self.shell = win32com.client.Dispatch("WScript.Shell")
        self.dict_path, num = self.traverse(self.work_dir)
        print(self.dict_path)

    # def lnk_parse(self, path: str):
    #     shortcut = self.shell.CreateShortCut(path)
    #     path_target = shortcut.Targetpath  # 解析快捷方式，并且输出其指向路径
    #     return path_target

    def traverse(self, dir: str):
        assert os.path.exists(dir), "PathError: {} doesn't exist.".format(dir)
        file_dist = defaultdict(list)
        files = glob.glob(dir+"/*")
        val_sum = 0
        for file in files:
            vsis = glob.glob(file + "/*.vsi")
            for vsi in vsis:
                name = os.path.basename(vsi)
                if name[0] == "8":
                    file_dist["lung"].append(vsi)
                elif name[0] == "5":
                    file_dist["ac"].append(vsi)
                elif name[0] == "1":
                    file_dist["psp"].append(vsi)
                val_sum += 1
        return file_dist, val_sum

    def generate(self, path_out: str, dir_anno: str, layer: int, size: List[int]):
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
                reader.setLayer(layer)
                reader.setTileSize(size)
                vsi_name = vsi.split("/")[-1].split(".")[0]
                print(vsi_name)
                num_tile_x, num_tile_y = reader.getNumTile()
                path_anno = os.path.join(dir_anno, vsi_name + ".vsi - 40x_annotation.json")
                anno = parse_vsi_anno(path_anno)
                contours = [elem["contour"]//(2**layer) for elem in anno]
                for i in range(num_tile_x):
                    for j in range(num_tile_y):
                        flag_in = False
                        tile, pos = reader.getTile(i, j)  # pos: ((x, y), (w, h))
                        # print(pos)
                        if tile.mean() < 210:  # 判断图像块均值并抛弃空白块， 这里的210是个人经验值。该步骤会降低运行速度
                            for contour in contours:  # test whehter the block is in the contour or not
                                if cv2.pointPolygonTest(contour, (pos[0][0] + pos[1][0]//2, pos[0][1] + pos[1][1]//2), 0) >= 0:
                                    flag_in = True
                                    break
                            if flag_in:
                                if tile.shape[1] < size[0] or tile.shape[0] < size[1]:
                                    base = np.zeros([size[1], size[0], 3], dtype=np.uint8)
                                    base[:tile.shape[0], :tile.shape[1]] = tile
                                    tile = base
                                cv2.imwrite(os.path.join(path_out, vsi_name + "_{}_{}_1.png".format(i, j)), tile)
                            else:
                                if tile.shape[1] < size[0] or tile.shape[0] < size[1]:
                                    base = np.zeros([size[1], size[0], 3], dtype=np.uint8)
                                    base[:tile.shape[0], :tile.shape[1]] = tile
                                    tile = base
                                cv2.imwrite(os.path.join(path_out, vsi_name + "_{}_{}_0.png".format(i, j)), tile)
                # path_anno = os.path.join(dir_anno, vsi_name + ".vsi - 40x_annotation.json")
                # base = np.zeros(img.shape[:2], dtype=np.uint8)
                # if os.path.exists(path_anno):
                #     anno = parse_vsi_anno(path_anno)
                #     for elem in anno:
                #         base = cv2.drawContours(base, [elem["contour"]//(2**layer)], 0, value, -1)
                # name = vsi_name + "_mask.png"
                # cv2.imwrite(os.path.join(path_out, name), base)
        return True


if __name__ == '__main__':
    import bioformats
    import javabridge

    javabridge.start_vm(class_path=bioformats.JARS)  # I code javabridge here, and maybe it will be proven wrong

    gen = Generator("F:\\Lung\\TJU-PSP-try\\Join")
    gen.generate(path_out="F:\\Lung\\Out5X", dir_anno="F:\\Lung\\data\\anno", layer=3, size=[256, 256])
    javabridge.kill_vm()

    print("End.")
