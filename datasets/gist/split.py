"""
split all json to validating dataset and training dataset
将全部文件分成训练集和测试集两部分，会移动测试集文件到指定文件夹
"""

# public
import os
import re
import random
import shutil


def split(path_work: str, path_out: str, prob: float=0.1):

    if not os.path.exists(path_work):
        raise NotADirectoryError("{} not a directory".format(path_work))

    if not os.path.exists(path_out):
        raise NotADirectoryError("{} not a directory".format(path_out))

    for idx_anno, name_anno in \
            enumerate([file for file in os.listdir(path_work) if "annotation" in file]):
        num = random.randint(0, 100)
        if num > int(100 * (1-prob)):
            print(name_anno, "bingo!")
            # move
            path_anno = os.path.join(path_work, name_anno)
            name_meta = re.sub("annotation", "meta", name_anno)
            path_meta = os.path.join(path_work, name_meta)
            shutil.move(path_anno, os.path.join(path_out, name_anno))
            shutil.move(path_meta, os.path.join(path_out, name_meta))
        else:
            print(name_anno)


if __name__ == '__main__':

    random.seed(2)
    split(path_work="I:\\GIST\\annos", path_out="I:\\GIST\\annos\\val", prob=0.2)

    print("End.")
