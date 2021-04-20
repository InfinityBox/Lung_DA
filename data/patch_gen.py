import sys
import os
import argparse
import logging
import time
from shutil import copyfile
from multiprocessing import Value, Lock, Pool
import glob
from data.vsi_reader import VsiReader
import javabridge
import bioformats
from PIL import Image

# sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')
#
# parser = argparse.ArgumentParser(description='Generate patches from a given '
#                                  'list of coordinates')
# parser.add_argument('--wsi_path', default='/media/ains/My Passport/CAMELYON16/training/tumor/', type=str,
#                     help='Path to the input directory of WSI files')
# parser.add_argument('--coords_path', default='/home/ains/An/histologic/SLFCD-master/train_spot_tumor/',
#                     type=str, help='Path to the input list of coordinates')
# parser.add_argument('--patch_path', default='/home/ains/An/histologic/SLFCD-master/PATCHES_TUMOR_TRAIN/', type=str,
#                     help='Path to the output directory of patch images')
# parser.add_argument('--patch_size', default=256, type=int, help='patch size, '
#                     'default 768')
# parser.add_argument('--level', default=0, type=int, help='level for WSI, to '
#                     'generate patches, default 0')
# parser.add_argument('--num_process', default=1, type=int,
#                     help='number of mutli-process, default 5')

count = Value('i', 0)
lock = Lock()


def process(opt):
    i, pid, x_center, y_center, args = opt[0], opt[1], opt[2], opt[3], opt[4]
    # i, pid, x_center, y_center, args = opts
    x = int(int(x_center) - args[0][3] / 2)
    y = int(int(y_center) - args[0][3] / 2)
    #javabridge.start_vm(class_path=bioformats.JARS)
    slide = VsiReader(args[0][0])
    img = Image.fromarray(slide.getRegion((x, y, args[0][3], args[0][3]), args[0][4])).convert('RGB')

    img.save(os.path.join(args[0][2], args[0][6] + str(i) + '.png'))

    global lock
    global count

    with lock:
        count.value += 1
        if (count.value) % 100 == 0:
            logging.info('{}, {} patches generated...'
                         .format(time.strftime("%Y-%m-%d %H:%M:%S"),
                                 count.value))

#javabridge.start_vm(class_path=bioformats.JARS)
level = 2
patch_size = 256
num_process = 1
# 1.generate tumor patches
vsipath = '/media/ains/DataA/An/lung/AC/metastatic/'
coordspath = '/home/ains/An/histologic/TJ_CH/lung/512_patch_DA/spots/train_meta/'
patch_path = '/home/ains/An/histologic/TJ_CH/lung/512_patch_DA/train/meta/'
# # 2.generate normal patches in tumor slide
# vsipath = '/media/ains/DataA/An/lung/AC/'
# coordspath = '/home/ains/An/histologic/TJ_CH/lung/spots/spot_tumor_normal/'
# patch_path = '/home/ains/An/histologic/TJ_CH/lung/256_patch/train/normal/'
# 3.generate normal patches in normal slide
# vsipath = '/media/ains/DataA/An/lung/lung/'
# coordspath = '/home/ains/An/histologic/TJ_CH/lung/spots/spot_normal/'
# patch_path = '/home/ains/An/histologic/TJ_CH/lung/256_patch/train/normal/'
vsis = glob.glob(coordspath + "*.txt")

for vsi_path in vsis:
    name = os.path.basename(vsi_path)[:-4]
    print(name)
    coords_path = vsi_path
    vsi = vsipath + name + '.vsi'
    args = []
    args.append((vsi, coords_path, patch_path, patch_size, level, num_process, name))
    if not os.path.exists(patch_path):
        os.mkdir(patch_path)

    copyfile(coords_path, os.path.join(patch_path, 'list.txt'))

    opts_list = []
    infile = open(coords_path)
    for i, line in enumerate(infile):
        pid, x_center, y_center = line.strip('\n').split(',')
        opts_list.append((i, pid, x_center, y_center, args))
    infile.close()

    # process(opts_list, args)
    pool = Pool(processes=num_process)
    pool.map(process, opts_list)
    javabridge.kill_vm()

# # move patches to valid folder
# import shutil
# path1 = '/home/ains/An/histologic/SLFCD-master/PATCHES_TUMOR_TRAIN/'
# wsis = glob.glob(path1 + "tumor_100*.png")
# wsis1 = glob.glob(path1 + "tumor_101*.png")
# wsis2 = glob.glob(path1 + "tumor_102*.png")
# wsis3 = glob.glob(path1 + "tumor_103*.png")
# path = '/home/ains/An/histologic/SLFCD-master/PATCHES_TUMOR_VALID/'
# for i in wsis:
#     shutil.move(i, path + os.path.basename(i)[:-4] + '.png')

# import shutil
# path1 = '/home/ains/An/histologic/TJ_CH/lung/256_patch_DA/train/tumor/'
# move_wsis = '/home/ains/An/histologic/TJ_CH/lung/256_patch_DA/spots/test/tumor/'
# move_path = '/home/ains/An/histologic/TJ_CH/lung/256_patch_DA/val/tumor/'
# a = glob.glob(move_wsis + "*.txt")
# for path in a:
#     name = os.path.basename(path)[:-4]
#     wsis = glob.glob(path1 + name + "*.png")
#     for i in wsis:
#         shutil.move(i, move_path + os.path.basename(i)[:-4] + '.png')














#
#
# from __future__ import print_function
# import os, time, random
# import javabridge
# from multiprocessing import Pool
#
#
# def work11(p):
#     print(p, '-start-', os.getpid())
#     javabridge.start_vm(run_headless=True)
#     print(javabridge.run_script('java.lang.String.format("Hello, %s!", greetee);', dict(greetee='world')))
#     time.sleep(1.5)
#     print(p, "-finish")
#
# def worker1(msg):
#     t_start = time.time()
#     print("%s开始执行,进程号为%d" % (msg,os.getpid()))
#     # random.random()随机生成0~1之间的浮点数
#     time.sleep(random.random()*2)
#     t_stop = time.time()
#     print(msg,"执行完毕，耗时%0.2f" % (t_stop-t_start))



# pool = Pool(processes=3)
# a = [0,1,2,3,4,5,6,7,8,9]
# pool.map(work11, a)
# print("----start----")
# pool.close()
# pool.join()
# print("-----end-----")
# javabridge.kill_vm()
