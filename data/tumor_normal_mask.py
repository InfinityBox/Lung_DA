import sys
import os
import argparse
import logging
import glob
import numpy as np

# sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")
#
# parser = argparse.ArgumentParser(description="Get the normal region"
#                                              " from tumor WSI ")
# parser.add_argument("tumor_path", default=None, metavar='TUMOR_PATH', type=str,
#                     help="Path to the tumor mask npy")
# parser.add_argument("tissue_path", default=None, metavar='TISSUE_PATH', type=str,
#                     help="Path to the tissue mask npy")
# parser.add_argument("normal_path", default=None, metavar='NORMAL_PATCH', type=str,
#                     help="Path to the output normal region from tumor WSI npy")

#
# def run(args):
#     tumor_mask = np.load(args.tumor_path)
#     tissue_mask = np.load(args.tissue_path)
#
#     normal_mask = tissue_mask & (~ tumor_mask)
#
#     np.save(args.normal_path, normal_mask)
#
# def main():
#     logging.basicConfig(level=logging.INFO)
#
#     args = parser.parse_args()
#     run(args)
#
#
# if __name__ == "__main__":
#     main()

tumorpath = '/home/ains/An/histologic/TJ_CH/lung/masks/tumor_mask/'
tissuepath = '/home/ains/An/histologic/TJ_CH/lung/masks/tissue_mask_AC/'
normalpath = '/home/ains/An/histologic/TJ_CH/lung/masks/tumor_normal_mask/'
wsis = glob.glob(tumorpath + "*.npy")

for tumor_path in wsis:
    tissue_path = tissuepath + os.path.basename(tumor_path)[:-4] + '.npy'
    normal_path = normalpath + os.path.basename(tumor_path)[:-4] + '.npy'
    tumor_mask = np.load(tumor_path)
    tissue_mask = np.load(tissue_path)

    normal_mask = tissue_mask & (~ tumor_mask)

    np.save(normal_path, normal_mask)
