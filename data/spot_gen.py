import os
import sys
import logging
import argparse
import glob
import numpy as np

sys.path.append(os.path.join(os.path.abspath(__file__), "/../../"))


parser = argparse.ArgumentParser(description="Get center points of patches "
                                             "from mask")
parser.add_argument("mask_path", default=None, metavar="MASK_PATH", type=str,
                    help="Path to the mask npy file")
parser.add_argument("txt_path", default=None, metavar="TXT_PATH", type=str,
                    help="Path to the txt file")
parser.add_argument("patch_number", default=None, metavar="PATCH_NUMB", type=int,
                    help="The number of patches extracted from WSI")
parser.add_argument("--level", default=6, metavar="LEVEL", type=int,
                    help="Bool format, whether or not")


class patch_point_in_mask_gen(object):
    '''
    extract centre point from mask
    inputs: mask path, centre point number
    outputs: centre point
    '''

    def __init__(self, mask_path, number):
        self.mask_path = mask_path
        self.number = number
        self.dele = 25

    def get_patch_point(self):
        mask_tissue = np.load(self.mask_path)

        mask_tissue[0:self.dele + 10, :] = 0  # top to down
        mask_tissue[:, 0:self.dele + 10] = 0  # left to right
        mask_tissue[mask_tissue.shape[0] - self.dele - 10:, :] = 0  # down to top
        mask_tissue[:, mask_tissue.shape[1] - self.dele - 10:] = 0  # right to left

        X_idcs, Y_idcs = np.where(mask_tissue)

        centre_points = np.stack(np.vstack((X_idcs.T, Y_idcs.T)), axis=1)

        if centre_points.shape[0] > self.number:
            sampled_points = centre_points[np.random.randint(centre_points.shape[0],
                                                             size=self.number), :]
        else:
            sampled_points = centre_points
        return sampled_points

#
# def run(args):
#     sampled_points = patch_point_in_mask_gen(args.mask_path, args.patch_number).get_patch_point()
#     sampled_points = (sampled_points * 2 ** args.level).astype(np.int32) # make sure the factor
#
#     mask_name = os.path.split(args.mask_path)[-1].split(".")[0]
#     name = np.full((sampled_points.shape[0], 1), mask_name)
#     center_points = np.hstack((name, sampled_points))
#
#     txt_path = args.txt_path
#
#     with open(txt_path, "a") as f:
#         np.savetxt(f, center_points, fmt="%s", delimiter=",")
#
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

maskpath = '/home/ains/An/histologic/TJ_CH/lung/512_patch_DA/masks/meta/'
txtpath = '/home/ains/An/histologic/TJ_CH/lung/512_patch_DA/spots/train_meta/'
vsis = glob.glob(maskpath + "*.npy")
patch_number = 500
level = 4  # make sure the factor

for mask_path in vsis:
    txt_path = txtpath + os.path.basename(mask_path)[:-4] + '.txt'
    sampled_points = patch_point_in_mask_gen(mask_path, patch_number).get_patch_point()
    sampled_points = (sampled_points * 2 ** level).astype(np.int32)  # make sure the factor

    mask_name = os.path.split(mask_path)[-1].split(".")[0]
    name = np.full((sampled_points.shape[0], 1), mask_name)
    center_points = np.hstack((name, sampled_points))

    txt_path = txt_path

    with open(txt_path, "a") as f:
        np.savetxt(f, center_points, fmt="%s", delimiter=",")
