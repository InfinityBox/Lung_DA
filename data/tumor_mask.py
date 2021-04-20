import os
import glob
import numpy as np
from data.vsi_reader import VsiReader
import cv2
from datasets.gist import parse_vsi_anno
import javabridge
import bioformats

# sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')
#
# parser = argparse.ArgumentParser(description='Get tumor mask of tumor-WSI and '
#                                              'save it in npy format')
# parser.add_argument('wsi_path', default=None, metavar='WSI_PATH', type=str,
#                     help='Path to the WSI file')
# parser.add_argument('json_path', default=None, metavar='JSON_PATH', type=str,
#                     help='Path to the JSON file')
# parser.add_argument('npy_path', default=None, metavar='NPY_PATH', type=str,
#                     help='Path to the output npy mask file')
# parser.add_argument('--level', default=6, type=int, help='at which WSI level'
#                     ' to obtain the mask, default 6')
#
#
# def run(args):
#
#     # get the level * dimensions e.g. tumor0.tif level 6 shape (1589, 7514)
#     slide = openslide.OpenSlide(args.wsi_path)
#     w, h = slide.level_dimensions[args.level]
#     mask_tumor = np.zeros((h, w)) # the init mask, and all the value is 0
#
#     # get the factor of level * e.g. level 6 is 2^6
#     factor = slide.level_downsamples[args.level]
#
#
#     with open(args.json_path) as f:
#         dicts = json.load(f)
#     tumor_polygons = dicts['positive']
#
#     for tumor_polygon in tumor_polygons:
#         # plot a polygon
#         name = tumor_polygon["name"]
#         vertices = np.array(tumor_polygon["vertices"]) / factor
#         vertices = vertices.astype(np.int32)
#
#         cv2.fillPoly(mask_tumor, [vertices], (255))
#
#     mask_tumor = mask_tumor[:] > 127
#     mask_tumor = np.transpose(mask_tumor)
#
#     np.save(args.npy_path, mask_tumor)
#
# def main():
#     logging.basicConfig(level=logging.INFO)
#
#     args = parser.parse_args()
#     run(args)
#
# if __name__ == "__main__":
#     main()
#

javabridge.start_vm(class_path=bioformats.JARS)
level = 6
vsipath = '/media/ains/DataA/An/lung/AC/'
npypath = '/home/ains/An/histologic/TJ_CH/lung/masks/tumor_mask/'
jsonpath = '/media/ains/DataA/An/lung/anno/'
vsis = glob.glob(vsipath + "*.vsi")

for vsi_path in vsis:
    name = os.path.basename(vsi_path)[:-4]
    npy_path = npypath + name + '.npy'
    json_path = jsonpath + name + '.vsi - 40x_annotation.json'
    slide = VsiReader(vsi_path)
    w, h = slide.getSize(level)
    mask_tumor = np.zeros((h, w))  # the init mask, and all the value is 0

    # get the factor of level * e.g. level 6 is 2^6
    factor = 2**level

    anno = parse_vsi_anno(json_path)
    tumor_polygons = [elem["contour"] // factor for elem in anno]

    for tumor_polygon in tumor_polygons:
        # plot a polygon
        cv2.fillPoly(mask_tumor, [tumor_polygon], (255))

    mask_tumor = mask_tumor[:] > 127
    mask_tumor = np.transpose(mask_tumor)

    np.save(npy_path, mask_tumor)
javabridge.kill_vm()
