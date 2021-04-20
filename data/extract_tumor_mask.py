import sys
import os
import argparse
import logging
import json
import time
import tqdm
import glob

from torch import nn
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import models
from models.model import resnet50, Classifier
from data.vsi2tif import vsi2tif

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

from test_stage.tif_producer import TIFPatchDataset  # noqa


path_to_bfconvert = '/home/ains/An/histologic/bftools/bfconvert'
java_env_var = {'BF_MAX_MEM': '4g'}
defaults = {
    'compression': 'lzw',
    'plane': 0,
    'tilesize': 1024,
    'quality': 85
}

## generate prob map with tif slides
parser = argparse.ArgumentParser(description='Get the probability map of tumor patch predictions given a TIF')
parser.add_argument('--vsi_path', default='/media/ains/DataA/An/lung/AC/or/', metavar='VSI_PATH', type=str,
                    help='Path to the input VSI file')
parser.add_argument('--tif_path', default='/home/ains/An/histologic/test/', metavar='TIF_PATH', type=str,
                    help='Path to the input TIF file')   #######change
parser.add_argument('--ckpt_path', default='/home/ains/An/histologic/TJ_CH/lung/256_patch_binary/result/best.ckpt', metavar='CKPT_PATH', type=str,
                    help='Path to the saved ckpt file of a pytorch model')    #######change
parser.add_argument('--cnn_path', default='/home/ains/An/histologic/TJ_CH/lung/256_patch_binary/result/cnn.json', metavar='CNN_PATH', type=str,
                    help='Path to the config file in json format related to the ckpt file')        #######change
parser.add_argument('--mask_path', default='/home/ains/An/histologic/TJ_CH/lung/masks/meta/tumor_mask/', metavar='MASK_PATH', type=str,
                    help='Path to the tissue mask of the input VSI file')        #######change
parser.add_argument('--tumor_mask_path', default='/home/ains/An/histologic/TJ_CH/lung/512_patch_DA/masks/tumor/', metavar='tumor_mask_PATH',
                    type=str, help='Path to the output tumor mask numpy file')         #######change
parser.add_argument('--num_workers', default=6, type=int, help='number of workers to use to make batch, default 5')
parser.add_argument("--compression",
                    help="Compression to use for tiff - default {} - use something that is compatible with "
                         "bfconvert and libvips - no checks implemented yet!".format(defaults['compression']),
                    default=defaults['compression'])
parser.add_argument("--plane", help="Plane to use from VSI - default {}".format(defaults['plane']),
                    default=defaults['plane'])
parser.add_argument("--tilesize", help="Tilesize to use during conversion and in final image - default {}".format(
    defaults['tilesize']), default=defaults['tilesize'])
parser.add_argument("--quality", help="Quality value for (if used by compression) final image - default {}".format(
    defaults['quality']), default=defaults['quality'])


def chose_model(mod):
    if mod == 'resnet18':
        model = models.resnet18(pretrained=False)
    else:
        raise Exception("I have not add any models. ")
    return model


def get_probs_map(model, dataloader):
    probs_map = np.zeros(dataloader.dataset._mask.shape)
    num_batch = len(dataloader)

    with torch.no_grad():
        count = 0
        time_now = time.time()
        print('crop start')
        for (data, x_mask, y_mask) in tqdm.tqdm(dataloader, total=num_batch, leave=False):
            data = Variable(data.cuda(non_blocking=True))

            output = model(data)
            # because of torch.squeeze at the end of forward in resnet.py, if the
            # len of dim_0 (batch_size) of data is 1, then output removes this dim.
            # should be fixed in resnet.py by specifying torch.squeeze(dim=2) later
            if len(output.shape) == 1:
                probs = output.sigmoid().cpu().data.numpy().flatten()
            else:
                probs = output[:,
                               :].sigmoid().cpu().data.numpy().flatten()
            probs_map[x_mask, y_mask] = probs
            count += 1

        print('crop done')
        time_spent = time.time() - time_now
        logging.info(
            '{}, flip : {}, rotate : {}, batch : {}/{}, Run Time : {:.2f}'
            .format(
                time.strftime("%Y-%m-%d %H:%M:%S"), dataloader.dataset._flip,
                dataloader.dataset._rotate, count, num_batch, time_spent))

    return probs_map


def make_dataloader(args, mask_path, tif_img, cnn, flip='NONE', rotate='NONE'):
    batch_size = cnn['batch_size']
    level = 6
    num_workers = args.num_workers

    dataloader = DataLoader(
        TIFPatchDataset(tif_img, mask_path,
                        image_size=cnn['image_size'],
                        crop_size=cnn['crop_size'], level=level, normalize=True,
                        flip=flip, rotate=rotate), batch_size=batch_size, num_workers=num_workers, drop_last=False)

    return dataloader


def run(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    logging.basicConfig(level=logging.INFO)

    with open(args.cnn_path) as f:
        cnn = json.load(f)
    vsis = glob.glob(args.vsi_path + "*.vsi")[100:150]
    ckpt = torch.load(args.ckpt_path)
    model = chose_model(cnn['model'])
    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features, 1)
    model.load_state_dict(ckpt['state_dict'])
    model = model.cuda().eval()

# ########################
#     npys = glob.glob('/home/ains/An/histologic/TJ_CH/lung/512_patch_DA/masks/meta/' + "*.npy")
#     n_names = []
#     for npy in npys:
#         n_name = os.path.basename(npy)[:-4]
#         n_names.append(n_name)
#
#     names = []
#     for vsi in vsis:
#         name = os.path.basename(vsi)[:-4]
#         if name not in n_names:
#             names.append(name)
# ########################
    results_dict = {}
    for vsi in vsis:
    # for name in names:
        name = os.path.basename(vsi)[:-4]
        #vsi = args.vsi_path + name + '.vsi'
        tif_img = vsi2tif(vsi, results_dict, path_to_bfconvert, java_env_var, args)
        mask_path = args.mask_path + name + '.npy'
        dataloader = make_dataloader(args, mask_path, tif_img, cnn, flip='NONE', rotate='NONE')

        probs_map = get_probs_map(model, dataloader)
        save_path = args.tumor_mask_path + name + '.npy'
        np.save(save_path, probs_map)
        os.remove(tif_img)
        print(vsi + 'done!')


def main():
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
