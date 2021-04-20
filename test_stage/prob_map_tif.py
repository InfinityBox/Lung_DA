import sys
import os
import argparse
import logging
import json
import time
import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import models
from models.model import resnet50, Classifier

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

from tif_producer import TIFPatchDataset  # noqa


## generate prob map with tif slides
parser = argparse.ArgumentParser(description='Get the probability map of tumor patch predictions given a TIF')
parser.add_argument('--tif_path', default='/home/ains/An/histologic/500-95-534302-3.tif', metavar='TIF_PATH', type=str,
                    help='Path to the input TIF file')   #######change
parser.add_argument('--ckpt_path', default='/home/ains/An/histologic/TJ_CH/lung/256_patch_DA/result/train.ckpt', metavar='CKPT_PATH', type=str,
                    help='Path to the saved ckpt file of a pytorch model')    #######change
parser.add_argument('--cnn_path', default='/home/ains/An/histologic/TJ_CH/lung_proj/DA/cnn.json', metavar='CNN_PATH', type=str,
                    help='Path to the config file in json format related to the ckpt file')        #######change
parser.add_argument('--mask_path', default='/home/ains/An/histologic/TJ_CH/lung/masks/meta/tumor_mask/500-95-534302-3.npy', metavar='MASK_PATH', type=str,
                    help='Path to the tissue mask of the input VSI file')        #######change
parser.add_argument('--probs_map_path', default='/home/ains/An/histologic/TJ_CH/lung/256_patch_DA/result/probmap/t500-95-534302-3.npy', metavar='PROBS_MAP_PATH',
                    type=str, help='Path to the output probs_map numpy file')         #######change
parser.add_argument('--GPU', default='0', type=str, help='which GPU to use, default 0')
parser.add_argument('--num_workers', default=5, type=int, help='number of workers to use to make batch, default 5')
parser.add_argument('--eight_avg', default=0, type=int, help='if using average'
                    ' of the 8 direction predictions for each patch,'
                    ' default 0, which means disabled')


def chose_model(mod):
    if mod == 'resnet18':
        model = models.resnet18(pretrained=False)
    else:
        raise Exception("I have not add any models. ")
    return model


def get_probs_map(model, dataloader, classifier):
    probs_map = np.zeros(dataloader.dataset._mask.shape)
    num_batch = len(dataloader)

    with torch.no_grad():
        count = 0
        time_now = time.time()
        print('crop start')
        for (data, x_mask, y_mask) in tqdm.tqdm(dataloader, total=num_batch, leave=False):
            data = Variable(data.cuda(non_blocking=True))

            output = model(data)
            output = classifier(output)
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


def make_dataloader(args, cnn, flip='NONE', rotate='NONE'):
    batch_size = cnn['batch_size']
    level = 6
    num_workers = args.num_workers

    dataloader = DataLoader(
        TIFPatchDataset(args.tif_path, args.mask_path,
                        image_size=cnn['image_size'],
                        crop_size=cnn['crop_size'], level=level, normalize=True,
                        flip=flip, rotate=rotate), batch_size=batch_size, num_workers=num_workers, drop_last=False)

    return dataloader


def run(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU
    logging.basicConfig(level=logging.INFO)

    with open(args.cnn_path) as f:
        cnn = json.load(f)

    mask = np.load(args.mask_path)
    ckpt = torch.load(args.ckpt_path)
    model = resnet50(num_classes=2)
    model.load_state_dict(ckpt['state_dict1'])
    classifier = Classifier(num_classes=1)
    classifier.load_state_dict(ckpt['state_dict2'])
    model = model.cuda().eval()
    classifier = classifier.cuda().eval()

    if not args.eight_avg:

        dataloader = make_dataloader(
            args, cnn, flip='NONE', rotate='NONE')

        probs_map = get_probs_map(model, dataloader, classifier)
    else:
        probs_map = np.zeros(mask.shape)

        dataloader = make_dataloader(
            args, cnn, flip='NONE', rotate='NONE')
        probs_map += get_probs_map(model, dataloader)

        dataloader = make_dataloader(
            args, cnn, flip='NONE', rotate='ROTATE_90')
        probs_map += get_probs_map(model, dataloader)

        dataloader = make_dataloader(
            args, cnn, flip='NONE', rotate='ROTATE_180')
        probs_map += get_probs_map(model, dataloader)

        dataloader = make_dataloader(
            args, cnn, flip='NONE', rotate='ROTATE_270')
        probs_map += get_probs_map(model, dataloader)

        dataloader = make_dataloader(
            args, cnn, flip='FLIP_LEFT_RIGHT', rotate='NONE')
        probs_map += get_probs_map(model, dataloader)

        dataloader = make_dataloader(
            args, cnn, flip='FLIP_LEFT_RIGHT', rotate='ROTATE_90')
        probs_map += get_probs_map(model, dataloader)

        dataloader = make_dataloader(
            args, cnn, flip='FLIP_LEFT_RIGHT', rotate='ROTATE_180')
        probs_map += get_probs_map(model, dataloader)

        dataloader = make_dataloader(
            args, cnn, flip='FLIP_LEFT_RIGHT', rotate='ROTATE_270')
        probs_map += get_probs_map(model, dataloader)

        probs_map /= 8

    np.save(args.probs_map_path, probs_map)


def main():
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
