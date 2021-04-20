# public
import os
import cv2
import math
import numpy as np
from typing import List
from torch.utils.data import Dataset, DataLoader


TABLE = {63: 3, 127: 2, 255: 1}  # psp: 255, ac: 127, background/lung: 0


class ThumbnailPatchDataset(Dataset):
    def __init__(self, dir: str, transform, sizeTile: List[int], buffer: bool=True, path_buffer: str=None):
        """
        The dataset to train a high resolution images,
        which extract a image patch from original huge image by grid
        :param dir: the original image directory
        :param transform: augmenting method from albumentations
        :param sizeTile: the size of image patch, [width, height]
        :param buffer: whether save image patch in the disk or not
        :param path_buffer: the buffer directory, if you use buffer
        """
        assert os.path.exists(dir), "DirectoryError: {} doesn't exist.".format(dir)
        self.width_tile, self.height_tile = sizeTile
        self.transform = transform
        self.buffer = buffer
        self.path_buffer = path_buffer
        if self.buffer:
            os.path.exists(path_buffer), "DirectoryError: {} doesn't exist.".format(path_buffer)
        # traverse all image patches
        self.list_image = self.traverse(dir)

    def __len__(self):
        return len(self.list_image)

    def __getitem__(self, item):
        path, x, y = self.list_image[item]
        image = self.load_img(item)
        mask = self.load_mask(item)
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        image = np.transpose(image, [2,0,1])
        mask = mask.astype(np.int64)
        return image, mask, (os.path.basename(path), x, y)

    def load_img(self, item: int):
        assert 0 <= item < len(self), "ParamError: item is out of range"
        if self.buffer:
            base = cv2.imread(os.path.join(self.path_buffer, str(item) + ".png"))
        else:
            path_image, x, y = self.list_image[item]
            image_entire = cv2.imread(path_image + ".png")
            height_image, width_image = image_entire.shape[:2]
            base = np.zeros([self.height_tile, self.width_tile, 3], dtype=np.uint8)
            base[0:min(self.height_tile, (height_image-y*self.height_tile)),
                 0:min(self.width_tile, (width_image-x*self.width_tile))] = \
            image_entire[y*self.height_tile:min((y+1)*self.height_tile, height_image),
                         x*self.width_tile:min((x+1)*self.width_tile, width_image)]
        return base


    def load_mask(self, item: int):
        assert 0 <= item < len(self), "ParamError: item is out of range"
        if self.buffer:
            base = cv2.imread(os.path.join(self.path_buffer, str(item) + "_mask.png"), cv2.IMREAD_GRAYSCALE)
            base = base
        else:
            path_image, x, y = self.list_image[item]
            image_entire = cv2.imread(path_image + "_mask.png", cv2.IMREAD_GRAYSCALE)
            height_image, width_image = image_entire.shape[:2]
            base = np.zeros([self.height_tile, self.width_tile], dtype=np.uint8)
            base[0:min(self.height_tile, (height_image - y * self.height_tile)),
                 0:min(self.width_tile, (width_image - x * self.width_tile))] = \
            image_entire[y * self.height_tile:min((y + 1) * self.height_tile, height_image),
                         x * self.width_tile:min((x + 1) * self.width_tile, width_image)]
            base = base
        return base

    def traverse(self, dir: str):
        list_image = []
        id = 0
        for file in os.listdir(dir):
            if "mask" not in file:
                path_image = os.path.join(dir, file.split(".")[0])
                image = cv2.imread(path_image+".png")
                if self.buffer:
                    mask = cv2.imread(path_image+"_mask.png", cv2.IMREAD_GRAYSCALE)
                height_img, width_img = image.shape[:2]
                for i in range(math.ceil(width_img/self.width_tile)):
                    for j in range(math.ceil(height_img/self.height_tile)):
                        list_image.append([path_image, i, j])
                        if self.buffer and not os.path.exists(os.path.join(self.path_buffer, str(id)+".png")):
                            # if we use buffer and the buffer directory is clean, we save the image pathces
                            # original image
                            base = np.zeros([self.height_tile, self.width_tile, 3], dtype=np.uint8)
                            base[0:min(self.height_tile, (height_img - j * self.height_tile)),
                                 0:min(self.width_tile, (width_img - i * self.width_tile))] = \
                            image[j * self.height_tile:min((j + 1) * self.height_tile, height_img),
                                  i * self.width_tile:min((i + 1) * self.width_tile, width_img)]
                            cv2.imwrite(os.path.join(self.path_buffer, str(id)+".png"), base)
                            # mask
                            base = np.zeros([self.height_tile, self.width_tile], dtype=np.uint8)
                            base[0:min(self.height_tile, (height_img - j * self.height_tile)),
                                 0:min(self.width_tile, (width_img - i * self.width_tile))] = \
                            mask[j * self.height_tile:min((j + 1) * self.height_tile, height_img),
                                 i * self.width_tile:min((i + 1) * self.width_tile, width_img)]
                            cv2.imwrite(os.path.join(self.path_buffer, str(id) + "_mask.png"), base)
                            id += 1

        return list_image


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    from albumentations import (CropNonEmptyMaskIfExists, Rotate, Compose, Resize, Normalize)
    mean_data = [203.81873346 / 255, 116.51206044 / 255, 152.45270792 / 255]
    std_data = [38.29048771 / 255, 52.78056623 / 255, 44.04232756 / 255]

    aug = Compose([
        # CropNonEmptyMaskIfExists(992, 992, p=1),
        Rotate((-90, 90), cv2.INTER_LINEAR, cv2.BORDER_REFLECT_101, p=1),
        # Resize(992, 992),
        Normalize(mean=mean_data, std=std_data, p=1),
    ])

    dataset = ThumbnailPatchDataset(dir="H:\\test_in", transform=aug, sizeTile=[496, 496], buffer=True, path_buffer="H:\\test")
    for idx in range(1, len(dataset)):
        image, mask, (name, x, y) = dataset[idx]
        print(name, x, y)

    print("End.")
