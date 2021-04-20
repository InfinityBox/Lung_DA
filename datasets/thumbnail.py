# public
import os
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from albumentations import (CropNonEmptyMaskIfExists, Rotate, HueSaturationValue, )


TABLE = {63: 3, 127: 2, 255: 1}  # psp: 255, ac: 127, background/lung: 0


class ThumbnailDataset(Dataset):
    def __init__(self, dir: str, transform):
        assert os.path.exists(dir), "DirectoryError: {} doesn't exist.".format(dir)
        self.list_image = self.traverse(dir)
        self.transform = transform

    def __len__(self):
        return len(self.list_image)

    def __getitem__(self, item):
        image = self.load_img(item)
        mask = self.load_mask(item)
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        image = np.transpose(image, [2,0,1])
        mask = mask.astype(np.int64)
        return image, mask

    def load_img(self, item: int):
        assert 0 <= item < len(self), "ParamError: item is out of range"
        return cv2.imread(self.list_image[item] + ".png")[..., ::-1]

    def load_mask(self, item: int):
        assert 0 <= item < len(self), "ParamError: item is out of range"
        mask = cv2.imread(self.list_image[item] + "_mask.png", cv2.IMREAD_GRAYSCALE)
        for origin, target in TABLE.items():
            mask[mask==origin] = target
        return mask

    def traverse(self, dir: str):
        list_image = []
        for file in os.listdir(dir):
            if "mask" not in file:
                list_image.append(os.path.join(dir, file.split(".")[0]))
        return list_image


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    from albumentations import (CropNonEmptyMaskIfExists, Rotate, Compose, Resize, Normalize)
    mean_data = [203.81873346 / 255, 116.51206044 / 255, 152.45270792 / 255]
    std_data = [38.29048771 / 255, 52.78056623 / 255, 44.04232756 / 255]

    aug = Compose([
        CropNonEmptyMaskIfExists(992, 992, p=1),
        Rotate((-90, 90), cv2.INTER_LINEAR, cv2.BORDER_REFLECT_101, p=1),
        Resize(992, 992),
        Normalize(mean=mean_data, std=std_data, p=1),
    ])

    dataset = ThumbnailDataset(dir="I:\\Out", transform=aug)
    for idx in range(len(dataset)):
        image, mask = dataset[idx]
        plt.subplot(1,2,1)
        plt.imshow(image)
        plt.subplot(1,2,2)
        plt.imshow(mask)
        plt.show()

    print("End.")



