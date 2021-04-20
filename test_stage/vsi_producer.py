import numpy as np
from torch.utils.data import Dataset
import javabridge
import bioformats
from data.vsi_reader import VsiReader


class VSIPatchDataset(Dataset):

    def __init__(self, vsi_path, mask_path, image_size=256, crop_size=224, level=6,
                 normalize=True, flip='NONE', rotate='NONE'):
        self._vsi_path = vsi_path
        self._level = level
        self._mask_path = mask_path
        self._image_size = image_size
        self._crop_size = crop_size
        self._normalize = normalize
        self._flip = flip
        self._rotate = rotate
        self._pre_process()


    def _pre_process(self):
        self._mask = np.load(self._mask_path)

        # if X_slide / X_mask != Y_slide / Y_mask:
        #     raise Exception('Slide/Mask dimension does not match ,'
        #                     'X_slide / X_mask: {} / {}, '
        #                     'Y_slide / Y_mask: {} / {}'
        #                     .format(X_slide, X_mask, Y_slide, Y_mask))

        # javabridge.start_vm(class_path=bioformats.JARS)
        self._slide = VsiReader(self._vsi_path)

        self._resolution = 2**self._level
        if not np.log2(self._resolution).is_integer():
            raise Exception('Resolution (X_slide / X_mask) is not power of 2: '
                            '{}'.format(self._resolution))

        #####  mention: got to rethink a more flexible way
        if self._image_size > self._resolution:
            dele = round((self._image_size - self._resolution) / (2 * self._resolution))
            self._mask[0:dele+5, :] = 0  # top to down
            self._mask[:, 0:dele+5] = 0  # left to right
            self._mask[self._mask.shape[0] - dele-5:, :] = 0  # down to top
            self._mask[:, self._mask.shape[1] - dele-5:] = 0  # right to left

        # all the idces for tissue region from the tissue mask
        self._X_idcs, self._Y_idcs = np.where(self._mask)
        self._idcs_num = len(self._X_idcs)

    def __len__(self):
        return self._idcs_num

    def __getitem__(self, idx):

        x_mask, y_mask = self._X_idcs[idx], self._Y_idcs[idx]

        x_center = int((x_mask + 0.5) * self._resolution)
        y_center = int((y_mask + 0.5) * self._resolution)

        x = int(x_center - self._image_size / 2)
        y = int(y_center - self._image_size / 2)

        #javabridge.start_vm(class_path=bioformats.JARS)
        img = self._slide.getRegion((x, y, self._image_size, self._image_size), 0)
        # javabridge.kill_vm()

            # PIL image:   H x W x C
            # torch image: C X H X W
        img = np.array(img, dtype=np.float32).transpose((2, 0, 1))

        if self._normalize:
            img = (img - 128.0) / 128.0

        return (img, x_mask, y_mask)
