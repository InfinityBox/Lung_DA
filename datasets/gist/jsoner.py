# public
import os
import re
import cv2
import json
import time
import random
import numpy as np
from typing import Union, Tuple, List

# private
from .utils import parse_vsi_meta, parse_vsi_anno
from data.vsi_reader import VsiReader


class Jsoner(object):
    """
    Traverse all vsi and generate coco-format json

    TODO: Note that the RLE mask haven't implemented.
    """

    def __init__(self, work_dir: str, split: str,
                 classes_valid: Union[Tuple[str], List[str]] = ["certain", "NO", "uncertain"],
                 write: bool = False):

        assert split in ["train", "val", "test", "evaluate", "debug"], \
            "ParamError: split must be train, val, and test"

        self.work_dir = work_dir
        self.dir_annos = os.path.join(work_dir, "annos", split)
        self.writable = write
        # self.dir_annos = "I:\\GIST\\annos"
        self.classes_list = ["certain", "NO", "uncertain"]
        self.classes_valid = classes_valid
        self.counter_image = 0
        self.counter_vsi = 0
        self.counter_annotation = 0
        vsis, images, annotaions = self._vsis(self.dir_annos, mode="all")
        self._base = {
            "info": self._info(),
            "liencse": self._license(),
            "categories": self._categories(),
            "vsis": vsis,
            "images": images,
            "annotations": annotaions
        }

    def _vsis(self, dir: str, mode: str = "exist"):
        """
        construct vsi part of coco formats

        Parameters
        ----------
        - **mode**: str

        the slicing mode of entire wsi,
        "exist" means only the patch containing mito will be saved,
        "all" means all the patch will be saved, and
        the patch whose shape is smaller than fixed shape will be padding with 0
        """
        assert mode.lower() in ["exist", "all"], \
            "ParamError: mode({}) must be in [exist, all]".format(mode)

        if not os.path.exists(dir):
            raise NotADirectoryError(f"{dir} not found.")
        vsis = []
        images = []
        annotations = []
        if mode.lower() == "exist":
            for idx_meta, file_meta in \
                    enumerate([file for file in os.listdir(dir) if "meta" in file]):
                path_meta = os.path.join(dir, file_meta)
                vsi = self._vsi(path_meta=path_meta, id_vsi=self.counter_vsi + 1)  # this id is vsi_id
                vsis.append(vsi)
                path_anno = re.sub("meta", "annotation", path_meta)
                if os.path.exists(path_anno):
                    annos = parse_vsi_anno(path_anno)
                    basename = os.path.basename(path_anno)
                    name_vsi = basename.split("_")[0] + ".vsi"
                    path_vsi = os.path.join(self.work_dir, "data", name_vsi)
                    reader = VsiReader(path=path_vsi, tilesize_init=(992, 992))
                    numX, numY = reader.getNumTile()
                    for idxX in range(numX):
                        for idxY in range(numY):
                            field = reader.getTileField(idxX, idxY)  # x1y1x2y2
                            flag_object_image = False
                            for anno in annos:
                                bbox = anno["bbox"]
                                bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
                                if self._testIn(bbox, field) and anno["tag"] in self.classes_valid:
                                    flag_object_image = True
                                    bbox_new = self.clip(bbox, field)  # x1y1x2y2
                                    bbox_deshift = [bbox_new[0] - field[0], bbox_new[1] - field[1],
                                                    bbox_new[2] - field[0], bbox_new[3] - field[1], ]
                                    class_bbox = self.classes_list.index(anno["tag"]) + 1
                                    annotation = self._annotation(id=self.counter_annotation + 1,
                                                                  id_image=self.counter_image + 1,
                                                                  id_category=class_bbox,
                                                                  bbox=bbox_deshift)
                                    annotations.append(annotation)
                                    self.counter_annotation += 1
                            if flag_object_image:
                                name_file = "{idimg}_{idvsi}_{x}_{y}.png".format(idimg=self.counter_image + 1,
                                                                                 idvsi=basename.split("_")[0],
                                                                                 x=field[0],
                                                                                 y=field[1])
                                coco_url = os.path.join("/mnt/nvme/GIST/testadd2020", name_file)
                                image = self._image(location=field[:2],
                                                    id=self.counter_image + 1,
                                                    file_name=name_file,
                                                    id_vsi=self.counter_vsi + 1,
                                                    url=coco_url)
                                block, _ = reader.getTile(indexTileX=idxX, indexTileY=idxY)  # read image block
                                if self.writable:
                                    cv2.imwrite(coco_url, block[..., ::-1])  # save images
                                    images.append(image)
                                self.counter_image += 1
                self.counter_vsi += 1

        elif mode.lower() == "all":
            for idx_meta, file_meta in \
                    enumerate([file for file in os.listdir(dir) if "meta" in file]):
                path_meta = os.path.join(dir, file_meta)
                vsi = self._vsi(path_meta=path_meta, id_vsi=self.counter_vsi + 1)  # this id is vsi_id
                vsis.append(vsi)

                basename = os.path.basename(path_meta)
                print(basename.split("_")[0])
                flag_anno_exist = False
                path_anno = re.sub("meta", "annotation", path_meta)
                if os.path.exists(path_anno):
                    annos = parse_vsi_anno(path_anno)
                    flag_anno_exist = True
                name_vsi = basename.split("_")[0] + ".vsi"
                path_vsi = os.path.join(self.work_dir, "data", name_vsi)
                reader = VsiReader(path=path_vsi, tilesize_init=(992, 992))
                numX, numY = reader.getNumTile()
                for idxX in range(numX):
                    for idxY in range(numY):
                        flag_object_image = False
                        field = reader.getTileField(idxX, idxY)  # x1y1x2y2
                        if flag_anno_exist:
                            for anno in annos:
                                bbox = anno["bbox"]
                                bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
                                if self._testIn(bbox, field) and anno["tag"] in self.classes_valid:
                                    flag_object_image = True
                                    bbox_new = self.clip(bbox, field)  # x1y1x2y2
                                    bbox_deshift = [bbox_new[0] - field[0], bbox_new[1] - field[1],
                                                    bbox_new[2] - field[0], bbox_new[3] - field[1], ]
                                    class_bbox = self.classes_list.index(anno["tag"]) + 1
                                    annotation = self._annotation(id=self.counter_annotation + 1,
                                                                  id_image=self.counter_image + 1,
                                                                  id_category=class_bbox,
                                                                  bbox=bbox_deshift)
                                    annotations.append(annotation)
                                    self.counter_annotation += 1
                        if flag_object_image:
                            block, _ = reader.getTile(indexTileX=idxX, indexTileY=idxY)  # read image block
                        else:
                            dice = random.randint(1, 10000)
                            if dice <= 128:
                                block, _ = reader.getTile(indexTileX=idxX, indexTileY=idxY)  # read image block
                                if block.mean() >= 220:  # remove the white bachground
                                    continue   # moderate the computation problem
                            else:
                                continue
                        if block.shape[0] < 992 or block.shape[1] < 992:  # padding the image
                            base = np.zeros([992, 992, 3], dtype=np.uint8)
                            base[:block.shape[0], :block.shape[1]] = block
                        else:
                            base = block
                        print("{x}/{h}, {y}/{v}".format(h=numX, v=numY, x=idxX, y=idxY))
                        name_file = "{idimg}_{idvsi}_{x}_{y}.png".format(idimg=self.counter_image + 1,
                                                                         idvsi=basename.split("_")[0],
                                                                         x=field[0],
                                                                         y=field[1])
                        coco_url = os.path.join("/mnt/nvme/GIST/trainadd2020", name_file)
                        image = self._image(location=field[:2],
                                            id=self.counter_image + 1,
                                            file_name=name_file,
                                            id_vsi=self.counter_vsi + 1,
                                            url=coco_url)
                        if self.writable:
                            cv2.imwrite(coco_url, base[..., ::-1])  # save RGB images
                            images.append(image)
                        self.counter_image += 1
                self.counter_vsi += 1
        return vsis, images, annotations

    def _annotation(self, id: int, id_image: int, id_category: int, bbox: list or tuple, ):
        """
        construct annotation element in coco json.

        Parameter
        ---------
        id : int
            the id of annotation
        id_image : int
            the id of corresponding image
        id_category : int
            the id of category this annotaion belongs to
        bbox : list or tuple
            the bounding box [x_min, y_min, x_max, y_max]
        """
        annotation = {
            "id": id,
            "image_id": id_image,
            "category_id": id_category,
            "segmentation": [],  # TODO : makeup the segmentaion calculation of bbox
            "area": (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]),
            "bbox": [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]],
            "iscrowd": 0,
        }
        return annotation

    def _image(self, location: tuple or list, id: int,
               file_name: str, id_vsi: int, url: str):
        image = {
            "id": id,
            "width": 992,
            "height": 992,
            "file_name": file_name,
            "vsi_id": id_vsi,
            "x": location[0],
            "y": location[1],
            "license": 0,
            "flickr_url": "",
            "coco_url": url,
            "date_captured": self._get_time_stamp(),
        }
        return image

    def clip(self, bbox: tuple or list, field: tuple or list) -> tuple:
        """
        clip the bbox to adjust the image field

        Parameter
        ---------
        bbox : tuple or list
            [x_min, y_min, x_max, y_max]
        field:tuple or list
            [x_min, y_min, x_max, y_max]

        Return
        -------
            (tuple or list) [x_min, y_min, x_max, y_max] new bounding box
        """
        x1_new = max(bbox[0], field[0])
        y1_new = max(bbox[1], field[1])
        x2_new = min(bbox[2], field[2])
        y2_new = min(bbox[3], field[3])

        return x1_new, y1_new, x2_new, y2_new

    def _testIn(self, bbox: tuple or list, field: tuple or list):
        """
        :param bbox: (tuple or list) [x_min, y_min, x_max, y_max]
        :param field: (tuple or list) [x_min, y_min, x_max, y_max]
        :return: (bool)
        """
        borderL, borderR, borderU, borderD = 20, 20, 20, 20
        x1_bbox, y1_bbox, x2_bbox, y2_bbox = bbox
        x1_field, y1_field, x2_field, y2_field = field
        if x1_bbox + borderL >= x1_field and \
                y1_bbox + borderU >= y1_field and \
                x2_bbox - borderR <= x2_field and \
                y2_bbox - borderD <= y2_field:
            return True
        else:
            return False

    def _vsi(self, path_meta: str, id_vsi: int):
        """
        read annotation and responding meta file of a vsi file
        :param path_meta: the path of annotation file
        :return:
        """
        if not os.path.exists(path_meta):
            raise FileNotFoundError(f"{path_meta} not found.")

        basename = os.path.basename(path_meta)
        meta = parse_vsi_meta(path_meta=path_meta)
        vsi = {
            "id": id_vsi,
            "width": meta["size"][0],
            "height": meta["size"][1],
            "case_id": basename.split("-")[0],
            "file_name": re.sub("_meta.json", ".vsi", basename),
        }
        return vsi

    def _images(self):
        pass

    def _info(self):
        info = {
            "year": 2020,
            "version": "0.3",
            "description": "This is unstable 0.3 version of the GIST dataset.",
            "contributor": "Yichen Yang, Tao Yuan",
            "url": "",
            "date_created": self._get_time_stamp(),
        }
        return info

    def _license(self):
        license = {
            "id": 0,
            "name": "",
            "url": "",
        }
        return license

    def _categories(self):
        categories = []
        category_certain = {
            "id": 1,  # 从 1 开始
            "name": "mito",
            "supercategory": "certain",
        }
        categories.append(category_certain) if "certain" in self.classes_valid else None
        category_no = {
            "id": 2,
            "name": "non",
            "supercategory": "certain",
        }
        categories.append(category_no) if "NO" in self.classes_valid else None
        category_uncertain = {
            "id": 3,
            "name": "uncertain",
            "supercategory": "uncertain",
        }
        categories.append(category_uncertain) if "uncertain" in self.classes_valid else None

        return categories

    def _get_time_stamp(self):
        ct = time.time()
        local_time = time.localtime(ct)
        data_head = time.strftime("%Y-%m-%d %H:%M:%S", local_time)
        data_secs = (ct - int(ct)) * 10e5
        time_stamp = "%s.%06d" % (data_head, data_secs)
        return time_stamp

    def write(self, path_out: str, force: bool = True):
        if os.path.exists(path_out):
            if not force:
                raise FileExistsError(f"{path_out} has existed.")
        json.dump(self._base, open(path_out, "w"), indent=2)


if __name__ == '__main__':
    import bioformats
    import javabridge

    javabridge.start_vm(class_path=bioformats.JARS)  # I code javabridge here, and maybe it will be proven wrong

    er = Jsoner("/media/lansv/passport/GIST/", split="train",
                classes_valid=["certain", "NO", "uncertain"], write=True)
    er.write("/mnt/nvme/GIST/annotations/train_add.json", force=True)
    javabridge.kill_vm()  # I code javabridge here, and maybe it will be proven wrong TODO

    print("End.")
