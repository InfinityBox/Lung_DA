# public
import os
import cv2
import json
import numpy as np


def parse_vsi_anno(path_anno: str):

    # read from files
    gson = {}
    with open(path_anno, "r") as file:
        content = file.read()
        gson.update(json.loads(content))

    # parsing
    annos = []
    if len(gson) > 0:
        for idx, elem in gson.items():
            if elem["geometry"]["type"] == "Polygon":  # Polygon only
                tag = elem["properties"]["classification"]["name"]
                contour = np.array(elem["geometry"]["coordinates"][0])
                bbox = cv2.boundingRect(contour.astype(np.int64))
                anno = {"tag": tag,
                        "contour": contour,
                        "bbox": bbox}
                annos.append(anno)
            else:  # we don't consider other annotation type
                pass
    return annos


def parse_vsi_meta(path_meta: str):

    # read from files
    gson = {}
    with open(path_meta, "r") as file:
        content = file.read()
        gson.update(json.loads(content))

    # parsing
    meta = {"size": [gson["width"], gson["height"]]}
    return meta

if __name__ == '__main__':

    path_anno = "I:\\test\\155-477366-1.vsi - 40x_annotation.json"
    anno = parse_vsi_anno(path_anno)

    path_meta = "I:\\test\\155-477366-1.vsi - 40x_meta.json"
    meta = parse_vsi_meta(path_meta)

    print("End.")