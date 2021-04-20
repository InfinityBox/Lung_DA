# public
import os
import json
import pickle
import hashlib
from typing import List

color = int("0x894276", 16) - int("0x1000000", 16)  # RGB hex value
TAB_CLASSES = ["infer", "NO", "uncertain"]
TAB_COLOR = {"infer": color, "NO": -980755, "uncertain": -179} # origin certain -16711936


def merge_annos(dir_source: str, dir_target: str):

    assert os.path.exists(dir_source), \
        "PathError: {dir} is not a directory".format(dir=dir_source)
    assert os.path.exists(dir_target), \
        "PathError: {dir} is not a directory".format(dir=dir_target)

    repo = {}
    for pkl in os.listdir(dir_source):
        title_patch, code = pkl.split(".")[0].split("_")
        if title_patch not in repo:
            repo[title_patch] = []
        with open(os.path.join(dir_source, pkl), "rb") as file:
            content = file.read()
            if code != hash2(content):
                print("{} breaks".format(pkl))
                continue
            annos_patch = pickle.loads(content)
            for bbox, category in \
                    zip(annos_patch["bboxes"], annos_patch["classes"]):
                if category in [0]:
                    x, y = annos_patch["location"]
                    anno_patch = {"type":"Rect", "class": category,
                                  "coordinates": [[bbox[0]+x, bbox[1]+y],
                                                  [bbox[0]+x, bbox[3]+y],
                                                  [bbox[2]+x, bbox[3]+y],
                                                  [bbox[2]+x, bbox[1]+y],
                                                  [bbox[0]+x, bbox[1]+y]]}
                    repo[title_patch].append(anno_patch)
    for key, values in repo.items():
        save_annos(os.path.join(dir_target, key + ".json"), values)


def load_anno(path_json: str):
    assert os.path.exists(path_json), \
        "PathError: path_json doesn't exist, {}".format(path_json)
    anno_list = []
    with open(path_json, "r") as file:
        content = file.read()
        annos = json.loads(content)
        for anno in annos:
            anno_list.append(json.loads(anno))

    return anno_list


def save_json(path_json: str, data):
    assert not os.path.exists(path_json), \
        "PathError: path_json doesn't exist, {}".format(path_json)
    with open(path_json, "w") as file:
        content = json.dumps(data, indent=2)
        file.write(content)


def construct_anno(anno_origin: dict):
    """
    Note that Rect and Eclipse all are stored by "Polygon",
    the color code in json is stored by complement

    :param anno_origin: {"type":"Rect", "class": 0,
        "coordinates": [[0,0], [100,0], [100,100], [0,100], [0,0]]}
    :return: dict
    """
    anno = dict()
    anno["type"] = "Feature"
    anno["id"] = "PathAnnotationObject"
    anno["geometry"] = {"type": "Polygon", "coordinates": [anno_origin["coordinates"]]}
    classification = {"name": TAB_CLASSES[anno_origin["class"]],
                      "colorRGB": TAB_COLOR[TAB_CLASSES[anno_origin["class"]]]}
    anno["properties"] = {"classification": classification,
                          "isLocked": False,
                          "measurements": []}
    return anno


def save_annos(path_json: str, annos: List[dict]):

    annostr_dict = {}
    for idx, anno in enumerate(annos):
        annostr_dict[idx] = json.dumps(construct_anno(anno), indent=2)
    save_json(path_json, annostr_dict)


def hash2(content: bytes):
    md5 = hashlib.md5()
    md5.update(content)
    code = md5.hexdigest()
    return code


if __name__ == '__main__':

    ############################# Test save_annos #############################
    # path = "I:\\GIST\\test_save.json"
    # anno_dict = save_annos(path, [{"type":"Rect", "class": 0,
    #                               "coordinates": [[0,0], [100,0], [100,100], [0,100], [0,0]]
    #                                },
    #                               {"type": "Polygon", "class": 1,
    #                                "coordinates": [[500, 500], [600, 500], [600, 600],
    #                                                 [500, 600], [500, 500]]
    #                                }
    #                               ])
    ############################# Test save_annos #############################

    ############################# Test merge_annos #############################
    path_source = "/home/lansv/Temp/mito"
    path_target = "/home/lansv/Temp/json"
    merge_annos(path_source, path_target)
    print("End.")