import os, json
import numpy as np
from detectron2.structures import BoxMode
from detectron2.data import MetadataCatalog, DatasetCatalog

def get_clevr_dicts(img_dir, scenes_dir, mode, height= 620, width= 840):  # 240, 320

    shape = {
        "cube" : 0,
        "sphere" : 1,
        "cylinder": 2,
    }

    ################### REMOVE

    material = {
        "metal": 0,
        "rubber": 1,
    }

    color = {
        "gray": 0,
        "red": 1,
        "blue": 2,
        "green": 3,
        "brown": 4,
        "purple": 5,
        "cyan": 6,
        "yellow": 7,
    }

    ################### REMOVE

    json_file = scenes_dir.format(mode)

    directory = os.fsencode(json_file)

    dataset_dicts = []

    idx = 0
    for file in os.listdir(directory):

        file = os.fsdecode(file)
        with open(json_file +file) as f:
            scn = json.load(f)

        record = {}
        filename = os.path.join(img_dir.format(mode ) +scn["image_filename"])

        record["file_name"] = filename
        record["image_id"] = scn["image_index"]
        record["height"] = height
        record["width"] = width

        items =  scn["objects"]
        objs = []
        for it in items:
            shape_code = shape[it["shape"]]
            material_code = material[it["material"]]  # REMOVE
            color_code = color[it["color"]]  # REMOVE

            px = [x[0] + 0.5 for x in it["segmentation"]]
            py = [x[1] + 0.5 for x in it["segmentation"]]
            poly = [(y, x) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            obj = {
                "bbox": [np.min(py), np.min(px), np.max(py), np.max(px) ,],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": shape_code,
                "material_id": material_code,  # REMOVE
                "color_id": color_code,  # REMOVE
                "iscrowd": 0
            }

            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)

        print("Image...", idx)
        idx += 1
    return dataset_dicts

def returnClevr_metadata(img_dir, scenes_dir):
    for d in ["train", "test"]:
        DatasetCatalog.register("clevr__" + d, lambda d=d: get_clevr_dicts(img_dir, scenes_dir, d))
        MetadataCatalog.get("clevr__" + d).set(thing_classes=["cube", "sphere", "cilinder"])
    return MetadataCatalog.get("clevr__train")