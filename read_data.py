import os, json
import numpy as np
import torch
from detectron2.structures import BoxMode
from detectron2.data import MetadataCatalog, DatasetCatalog
import random, cv2
import dgl
from torch.utils.data import Dataset


class FeatureExtractionDataset(Dataset):
    def __init__(self, img_dir, scenes_dir, mode, height= 440, width= 520, device="cuda"):
        self.img_dir = img_dir.format(mode)
        self.scenes_dir = scenes_dir.format(mode)
        self.img_labels = sorted(os.listdir(self.img_dir))
        self.scenes_labels = sorted(os.listdir(self.scenes_dir))
        self.height = height
        self.width = width
        self.device = device

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        material = {
            "metal": 0,
            "rubber": 1,
        }

        size = {
            "small": 0,
            "large": 1,
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

        img_path = os.path.join(self.img_dir, self.img_labels[idx])
        scene_path = os.path.join(self.scenes_dir, self.scenes_labels[idx])
        img = cv2.imread(img_path)

        with open(scene_path ) as f:
            scn = json.load(f)

        record = {}
        record["image_id"] = scn["image_index"]

        items = scn["objects"]

        materials = []
        colors = []
        sizes = []
        temp_color_mask = []

        for it in items:
            materials.append(material[it["material"]])
            colors.append(color[it["color"]])
            sizes.append(size[it["size"]])
            temp_colored_img = np.zeros((self.height, self.width, 3))
            for pair in it["pixel_mask"]:
                temp_colored_img[pair[0], pair[1]] = img[pair[0], pair[1]]
            temp_color_mask.append(temp_colored_img)

        record["pixel_mask"] = torch.tensor(np.array(temp_color_mask), dtype=torch.float).to(self.device)
        record["materials"] = torch.tensor(np.array(materials), dtype=torch.long).to(self.device)
        record["colors"] = torch.tensor(np.array(colors), dtype=torch.long).to(self.device)
        record["sizes"] = torch.tensor(np.array(sizes), dtype=torch.long).to(self.device)

        return record

def get_clevr_dicts(img_dir, scenes_dir, mode, n=150, height= 440, width= 520):
    shape = {
        "cube" : 0,
        "sphere" : 1,
        "cylinder": 2,
    }

    json_file = scenes_dir.format(mode)
    directory = os.fsencode(json_file)
    dataset_dicts = []

    idx = 0
    for file in os.listdir(directory):

        file = os.fsdecode(file)
        with open(json_file +file) as f:
            scn = json.load(f)

        record = {}
        filename = os.path.join(img_dir.format(mode) +scn["image_filename"])

        record["file_name"] = filename
        record["image_id"] = scn["image_index"]
        record["height"] = height
        record["width"] = width

        items =  scn["objects"]
        objs = []
        for it in items:
            shape_code = shape[it["shape"]]

            px = [x[0] + 0.5 for x in it["segmentation"]]
            py = [x[1] + 0.5 for x in it["segmentation"]]
            poly = [(y, x) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            obj = {
                "bbox": [np.min(py), np.min(px), np.max(py), np.max(px) ,],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": shape_code,
                "iscrowd": 0
            }

            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)

        print("Image...", idx)
        idx += 1
        if idx == n:
            break

    with open("/home/brenda/Documents/master/semestre3/independent_study/generate_graph_clean/output/data.json", "w") as f:
        json.dump(dataset_dicts, f)
    return dataset_dicts

def get_clevr_samples(img_dir, scenes_dir, mode, n=3, height= 440, width= 520):  # 240, 320

    shape = {
        "cube" : 0,
        "sphere" : 1,
        "cylinder": 2,
    }

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

            px = [x[0] + 0.5 for x in it["segmentation"]]
            py = [x[1] + 0.5 for x in it["segmentation"]]
            poly = [(y, x) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            obj = {
                "bbox": [np.min(py), np.min(px), np.max(py), np.max(px) ,],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": shape_code,
                "iscrowd": 0
            }

            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)

        print("Image...", idx)
        idx += 1

        if idx == n:
            break
    return dataset_dicts

def get_clevr_geometric_data(img_dir, scenes_dir, mode, n=3, height= 440, width= 520):
    json_file = scenes_dir.format(mode)
    directory = os.fsencode(json_file)

    idx = 0
    data_list = []
    for file in os.listdir(directory):

        file = os.fsdecode(file)
        with open(json_file + file) as f:
            scn = json.load(f)

        relations_dictionary = scn["relationships_modified"]
        filename = os.path.join(img_dir.format(mode) + scn["image_filename"])

        img = cv2.imread(filename)

        first_array = [int(x.split("-")[0]) for x in list(relations_dictionary.keys())]
        second_array = [int(x.split("-")[1]) for x in list(relations_dictionary.keys())]
        features = [x for x in list(relations_dictionary.values())]

        #print(first_array)

        items = scn["objects"]
        temp_color_mask = []
        for it in items:
            temp_colored_img = np.zeros((height, width, 3))
            for pair in it["segmentation"]:
                temp_colored_img[pair[0], pair[1]] = img[pair[0], pair[1]]
            temp_color_mask.append(temp_colored_img)

        temp = dgl.graph((first_array, second_array)).to("cuda")
        temp.ndata['feature'] = torch.tensor(np.array(temp_color_mask), dtype=torch.float).to("cuda")
        temp.edata['label'] = torch.tensor(np.array(features), dtype=torch.long).to("cuda")

        data_list.append(temp)#.to("cuda"))

        print("Image...", idx)
        idx += 1

        if idx == n:
            break
    return data_list

def read_datafile():
    with open("/home/brenda/Documents/master/semestre3/independent_study/generate_graph_clean/output/data.json") as f:
        scn = json.load(f)
    return scn

def returnClevr_metadata():
    for d in ["train", "test"]:
        DatasetCatalog.register("clevr__" + d, lambda d=d: read_datafile())
        MetadataCatalog.get("clevr__" + d).set(thing_classes=["cube", "sphere", "cilinder"])
    return MetadataCatalog.get("clevr__train")
