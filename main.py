# import some common libraries
import numpy as np
import os, json, cv2, random
import torch
import torchvision
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import sys

from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.structures import BoxMode
from detectron2.engine import DefaultTrainer
from mask_rcnn import get_config, show_predictions
from detectron2.utils.visualizer import ColorMode

from read_data import get_clevr_dicts, returnClevr_metadata

def train_mask_rcnn(img_dir, scenes_dir):
    clevr_metadata = returnClevr_metadata(img_dir, scenes_dir)
    dataset_dicts = get_clevr_dicts(img_dir, scenes_dir, "train")

    cfg = get_config()
    print("config ready")
    trainer = DefaultTrainer(cfg)

    trainer.resume_or_load(resume=False)
    trainer.train()

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set a custom testing threshold

    predictor = DefaultPredictor(cfg)
    train_mask_rcnn(img_dir, scenes_dir)
    show_predictions(dataset_dicts, clevr_metadata, predictor)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    img_dir = "/home/brenda/Escritorio/Dataset/CLEVR_dataset/CLEVR_dataset/output/{}/images/"
    scenes_dir = "/home/brenda/Escritorio/Dataset/CLEVR_dataset/CLEVR_dataset/output/{}/scenes/"
    train_mask_rcnn(img_dir, scenes_dir)


