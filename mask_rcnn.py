import random, cv2
from detectron2.utils.visualizer import Visualizer
import matplotlib.pyplot as plt
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import ColorMode

def show_images(dataset_dicts, clevr_metadata, n=3):
    for d in random.sample(dataset_dicts, n):
        img = cv2.imread(d["file_name"])

        print(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=clevr_metadata, scale=0.5)
        out = visualizer.draw_dataset_dict(d)
        plt.imshow(out.get_image()[:, :, ::-1])

def show_predictions(dataset_dicts, clevr_metadata, predictor):
    image_example = None
    for d in random.sample(dataset_dicts, 3):
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)

        image_example = outputs
        v = Visualizer(im[:, :, ::-1],
                       metadata=clevr_metadata,
                       scale=0.5,
                       instance_mode=ColorMode.IMAGE_BW
                       )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        plt.imshow(out.get_image()[:, :, ::-1])

def get_config():
    cfg = get_cfg()
    # mask_rcnn_R_101_FPN_3x.yaml
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
    cfg.MODEL.DEVICE='cpu'
    cfg.DATASETS.TRAIN = ("clevr__train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")  # Let training initialize from model zoo NOTE: FPN backbone with restnet 101.
    cfg.SOLVER.IMS_PER_BATCH = 8  # 2
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR 0.002 00025
    cfg.SOLVER.MAX_ITER = 1  # 500 iterations NOTE: I have to increase this  r 30,000
    cfg.SOLVER.STEPS = []  # do not decay learning rate 4k img

    print(cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST)
    print(cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST)
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # # NOTE: this config means the number of classes,
    cfg.MODEL.ROI_HEADS.NUM_MATERIALS = 2  # # NOTE: this config means the number of materials
    cfg.MODEL.ROI_HEADS.NUM_COLORS = 8  # # NOTE: this config means the number of colors

    cfg.OUTPUT_DIR = "/home/brenda/PycharmProjects/generate_graph_clean/output_model/"
    print(cfg.OUTPUT_DIR)
    return cfg
    # os.makedirs("/content/drive/MyDrive/Mask-rcnn-model"+cfg.OUTPUT_DIR, exist_ok=True)
