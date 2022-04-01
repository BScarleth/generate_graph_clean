import os, json, cv2, random
import torch
import numpy as np
from itertools import permutations
import dgl
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from detectron2.engine import DefaultPredictor
from detectron2.engine import DefaultTrainer
from Mask_Rcnn import get_config, show_predictions, show_images, show_prediction
from utils import showPlot, from_mask_to_image, prepareData, load_glove_model
from Graph_NN import GraphModel, train_graph_model, evaluate_graph_model,graphnn_predict
from Features_Extraction import train_conv_net, Model, evaluate_model, covnet_predict
from LSTM import train_lstm, read_lines, predict_encode, EncoderRNN, AttnDecoderRNN, decode_question
from Graph_Embeddings import GraphEmbeddings
from NSM import neural_state_machine, from_image_to_graph, load_vocabulary, from_preds_to_embeddings,load_vocabulary_embeddings, evalute_neural_state_machine

from read_data import FeatureExtractionDataset, get_clevr_dicts, returnClevr_metadata, get_clevr_samples, read_datafile, data_for_feature_extraction, get_clevr_geometric_data

def train_lstmnn(iters= 8000, pretrained=True):
    path = "/home/brenda/Documents/master/semestre3/independent_study/generate_graph_clean/output/"
    lines, _ = read_lines()
    train_lstm(lines, "default", path, iters, True) #5000

def train_mask_rcnn(img_dir, scenes_dir):
    clevr_metadata = returnClevr_metadata()
    dataset_dicts = get_clevr_samples(img_dir, scenes_dir, "train", 3)

    cfg = get_config()
    print("config ready")
    trainer = DefaultTrainer(cfg)

    trainer.resume_or_load(resume=False)
    trainer.train()

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set a custom testing threshold

    predictor = DefaultPredictor(cfg)
    show_predictions(dataset_dicts, clevr_metadata, predictor)

def test_mask_rcnn(img_dir, scenes_dir):
    clevr_metadata = returnClevr_metadata()
    # dataset_dicts = get_clevr_dicts(img_dir, scenes_dir, "train", 150)
    dataset_dicts = get_clevr_samples(img_dir, scenes_dir, "train", 3)

    cfg = get_config()
    print("config ready")

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set a custom testing threshold

    predictor = DefaultPredictor(cfg)
    show_predictions(dataset_dicts, clevr_metadata, predictor)

def train_graphnn(img_dir, scenes_dir, img_dir_t, scenes_dir_t):
    print("loading data...")
    data_graphs = get_clevr_geometric_data(img_dir, scenes_dir, "train", 60)
    print("data ready...")
    #model = GraphModel(pretrained=True, timestamp="gnn_model_20220314_123135")
    model = GraphModel()

    plot_losses, plot_acc = train_graph_model(model, data_graphs, 10)
    print("evaluation started...")
    data_graphs = get_clevr_geometric_data(img_dir_t, scenes_dir_t, "test", 50)
    evaluate_graph_model(data_graphs, model)

def train_conv_net_loader(img_dir, scenes_dir, img_dir_v, scenes_dir_v, img_dir_t, scenes_dir_t):
    print("loading data...")
    #dataset_dicts = data_for_feature_extraction(img_dir, scenes_dir, "train", 130, device="cpu")
    training_dataset = FeatureExtractionDataset(img_dir, scenes_dir, "train", device="cuda")
    train_dataloader = DataLoader(training_dataset, batch_size=5, shuffle=True)

    #validation_dataset = FeatureExtractionDataset(img_dir_v, scenes_dir_v, "val", device="cuda")
    #validation_dataloader = DataLoader(validation_dataset, batch_size=5, shuffle=True)

    print("data ready...")
    model = Model(device="cuda")

    #print(len(train_dataloader))
    #print(next(iter(train_dataloader)))

    plot_losses, plot_acc = train_conv_net(model, train_dataloader, 40, device="cuda") #40
    print("evaluation started...")

    test_dataset = FeatureExtractionDataset(img_dir_t, scenes_dir_t, "test", device="cpu")
    test_dataloader = DataLoader(test_dataset, batch_size=5, shuffle=True)

    #dataset_dicts = data_for_feature_extraction(img_dir_t, scenes_dir_t, "train", 20, device="cpu")
    evaluate_model(model, test_dataloader, device="cpu")
    #showPlot(plot_losses)
    #showPlot(plot_acc)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #/home/brenda/Documents/master/semestre3/independent_study/CLEVR_dataset-20220227T212835Z-002/CLEVR_dataset/output
    img_dir = "/home/brenda/Documents/master/semestre3/independent_study/clevr-dataset-generation-mask/output/images/"
    scenes_dir = "/home/brenda/Documents/master/semestre3/independent_study/clevr-dataset-generation-mask/output/scenes/"

    img_dir_validation = "/home/brenda/Documents/master/semestre3/independent_study/clevr-dataset-generation-mask/output/images_validation/"
    scenes_dir_validation = "/home/brenda/Documents/master/semestre3/independent_study/clevr-dataset-generation-mask/output/scenes_validation/"

    img_dir_test = "/home/brenda/Documents/master/semestre3/independent_study/clevr-dataset-generation-mask/output/images_test/"
    scenes_dir_test = "/home/brenda/Documents/master/semestre3/independent_study/clevr-dataset-generation-mask/output/scenes_test/"

    print("Is CUDA available?: ",torch.cuda.is_available())

    #train_graphnn(img_dir, scenes_dir, img_dir_test,  scenes_dir_test)

    """
    1-Load the dataset for training:
        n = 150 for mask-rcnn
        n = 50 for conv-net
        dataset_dicts = get_clevr_dicts(img_dir, scenes_dir, "train")
    2-Train Mask-rcnn:
        train_mask_rcnn(img_dir, scenes_dir)
        2.1- Test Mask-rcnn:
            test_mask_rcnn(img_dir_tes, scenes_dir_test)
    3-Train and evaluate Conv-net:
        train_conv_net_loader(img_dir, scenes_dir, img_dir_test,  scenes_dir_test)
    4-From image to graph:
        from_image_to_graph(img_dir, scenes_dir)
    """

    #from_image_to_graph(n= 130)
    #neural_state_machine(100, 5, 5, DEVICE="cuda") #,  pretrained=True)
    #evalute_neural_state_machine()
    #questions_lines, images = read_lines()
    #input_lang, questions, indices = prepareData('questions', questions_lines)
    #print(questions)
    #train_lstmnn(iters=550, pretrained=True)
    #train_conv_net_loader(img_dir, scenes_dir, img_dir_validation, scenes_dir_validation, img_dir_test, scenes_dir_test)
    #training_dataset = FeatureExtractionDataset(img_dir, scenes_dir, "train", device="cuda")
    #train_dataloader = DataLoader(training_dataset, batch_size=5, shuffle=False)

    """
    j = 0
    for d in train_dataloader:  # range(data.__len__):
        #print("len: ", len(d))
        for i in range(len(d)):
            j+=1
            print("iter ", j, " ", d["image_id"][i])
    """
    train_graphnn(img_dir, scenes_dir, img_dir_test, scenes_dir_test)



