import os, json, cv2, random
import torch
import numpy as np
from itertools import permutations
import dgl
import time
import torch.nn as nn
from datetime import datetime
import torch.nn.functional as F
from detectron2.engine import DefaultPredictor
from detectron2.engine import DefaultTrainer
from Mask_Rcnn import get_config, show_predictions, show_images, show_prediction
from utils import showPlot, from_mask_to_image, prepareData, load_glove_model, read_answers
from Graph_NN import GraphModel, train_graph_model, evaluate_graph_model,graphnn_predict
from Features_Extraction import train_conv_net, Model, evaluate_model, covnet_predict
from LSTM import train_lstm, read_lines, predict_encode, EncoderRNN, AttnDecoderRNN, decode_question
from Graph_Embeddings import GraphEmbeddings
import torch.nn as nn
import torch.nn.functional as F
import sys
from torch import optim
from utils import timeSince, showPlot, Lang, read_lines, tensorsFromPair, tensorFromSentence, prepareData, load_glove_model, tensorsFromPairAnswer

from read_data import get_clevr_dicts, returnClevr_metadata, get_clevr_samples, read_datafile, data_for_feature_extraction, get_clevr_geometric_data

def from_preds_to_embeddings(corpus_embedding, D):
    with open("/home/brenda/Documents/master/semestre3/independent_study/generate_graph_clean/output/scene_predictions.json") as f:
        scn = json.load(f)

    scene_embeddings = []
    for scene in scn:
        scene_name = scene["scene"].split("/")[-1]
        grp_embedding = GraphEmbeddings(5, scene_name)

        grp_embedding.obj_embedding(D, corpus_embedding, scene["shape_p"], scene["color_p"],  scene["size_p"], scene["material_p"])
        grp_embedding.edge_embedding(D["relation"], corpus_embedding, scene["edge0_p"],  scene["edge1_p"])
        scene_embeddings.append(grp_embedding)
    return scene_embeddings

def from_image_to_graph(show_img = False, n=3, height = 440, width = 520, device = "cuda"):
    dataset_dicts = read_datafile()
    clevr_metadata = returnClevr_metadata()
    print("data: loaded")

    cfg = get_config()
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set a custom testing threshold
    predictor = DefaultPredictor(cfg)
    print("mask rcnn: loaded")

    model_convnet = Model(pretrained=True, device= device)
    model_grapnet = GraphModel(pretrained=True, timestamp="gnn_model_20220314_123135", device= device)
    print("Conv model: loaded")

    shape = {
        0: "cube",
        1: "sphere",
        2: "cylinder",
    }

    indx = 0

    prediction_dicts = []

    for d in dataset_dicts:
        im = cv2.imread(d["file_name"])

        outputs = predictor(im)

        num_instances = len(outputs["instances"].pred_classes)
        print("number of objects: ", num_instances)

        if num_instances != 5:
            continue

        l = list(permutations(range(num_instances), 2))
        first_array = [x[0] for x in l]
        second_array = [x[1] for x in l]

        shapes = []
        for s in list(outputs["instances"].pred_classes):
            shapes.append(shape[s.item()])

        temp_color_mask = from_mask_to_image(im, outputs, height, width)

        temp_colored_img = np.zeros((height, width, 3))
        while len(temp_color_mask) < 5:
            temp_color_mask.append(temp_colored_img)

        pixel_features = torch.tensor(np.array(temp_color_mask), dtype=torch.float).to(device)
        temp = dgl.graph((first_array, second_array)).to(device)
        temp.ndata['feature'] = pixel_features

        colors, sizes, materials, color_prob, size_prob, material_prob = covnet_predict(model_convnet, pixel_features, device)
        edge0, edge_prob0, edge1, edge_prob1 = graphnn_predict(model_grapnet, temp, device)

        record_predictions = {}
        record_predictions["scene"] = d["file_name"]
        record_predictions["shape_p"] = outputs["instances"].probabilities.tolist()
        record_predictions["color_p"] = color_prob.tolist()
        record_predictions["size_p"] = size_prob.tolist()
        record_predictions["material_p"] = material_prob.tolist()
        record_predictions["edge0_p"] = edge_prob0.tolist()
        record_predictions["edge1_p"] = edge_prob1.tolist()

        prediction_dicts.append(record_predictions)

        #print("file: ", d["file_name"])
        #print("shapes: ", shapes)
        #print("colors: ", colors)
        #print("size: ", sizes)
        #print("material: ", materials)
        #print("edges: ", edge0)
        #print("edges: ", edge1)

        if show_img:
            show_prediction(im, clevr_metadata, d)

        #grp_embedding = GraphEmbeddings(5, first_array, second_array)

        #if corpus != None:
        #    grp_embedding.obj_embedding(D, corpus_embedding, outputs["instances"].probabilities, color_prob, size_prob, material_prob)
        #    grp_embedding.edge_embedding(D["relation"], corpus_embedding, edge_prob0, edge_prob1)
        print("ind: ", indx)
        indx += 1
        if indx == n:
            break

    with open("/home/brenda/Documents/master/semestre3/independent_study/generate_graph_clean/output/scene_predictions.json",
              "w") as f:
        json.dump(prediction_dicts, f)

    #return grp_embedding

def calculate_relevance_scores(r, R, graphEmbeddings, p_current):
    W_l = torch.eye(300)
    property_W = torch.stack([torch.eye(300) for _ in range(4)], dim=0)

    g0 = F.elu(torch.sum(torch.matmul(R[:4], torch.matmul(r[0], torch.matmul(property_W, torch.tensor(graphEmbeddings.embedded_concepts[0], dtype=torch.float32).transpose(2, 1)) ).transpose(1, 0)  ), dim=0))
    g1 = F.elu(torch.sum(torch.matmul(R[:4], torch.matmul(r[0], torch.matmul(property_W, torch.tensor(graphEmbeddings.embedded_concepts[1], dtype=torch.float32).transpose(2, 1)) ).transpose(1, 0) ), dim=0))
    g2 = F.elu(torch.sum(torch.matmul(R[:4], torch.matmul(r[0], torch.matmul(property_W, torch.tensor(graphEmbeddings.embedded_concepts[2], dtype=torch.float32).transpose(2, 1)) ).transpose(1, 0)  ), dim=0))
    g3 = F.elu(torch.sum(torch.matmul(R[:4], torch.matmul(r[0], torch.matmul(property_W, torch.tensor(graphEmbeddings.embedded_concepts[3], dtype=torch.float32).transpose(2, 1)) ).transpose(1, 0) ), dim=0))
    g4 = F.elu(torch.sum(torch.matmul(R[:4], torch.matmul(r[0], torch.matmul(property_W, torch.tensor(graphEmbeddings.embedded_concepts[4], dtype=torch.float32).transpose(2, 1)) ).transpose(1, 0) ), dim=0))

    gamma_s = torch.stack((g0, g1, g2, g3, g4))
    gamma_e = F.elu(torch.matmul(r[0], torch.matmul(W_l, torch.tensor(graphEmbeddings.embedded_relations, dtype=torch.float32).transpose(2, 1) )))

    p_s_next = F.softmax(gamma_s.transpose(1, 0), dim=1)

    sum_t = torch.zeros((5,1)) # 5x5 30x300
    for i in range(len(p_current)):
        for e in gamma_e[i:i+4]:
            sum_t[i] += p_current[i] * e[0]

    p_r_next = F.softmax(sum_t.transpose(1, 0), dim=1)
    p_next = R[4] * p_r_next + (1 - R[4]) * p_s_next

    return p_next

def load_vocabulary_embeddings():
    corpus = ["material", "shape", "color", "size", "relation"]
    d = {}
    d["shape"] = ["cube", "sphere", "cylinder"]
    d["material"] = ["metal", "rubber"]
    d["color"] = ["red", "gray", "blue", "yellow", "green", "purple", "cyan", "brown"]
    d["size"] = ["large", "small"]
    d["relation"] = ["front", "behind","left", "right"]

    corpus_embedding = {}
    glove_model = load_glove_model("/home/brenda/Documents/master/semestre3/independent_study/generate_graph_clean/output/glove.6B.300d.txt")

    for c in corpus:
        corpus_embedding[c] = glove_model[c].tolist()
        for item in d[c]:
            corpus_embedding[item] = glove_model[item].tolist()

    with open("/home/brenda/Documents/master/semestre3/independent_study/generate_graph_clean/output/vocabulary.json",
              "w") as f:
        json.dump(corpus_embedding, f)

    #return corpus, corpus_embedding, d

def load_vocabulary():
    corpus = ["material", "shape", "color", "size", "relation"]
    d = {}
    d["shape"] = ["cube", "sphere", "cylinder"]
    d["material"] = ["metal", "rubber"]
    d["color"] = ["red", "gray", "blue", "yellow", "green", "purple", "cyan", "brown"]
    d["size"] = ["large", "small"]
    d["relation"] = ["front", "behind", "left", "right"]

    with open(
            "/home/brenda/Documents/master/semestre3/independent_study/generate_graph_clean/output/vocabulary.json") as f:
        corpus_embedding = json.load(f)

    for key, value in corpus_embedding.items():
        corpus_embedding[key] = np.array(value)

    return corpus, corpus_embedding, d

def find_instruction_type(r, corpus, corpus_embedding):
    temp = []
    for c in corpus:
        embedded_temp = torch.tensor([corpus_embedding[c]], dtype=torch.float32)
        dot_product = torch.matmul(r[0], embedded_temp.transpose(1,0))
        temp.append(dot_product)
    R = F.softmax(torch.tensor(temp), dim=0)
    return R

def evalute_neural_state_machine(DEVICE="cuda", num_graphs=None):
    path = "/home/brenda/Documents/master/semestre3/independent_study/generate_graph_clean/output/"
    name = "default"
    # DEVICE = "cpu"
    hidden_size = 300
    max_iterations = 10
    SOS_token = 0
    EOS_token = 1
    p_initial = np.array([1 / 5, 1 / 5, 1 / 5, 1 / 5, 1 / 5])

    # Loading vocabulary
    corpus, corpus_embedding, d = load_vocabulary()

    # reading questions and indexes of corresponding images (circa 1500 q for 130 images)
    questions_lines, images = read_lines()

    print("images ", len(images))
    # preparing data from questions
    input_lang, questions, indices = prepareData('questions', questions_lines)
    pair = random.choice(questions)
    print("Questions loaded: ", pair)
    print(input_lang.index2word)

    # reading answers
    answer_lines = read_answers()
    answer_lines = list(map(answer_lines.__getitem__, indices))  # filtering answers using the indices of the questions
    images = list(map(images.__getitem__, indices))  # filtering images indexes using the indices of the questions
    print("images ", len(images))

    answer_lang, answers, _ = prepareData('answers', answer_lines, False)
    training_answers = [tensorsFromPairAnswer(answers[i], answer_lang) for i in range(len(answers))]
    print(answer_lang.index2word)

    # Loading scenes probabilities
    scene_embbs = from_preds_to_embeddings(corpus_embedding, d)
    print(type(scene_embbs))
    # scene_embbs = [scn for scn in scene_embbs if scn["index"] in images] #filtering scenes using the indices of the images
    print(len(scene_embbs), " scenes loaded!")

    scenes_dic = {}
    # print(images)
    for scn, z in zip(scene_embbs, range(len(scene_embbs))):
        if scn.index in images:
            scenes_dic[scn.index] = scn

    print(len(scenes_dic.keys()))

    p_i = p_initial
    # Classifier hyper-parameters
    classifier = Classifier(300, model="nms_20220328_140030", pretrained=True).to(DEVICE)
    classifier.eval()

    # Loading encoder and decoder
    encoder = EncoderRNN(hidden_size, input_size=input_lang.n_words)  # .to(DEVICE)
    decoder = AttnDecoderRNN(hidden_size, input_lang.n_words, dropout_p=0.1)  # .to(DEVICE)

    encoder.load_state_dict(torch.load(path + "encoder_{}".format(name)))
    decoder.load_state_dict(torch.load(path + "decoder_{}".format(name)))

    indx = 0
    acc = 0

    bad_examples = 0
    for q, img in zip(questions, images):
        if img not in scenes_dic.keys():
            indx += 1
            continue
        s = scenes_dic[img]

        encoded_hidden, encoder_outputs = predict_encode(encoder, input_lang, q[0])  # questions[idx_g][0])
        decoder_hidden = encoded_hidden
        decoder_input = torch.tensor([[SOS_token]])


        for index in range(max_iterations):
            decoded_words, decoder_input, decoder_hidden = decode_question(decoder, decoder_input, input_lang,
                                                                               decoder_hidden, encoder_outputs)

            R = find_instruction_type(decoder_hidden, corpus, corpus_embedding)
            p_i = calculate_relevance_scores(decoder_hidden, R, s, p_i)[0]

            if decoder_input == EOS_token:
                with torch.no_grad():
                    m0 = torch.mul(p_i, torch.sum(
                            torch.matmul(R[:4], torch.tensor(s.embedded_concepts[0], dtype=torch.float32).transpose(1, 0))))
                    m1 = torch.mul(p_i, torch.sum(
                            torch.matmul(R[:4], torch.tensor(s.embedded_concepts[1], dtype=torch.float32).transpose(1, 0))))
                    m2 = torch.mul(p_i, torch.sum(
                            torch.matmul(R[:4], torch.tensor(s.embedded_concepts[2], dtype=torch.float32).transpose(1, 0))))
                    m3 = torch.mul(p_i, torch.sum(
                            torch.matmul(R[:4], torch.tensor(s.embedded_concepts[3], dtype=torch.float32).transpose(1, 0))))
                    m4 = torch.mul(p_i, torch.sum(
                            torch.matmul(R[:4], torch.tensor(s.embedded_concepts[4], dtype=torch.float32).transpose(1, 0))))

                    m = torch.sum(torch.stack((m0, m1, m2, m3, m4)), dim=0)
                    combine = torch.cat((encoder_outputs[0], m)).to(DEVICE)

                    pred_answers = classifier(combine)
                    _, indices = torch.max(pred_answers, dim=0)

                    acc += 1 if indices.to("cpu") == training_answers[indx] else 0
                    if indices.to("cpu") == training_answers[indx]:
                        print("good " ,img, " " , q[0], ": ", training_answers[indx])
                    elif bad_examples <3:
                        print("bad " , img, " ", q[0], ": ", training_answers[indx], " ", indices.to("cpu"))
                        bad_examples+=1

                break
        indx += 1
        if indx == 100:
            break
    print("indx ", indx)
    print("Acc: ", acc)
    print("Accuracy: ", acc/indx)
    #return acc/len(scene_embbs)

def neural_state_machine(n_iters = 10, print_every=100, plot_every=100, learning_rate=0.01, DEVICE="cuda", num_graphs=None, pretrained= False):
    path = "/home/brenda/Documents/master/semestre3/independent_study/generate_graph_clean/output/"
    name = "default"
    #DEVICE = "cpu"
    hidden_size = 300
    max_iterations = 10
    SOS_token = 0
    EOS_token = 1
    p_initial = np.array([1 / 5, 1 / 5, 1 / 5, 1 / 5, 1 / 5])

    #Loading vocabulary
    corpus, corpus_embedding, d = load_vocabulary()

    #reading questions and indexes of corresponding images (circa 1500 q for 130 images)
    questions_lines, images = read_lines()

    print("images ", len(images))
    #preparing data from questions
    input_lang, questions, indices = prepareData('questions', questions_lines)
    pair = random.choice(questions)
    print("Questions loaded: ", pair)

    #reading answers
    answer_lines = read_answers()
    answer_lines = list(map(answer_lines.__getitem__, indices)) #filtering answers using the indices of the questions
    images = list(map(images.__getitem__, indices)) #filtering images indexes using the indices of the questions
    print("images ", len(images))

    answer_lang, answers, _ = prepareData('answers', answer_lines, False)
    training_answers = [tensorsFromPairAnswer(answers[i], answer_lang) for i in range(len(answers))]

    #Loading scenes probabilities
    scene_embbs = from_preds_to_embeddings(corpus_embedding, d)
    print(type(scene_embbs))
    #scene_embbs = [scn for scn in scene_embbs if scn["index"] in images] #filtering scenes using the indices of the images
    print(len(scene_embbs), " scenes loaded!")

    scenes_dic = {}
    #print(images)
    for scn, z in zip(scene_embbs, range(len(scene_embbs))):
        if scn.index in images:
            scenes_dic[scn.index] = scn

    p_i = p_initial
    loss = 0

    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    plot_losses = []
    start = time.time()

    #Classifier hyper-parameters
    if pretrained:
        classifier = Classifier(300, model="nms_20220327_194255", pretrained=True).to(DEVICE)
    else:
        classifier = Classifier(300).to(DEVICE)  # device
    criterion = nn.NLLLoss()
    #optimizer = optim.Adam(classifier.parameters(), lr=learning_rate) #, lr=3e-4, weight_decay=0.0001
    optimizer = torch.optim.Adam(classifier.parameters(), lr=3e-4, weight_decay=0.0001)
    classifier.train()

    #Loading encoder and decoder
    encoder = EncoderRNN(hidden_size, input_size=input_lang.n_words)  # .to(DEVICE)
    decoder = AttnDecoderRNN(hidden_size, input_lang.n_words, dropout_p=0.1)  # .to(DEVICE)

    encoder.load_state_dict(torch.load(path + "encoder_{}".format(name)))
    encoder.eval()
    decoder.load_state_dict(torch.load(path + "decoder_{}".format(name)))
    decoder.eval()

    t = datetime.now().strftime("%Y%m%d_%H%M%S")
    for iter in range(1, n_iters + 1):
        epoch_loss = 0
        idx_g = 0
        #para cada pregunta, imagen
        #for s in scene_embbs:
        for q, img in zip(questions, images):
            if img not in scenes_dic.keys():
                idx_g += 1
                continue
            s = scenes_dic[img]
            encoded_hidden, encoder_outputs = predict_encode(encoder, input_lang, questions[idx_g][0]) #questions[idx_g][0])
            decoder_hidden = encoded_hidden
            decoder_input = torch.tensor([[SOS_token]])

            for index in range(max_iterations):
                decoded_words, decoder_input, decoder_hidden = decode_question(decoder, decoder_input, input_lang, decoder_hidden, encoder_outputs)

                R = find_instruction_type(decoder_hidden, corpus, corpus_embedding)
                p_i = calculate_relevance_scores(decoder_hidden, R, s, p_i)[0]

                #print(decoded_words)
                if decoder_input == EOS_token or index==max_iterations-1:
                    with torch.no_grad():
                        m0 = torch.mul(p_i, torch.sum(torch.matmul(R[:4], torch.tensor(s.embedded_concepts[0],dtype=torch.float32).transpose(1,0))))
                        m1 = torch.mul(p_i, torch.sum(torch.matmul(R[:4], torch.tensor(s.embedded_concepts[1], dtype=torch.float32).transpose(1, 0))))
                        m2 = torch.mul(p_i, torch.sum(torch.matmul(R[:4], torch.tensor(s.embedded_concepts[2], dtype=torch.float32).transpose(1, 0))))
                        m3 = torch.mul(p_i, torch.sum(torch.matmul(R[:4], torch.tensor(s.embedded_concepts[3], dtype=torch.float32).transpose(1, 0))))
                        m4 = torch.mul(p_i, torch.sum(torch.matmul(R[:4], torch.tensor(s.embedded_concepts[4], dtype=torch.float32).transpose(1, 0))))

                        m = torch.sum(torch.stack((m0, m1, m2, m3, m4)), dim=0)
                        combine = torch.cat((encoder_outputs[0], m)).to(DEVICE)

                    pred_answers = classifier(combine)
                    loss = criterion(pred_answers, training_answers[idx_g][0].to(DEVICE))
                    break

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss)
            idx_g += 1

        print_loss_total += (epoch_loss / idx_g)
        plot_loss_total += (epoch_loss / idx_g)

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                                     iter, iter / n_iters * 100, print_loss_avg))

            torch.save(classifier.state_dict(), path + "nms_{}".format(t))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

        print("Iter: ", iter, " graphs: ", idx_g)
    showPlot(plot_losses)

class Classifier(nn.Module):
    def __init__(self, output, pretrained= False, model = None, device="cuda"):
        super().__init__()
        self.l1 = nn.Linear(305, 1000)
        self.l2 = nn.Linear(1000, output)

        if pretrained:
            PATH_OUTPUT_MODEL = "/home/brenda/Documents/master/semestre3/independent_study/generate_graph_clean/output/{}".format(model)
            self.load_state_dict(torch.load(PATH_OUTPUT_MODEL, map_location=device))

    def forward(self, x):
        y = self.l1(x)
        y = self.l2(y)
        c = F.log_softmax(y, dim=0)
        return c