import time
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import json, unicodedata, torch
import re

DEVICE = "cpu"

def readLangs(lang1, lines):
    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in str(l).split('\t')] for l in lines] #here ?
    input_lang = Lang(lang1)
    return input_lang, pairs

def readLangsAnswers(lang1, lines):
    # Split every line into pairs and normalize
    pairs = [[s for s in str(l).split('\t')] for l in lines] #here ?
    input_lang = Lang(lang1, extra_tokens=False)
    return input_lang, pairs

def filterPair(p, max_length=15):
    return len(p[0].split(' ')) < max_length

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def filterPairsQuestion(pairs):
    s = range(len(pairs))
    return [pair for pair in pairs if filterPair(pair)],  [p for pair, p in zip(pairs, s) if filterPair(pair)],

def prepareData(lang1, lines, questions = True):
    indices = []

    if questions:
        input_lang, pairs = readLangs(lang1, lines)
        pairs, indices = filterPairsQuestion(pairs)
    else:
        input_lang, pairs,  = readLangsAnswers(lang1, lines)
        pairs = filterPairs(pairs)

    for pair in pairs:
        input_lang.addSentence(pair[0])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    return input_lang, pairs, indices

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def tensorFromSentence(lang, sentence, EOS_token = 1):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=DEVICE).view(-1, 1)

def tensorFromSentenceAnswer(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    return torch.tensor(indexes, dtype=torch.long, device=DEVICE)

def tensorsFromPair(pair, input_lang):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(input_lang, pair[0])
    return (input_tensor, target_tensor)

def tensorsFromPairAnswer(pair, input_lang):
    input_tensor = tensorFromSentenceAnswer(input_lang, pair[0])
    return input_tensor

def load_glove_model(file):
    print("Loading Glove Model")
    glove_model = {}
    with open(file,'r') as f:
        for line in f:
            split_line = line.split()
            word = split_line[0]
            embedding = np.array(split_line[1:], dtype=np.float64)
            glove_model[word] = embedding
    print(f"{len(glove_model)} words loaded!")
    return glove_model

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r"", s)
    s = re.sub(r"[^a-zA-Z]+", r" ", s)

    s_new = []
    for s in s.split(" "):
        if s in ["spheres", "cubes", "cylinders", "things"]:
            s_new.append(s[:-1])
        else:
            s_new.append(s)
    return ' '.join(s_new)

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def read_lines(n = 1500):
    question_dir = "/home/brenda/Documents/master/semestre3/independent_study/clevr-dataset-generation-mask/output/CLEVR_questions.json"
    lines = []
    images = []
    f = open(question_dir, "r")
    scn = json.load(f)
    questions = scn["questions"]
    indx = 0
    for question_block in questions:
        lines.append(question_block["question"])
        images.append(question_block["image"]+".png")
        if indx+1 == n:
            break

        indx += 1
    print("Number of Questions:", len(lines))
    return lines, images

def read_answers(n = 1500):
    question_dir = "/home/brenda/Documents/master/semestre3/independent_study/clevr-dataset-generation-mask/output/CLEVR_questions.json"
    lines = []
    f = open(question_dir, "r")
    scn = json.load(f)
    questions = scn["questions"]
    indx = 0
    for question_block in questions:
        lines.append(question_block["answer"])

        if indx+1 == n:
            break

        indx += 1
    print("Number of Answers:", len(lines))
    return lines

class Lang:
    def __init__(self, name, extra_tokens = True):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        if extra_tokens:
            self.index2word = {0: "SOS", 1: "EOS"}
        else:
            self.index2word = {0: "SOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.show()

def from_mask_to_image(im, outputs, height, width):
    temp_color_mask = []
    for indx in range(len(outputs["instances"].pred_classes)):
        temp_colored_img = np.zeros((height, width, 3))
        for i, pair_0 in zip(range(len(outputs["instances"].pred_masks[indx])), outputs["instances"].pred_masks[indx]):
            for j, pair_1 in zip(range(len(pair_0)), pair_0):
                temp_colored_img[i, j] = im[i, j] if pair_1 else temp_colored_img[i, j]
        temp_color_mask.append(temp_colored_img)
    return temp_color_mask