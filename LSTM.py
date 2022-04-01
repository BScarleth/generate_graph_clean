from __future__ import unicode_literals, print_function, division

import numpy as np
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import time, json
from utils import timeSince, showPlot, Lang, read_lines, tensorsFromPair, tensorFromSentence, prepareData, load_glove_model

DEVICE = "cpu"

class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, input_size = None, wembeddings = None):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        if wembeddings is None:
            self.embedding = nn.Embedding(input_size, hidden_size)
        else:
            print("Embeddings loaded!")
            self.embedding = torch.nn.Embedding.from_pretrained(torch.from_numpy(wembeddings).float())
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=DEVICE)

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length= 15):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=DEVICE)

def train_lstm(lines, name, path, iters, pretrained=True):
    hidden_size = 300
    input_lang, pairs, _ = prepareData('questions', lines)
    print(random.choice(pairs))

    words = input_lang.word2count.keys()
    glove_model = load_glove_model("/home/brenda/Documents/master/semestre3/independent_study/generate_graph_clean/output/glove.6B.300d.txt")
    words_embedding = []

    words_embedding.append(np.random.rand(300))
    words_embedding.append(np.random.rand(300))
    for w in words:
        words_embedding.append(glove_model[w])

    wembeddings = np.array(words_embedding)

    print("Embeddings shape: ", wembeddings.shape, " words: ", input_lang.n_words)
    encoder = EncoderRNN(hidden_size, wembeddings=wembeddings).to(DEVICE)
    attn_decoder = AttnDecoderRNN(hidden_size, input_lang.n_words, dropout_p=0.1).to(DEVICE)

    if pretrained:
        encoder.load_state_dict(torch.load(path + "encoder_{}".format(name)))
        attn_decoder.load_state_dict(torch.load(path + "decoder_{}".format(name)))

    trainIters(encoder, attn_decoder, pairs, input_lang, n_iters=iters, print_every=100, name=name, path_model=path)
    evaluate_randomly(encoder, attn_decoder, pairs, input_lang)

def trainIters(encoder, decoder, pairs, input_lang, n_iters, print_every=100, plot_every=100, learning_rate=0.01, name="default", path_model = None):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [tensorsFromPair(random.choice(pairs), input_lang) for i in range(n_iters)]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[0]  # no 1

        loss = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)

        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))
            #print(path_model + "encoder_{}".format(name))
            torch.save(encoder.state_dict(), path_model + "encoder_{}".format(name))
            torch.save(decoder.state_dict(), path_model + "decoder_{}".format(name))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    #showPlot(plot_losses)

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=15, SOS_token = 0, EOS_token = 1):
    teacher_forcing_ratio =  0.5
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=DEVICE)

    loss = 0

    #print("input_length ", input_length)
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        #print("encoder_output ", encoder_output.shape)
        #print(ei)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=DEVICE)
    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)

            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

def test_lstm(lines, name, encoder, attn_decoder, path_model):
    input_lang, pairs, _ = prepareData('questions', lines)
    print(random.choice(pairs))

    encoder.load_state_dict(torch.load(path_model + "encoder_{}".format(name)))
    attn_decoder.load_state_dict(torch.load(path_model + "decoder_{}".format(name)))
    evaluate_randomly(encoder, attn_decoder, pairs, input_lang)

def evaluate_randomly(encoder, decoder, pairs, input_lang, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[0])
        output_words, attentions = evaluate(encoder, decoder, pair[0], input_lang)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')

def evaluate(encoder, decoder, sentence, input_lang, max_length=15, SOS_token = 0, EOS_token = 1):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=DEVICE)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=DEVICE)  # SOS
        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(input_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]

def predict_encode(encoder, input_lang, sentence, max_length=15):
    #print("input size: ", input_lang.n_words)

    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=DEVICE)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]
    return encoder_hidden, encoder_outputs

def decode_question(decoder, decoder_input, input_lang, decoder_hidden, encoder_outputs):
    with torch.no_grad():
        EOS_token = 1
        decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
        topv, topi = decoder_output.data.topk(1)
        if topi.item() == EOS_token:
            decoded_words = '<EOS>'
        else:
            decoded_words = input_lang.index2word[topi.item()]

        decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_input, decoder_hidden







