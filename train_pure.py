#AGGRESSIVE EXHAUSTIVE INDIVIDUAL MODEL ARCHITECTURE TESTING

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from data_cleaning import *
from seq2seq import *
import random
from torch.autograd import Variable

device = torch.device("cuda")
teacher_forcing_ratio = 0.5
criterion = nn.CosineSimilarity()
#criterion = nn.CosineEmbeddingLoss()
#criterion = nn.NLLLoss()
def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden
    decoded_words = []

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        
    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            #loss += criterion(decoder_output, target_tensor[di].float(), torch.ones(target_tensor.size(), dtype = torch.float))
            decoder_input = target_tensor[di]  # Teacher forcing
            topv, topi = decoder_output.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

    out = ' '.join(decoded_words[i] for i in range(len(decoded_words)))
    dec_op = tensorsFromSentence(output_lang, out)
    print(decoder_output.size())
    print(target_tensor.size())
    loss = criterion(decoder_output, target_tensor)
    #dec_op, target_tensor = pad_diff(dec_op, target_tensor)
    #loss = criterion(Variable(target_tensor.float(), requires_grad = True), Variable(dec_op.float(), requires_grad = True), torch.ones(target_tensor.size(), dtype = torch.float).cuda())

            #loss += criterion(decoder_output, target_tensor[di].float(), torch.ones(target_tensor.size(), dtype = torch.float))
            #if decoder_input.item() == EOS_token:
            #   break


    '''
    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)

            v1, v2 = one_hot_both(target_tensor[di], decoder_output)
            print(v1, v2)
            loss += criterion(v1, v2)
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            v1, v2 = one_hot_both(target_tensor[di], decoder_output)
            print(v1, v2)
            loss += criterion(v1, v2)

            if decoder_input.item() == EOS_token:
                break
    '''

                
    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

import time
import math


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

def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    #IF DIMENSIONALITY PROBLEM, LOOK HERE, MAYBE YOULL NEED TO REDUCE ONE DIMENSION BECAUSE WE'RE APPENDING 
    training_pairs = []
    odd = []
    for i in range(n_iters):
        try:
            training_pairs.append(tensorsFromPair(random.choice(pairs)))
        except:
            odd.append(i)
            training_pairs.append(training_pairs[i-1])
    #training_pairs = [tensorsFromPair(random.choice(pairs))
    #                  for i in range(n_iters)]

    #criterion = nn.NLLLoss()
    

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    #showPlot(plot_losses)

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorsFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]


def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')

hidden_size = 256
encoder1 = EncoderRNN(input_lang.n, hidden_size).to(device)
attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n, dropout_p=0.1).to(device)

trainIters(encoder1, attn_decoder1, 1000, print_every=50)
evaluateRandomly(encoder1, attn_decoder1)


'''
torch.save(encoder1.state_dict(), "D:/Deep Learning Trained Models/ValueLabs/encoder1_10000_tf075_emb_pure.pth")
torch.save(attn_decoder1.state_dict(), "D:/Deep Learning Trained Models/ValueLabs/attn_decoder1_0000_tf075_emb_pure.pth")


encoder_test = EncoderRNN(input_lang.n, hidden_size).to(device)
encoder_input = tensorsFromSentence(input_lang, pairs[34][0])
for ei in range(len(encoder_input)):
	output, hidden = encoder_test(encoder_input[ei], encoder_test.initHidden())
print(output)
print(hidden)
print(output.size())
print(hidden.size())
print(hidden == output)
'''