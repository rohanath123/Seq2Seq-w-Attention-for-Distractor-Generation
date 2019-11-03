import numpy as np 
import pandas as pd 
import csv 
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import *
from sklearn.feature_extraction.text import CountVectorizer
from data_cleaning_test import *
from seq2seq import *
from arch_testing import *


hidden_size = 256
encoder1 = EncoderRNN(input_lang.n, hidden_size)
encoder1.load_state_dict(torch.load("D:/Deep Learning Trained Models/ValueLabs/encoder1_1m_1_pure.pth"))
encoder1 = encoder1.cuda()

attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n, dropout_p=0.1)
attn_decoder1.load_state_dict(torch.load("D:/Deep Learning Trained Models/ValueLabs/attn_decoder1_1m_1_pure.pth"))
attn_decoder1 = attn_decoder1.cuda()

def evaluateRandomly(encoder, decoder):
    for i in range(len(test)):
    	print(test[i])
    	output_words, attentions = evaluate(encoder, decoder, test[i])
    	output_sentence = ' '.join(output_words)
    	print('<', output_sentence)
    	print('')

evaluateRandomly(encoder1, attn_decoder1)