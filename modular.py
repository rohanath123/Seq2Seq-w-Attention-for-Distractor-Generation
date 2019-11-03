#MODULAR TESTING AND IMPLEMENTATION OF CODE BLOCKS

#GLOVE TESTING 

import numpy as np 
import pandas as pd 
import csv 
import torch
#from data_cleaning_final import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import *
from sklearn.feature_extraction.text import CountVectorizer
from data_cleaning_test import *
'''

vecs = pd.read_csv("D:/Random/Glove/glove.6B.50d.csv", sep = " ", index_col = 0, header=None, quoting = csv.QUOTE_NONE)



input_size = input_lang.n
output_size = output_lang.n
hidden_size = 256
emb_dim = hidden_size
#emb_size_enc = (input_size, hidden_size)
#emb_size_dec = (output_size, hidden_size)


#FOR DEC 
matrix_len = input_size
weights_matrix = np.zeros((matrix_len, 256))
words_found = 0

for i, word in enumerate(target_vocab):
    try: 
        weights_matrix[i] = glove[word]
        words_found += 1
    except KeyError:
        weights_matrix[i] = np.random.normal(scale=0.6, size=(emb_dim, ))


for i in range(len(input_lang.index2word)):
	try: 
		weights_matrix[i] = np.resize(vec(input_lang.index2word[i]), 256)
		words_found += 1
	except KeyError:
		#print(i)
		weights_matrix[i] = np.random.normal(scale = 0.6, size = (256, ))

input = tensorsFromSentence(input_lang, pairs[50][0])
emb_layer = nn.Embedding.from_pretrained(torch.FloatTensor(weights_matrix))
#emb_layer.weight.requires_grad = False
print(emb_layer(input.cpu()))


vectorizer = CountVectorizer()

corpus = [['This is the first document.' 'This document is the second document.' 'And this is the third one.' 'Is this the first document?']]
input = pairs[0][0]
output = pairs[0][1]
corpus = [[input+output]]
vectorizer.fit(corpus[0])
vector1 = vectorizer.transform([input]).toarray()
vector2 = vectorizer.transform([output]).toarray()
cos = nn.CosineSimilarity(dim = 1)
loss = cos(torch.FloatTensor(vector2), torch.FloatTensor(vector1))
print(loss.item())



t = torch.tensor([2])
print(t)
print(t.item())
#print(vectorizer.transform(['my name is']).toarray())

def pretrained_embeddings(size1, size2, lang):
    matrix_len = size1
    weights_matrix = np.zeros((matrix_len, size2))
    for i in range(len(lang.index2word)):
        try: 
            weights_matrix[i] = np.resize(vec(lang.index2word[i]), size2)
        except KeyError:
            weights_matrix[i] = np.random.normal(scale = 0.6, size = (size2, ))

    emb_layer = nn.Embedding.from_pretrained(torch.FloatTensor(weights_matrix), freeze = False)
    return emb_layer

enc_emb = pretrained_embeddings(input_lang.n, 256, input_lang)
dec_emb = pretrained_embeddings(output_lang.n, 256, output_lang)
torch.save(enc_emb, 'D:/Machine Learning Datasets/ValueLabs Distractors/DataSet/enc_emb.pt')
torch.save(dec_emb, 'D:/Machine Learning Datasets/ValueLabs Distractors/DataSet/dec_emb.pt')

vecs = pd.read_csv("D:/Random/Glove/glove.6B.50d.csv", sep = " ", index_col = 0, header=None, quoting = csv.QUOTE_NONE)

def vec(w):
    return vecs.loc[w].as_matrix()

enc_emb = torch.load('D:/Machine Learning Datasets/ValueLabs Distractors/DataSet/enc_emb.pt')
dec_emd = torch.load('D:/Machine Learning Datasets/ValueLabs Distractors/DataSet/dec_emb.pt')

c = nn.CosineEmbeddingLoss()
input = tensorsFromSentence(input_lang, pairs[50][0])
target = tensorsFromSentence(output_lang, pairs[50][1])
print(input.size())
print(target.size())
target = torch.nn.functional.pad(target, (0,0,0, 6), value = 0)
print(input.size())
print(target.size())
loss = c(target.float(), input.float(), torch.ones(target.size(), dtype = torch.float).cuda())
print(loss.item())
'''
import random
r = random.randint(0, len(pairs))
print(pairs[r])
#print(tensorsFromSentence(input_lang, pairs[r][0]).size())
#print(tensorsFromSentence(output_lang, pairs[r][1]).size())