import torch 
import numpy as np 
from seq2seq import *
from data_cleaning import *
#from train import *

import random
from torch import optim
import torch.nn as nn


'''
hidden_size = 256
encoder1 = EncoderRNN(input_lang.n, hidden_size).to(device)
attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n, dropout_p=0.1).to(device)

encoder1.load_state_dict(torch.load( "D:/Deep Learning Trained Models/Seq2Seq Test/encoder1_75000.pth"))
attn_decoder1.load_state_dict(torch.load( "D:/Deep Learning Trained Models/Seq2Seq Test/attn_decoder1_75000.pth"))


trainIters(encoder1, attn_decoder1, 75000)


#TEST
print(pairs[30][0])
print(pairs[30][1])
output, attns = evaluate(encoder1, attn_decoder1, pairs[30][0])
print(output)

	

torch.save(encoder1.state_dict(), "D:/Deep Learning Trained Models/Seq2Seq Test/encoder1_75000.pth")
torch.save(attn_decoder1.state_dict(), "D:/Deep Learning Trained Models/Seq2Seq Test/attn_decoder1_75000.pth")



output, attns, decoder_outputs = evaluate(encoder1, attn_decoder1, pairs[30][0])

for i in range(len(decoder_outputs)):
	topv, topi = decoder_outputs[i].data.topk(1)
	print(topi)
	print(topv)
'''

#print(tensorsFromSentence(output_lang, 'Je suis parti.'))
#trainIters(encoder1, attn_decoder1, 1)
#print(tensorsFromSentence(input_lang, 'am i first ?'))
#print(input_lang.index2word)


''''
for i in range(10):
	r = random.randint(0, len(pairs))
	print(pairs[r][0])
	print(pairs[r][1])
	#output, attns, x = evaluate(encoder1, attn_decoder1, pairs[r][0])
	#print(output)

print('NEXT')
for i in range(10):
	r = random.randint(0, len(pairs2))
	print(pairs2[r][0])
	print(pairs2[r][1])

#print(output_lang.index2word)


print(input.size())
print(output.size())
#cos = nn.CosineSimilarity(dim = 1, eps = 1e-6)
#print(cos(input, output))

import gensim
model = gensim.models.KeyedVectors.load_word2vec_format('D:/Deep Learning Training Data/GoogleNews-vectors-negative300.bin', encoding = 'unicode_escape')
weights = torch.FloatTensor(model.vectors) 
print(weights)
'''


r = random.randint(0, 200)
input = tensorsFromSentence(input_lang, pairs[r][0])
output = tensorsFromSentence(output_lang, pairs[r][1])
print(pairs[r][0])
print(input)
print(pairs[r][1])
print(output)