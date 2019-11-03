import torch 
import numpy as np 
from seq2seq import *
from data_cleaning import *
import random
from torch import optim

device = torch.device("cuda")

teacher_forcing_ratio = 0.5


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length = 10):
	encoder_hidden = encoder.initHidden()

	encoder_optimizer.zero_grad()
	decoder_optimizer.zero_grad()

	input_length = input_tensor.size(0)
	target_length = target_tensor.size(0)



	encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device = device)

	loss = 0

	for ei in range(input_length):
		encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
		encoder_outputs[ei] = encoder_hidden[0, 0]

	decoder_input = torch.tensor([[SOS_token]], device = device)
	decoder_hidden = encoder_hidden

	use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

	#cos = nn.CosineSimilarity(dim = 1, eps = 1e-6)

	if use_teacher_forcing:
		for di in range(target_length):
			decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
			loss+= criterion(decoder_output, target_tensor[di])
			print(target_tensor[di].size())
			print(decoder_output.size())

			
			decoder_input = target_tensor[di]

	else:
		for di in range(target_length):
			decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
			topv, topi = decoder_output.topk(1)
			decoder_input = topi.squeeze().detach()

			loss += criterion(decoder_output, target_tensor[di])
			print(target_tensor[di].size())
			print(decoder_output.size())
			

			if decoder_input.item() == EOS_token:
				break

	loss.backward()

	encoder_optimizer.step()
	decoder_optimizer.step()

	return loss.item()/target_length

def trainIters(encoder, decoder, iters, learning_rate = 0.01):
	print_loss_total = []

	encoder_optimizer = optim.SGD(encoder.parameters(), lr = learning_rate)
	decoder_optimizer = optim.SGD(decoder.parameters(), lr = learning_rate)

	train_pairs = [tensorsFromPair(random.choice(pairs)) for i in range(iters)]
	criterion = nn.NLLLoss()

	for iter in range(iters):
		train_pair = train_pairs[iter]
		input_tensor = train_pair[0]
		target_tensor = train_pair[1]
		#print(input_tensor)
		#print(target_tensor)

		loss= train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)

		print("Iter = ", iter, " Loss = ", loss)

def evaluate(encoder, decoder, sentence, max_length = 10):
	with torch.no_grad():
		input_tensor = tensorsFromSentence(input_lang, sentence)
		#print(input_tensor)
		input_length = input_tensor.size()[0]
		#print(input_length)
		encoder_hidden = encoder.initHidden()

		encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device = device)

		for ei in range(input_length):
			#print(input_tensor[ei])
			encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
			encoder_outputs[ei] += encoder_output[0,0]


		decoder_input = torch.tensor([[SOS_token]], device = device)
		decoder_hidden = encoder_hidden

		decoded_words = []
		decoder_attentions = torch.zeros(max_length, max_length)
		decoder_outputs = []

		for di in range(max_length):
			decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
			decoder_outputs.append(decoder_output)
			decoder_attentions[di] = decoder_attention.data
			##################
			topv, topi = decoder_output.data.topk(1)
			if topi.item() == EOS_token:
				decoded_words.append('<EOS>')
				break

			else:
				decoded_words.append(output_lang.index2word[topi.item()])

			decoded_input = topi.squeeze().detach()

		return decoded_words, decoder_attentions[:di+1], decoder_outputs

hidden_size = 256
encoder1 = EncoderRNN(input_lang.n, hidden_size).to(device)
attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n, dropout_p=0.1).to(device)
trainIters(encoder1, attn_decoder1, 100)
for i in range(10):
	r = random.randint(0, 200)
	print(pairs[r][0])
	print(pairs[r][1])
	output, attns, x = evaluate(encoder1, attn_decoder1, pairs[r][0])
	print(output)

