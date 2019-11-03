import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import re
import unicodedata
from sklearn.feature_extraction.text import CountVectorizer
from torch.autograd import Variable

device = torch.device("cuda")

SOS_token  = 0
EOS_token = 1

class Lang:
	def __init__(self, name):
		self.name = name
		self.word2index = {}
		self.index2word = {0: 'SOS', 1: 'EOS'}
		self.word2count = {}
		self.n = 2

	def addSentence(self, sentence):
		for word in sentence.split(' '):
			self.addWord(word)

	def addWord(self, word):
		if word not in self.word2index:
			self.word2index[word] = self.n
			self.index2word[self.n] = word
			self.n += 1
			self.word2count[word] = 1

		else:
			self.word2count[word] += 1


	def print_everything(self):
		print(self.word2index)
		print(self.word2count)
		print(self.index2word)
		print(self.n)

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def readLangs(lang1, lang2, PATH, reverse):
	lines = open(PATH, encoding = 'utf-8').read().strip().split('\n')

	pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
	#print(len(pairs[0]))

	if reverse:
		pairs = [list(reversed(p)) for p in pairs]
		input_lang = Lang(lang2)
		output_lang = Lang(lang1)
	else:
		input_lang = Lang(lang1)
		output_lang = Lang(lang2)

	return input_lang, output_lang, pairs

		
MAX_LENGTH = 680
odd = []
def prepareData(lang1, lang2, PATH, reverse = False):
	input_lang, output_lang, pairs = readLangs(lang1, lang2, PATH, reverse)
	#pairs = filterPairs(pairs)
	for i in range(len(pairs)):
		try:
			input_lang.addSentence(pairs[i][0])
			output_lang.addSentence(pairs[i][1])
		except:
			odd.append(i)
	return input_lang, output_lang, pairs

def addData(input_lang, output_lang, old_pairs, new_pairs):
	for i in range(len(new_pairs)):
		try:
			input_lang.addSentence(new_pairs[i][0])
			output_lang.addSentence(new_pairs[i][1])
		except:
			odd.append(i)
		total_pairs = old_pairs + new_pairs
	return input_lang, output_lang, total_pairs

input_lang, output_lang, pairs = prepareData('train', 'distractors', 'D:/Machine Learning Datasets/ValueLabs Distractors/DataSet/train_single2.txt')
#_, _, pairs2 = prepareData('train', 'distractors', 'D:/Machine Learning Datasets/ValueLabs Distractors/DataSet/train_test.txt')
#input_lang, output_lang, pairs = addData(input_lang, output_lang, pairs, pairs2)

def init_vectors():
	x = [input_lang.index2word[i] for i in range(input_lang.n)]
	string1 = ' '.join(x[i] for i in range(len(x)))
	x = [output_lang.index2word[i] for i in range(output_lang.n)]
	string2 = ' '.join(x[i] for i in range(len(x)))
	vectorizer = CountVectorizer()
	corpus = [string1+string2]
	vectorizer.fit(corpus)
	return vectorizer

def indexesFromSentence(lang, sentence):
	return [lang.word2index[word] for word in sentence.split(' ')]

def tensorsFromSentence(lang, sentence):
	indexes = indexesFromSentence(lang, sentence)
	indexes.append(EOS_token)
	return torch.tensor(indexes, dtype = torch.long, device = device).view(-1, 1)

def tensorsFromPair(p):
	input_tensor = tensorsFromSentence(input_lang, p[0])
	output_tensor = tensorsFromSentence(output_lang, p[1])
	return (input_tensor, output_tensor)

vectorizer = init_vectors()

def one_hot_both(input, target):
	input_word = input_lang.index2word[input.item()]
	topv, topi = target.data.topk(1)
	target_word = output_lang.index2word[topi.item()]

	print(input_word)
	print(target_word)

	vector1 = vectorizer.transform([input_word]).toarray()
	vector2 = vectorizer.transform([target_word]).toarray()
	
	#return Variable(torch.FloatTensor(vector1), requires_grad = True), Variable(torch.FloatTensor(vector2), requires_grad = True)
	return vector1, vector2

def pad_diff(tens1, tens2):
	if tens1.size()[0] >= tens2.size()[0]:
		diff = tens1.size()[0] - tens2.size()[0]
		tens2 = F.pad(tens2, (0, 0, 0, diff), value = 1)

	else:
		diff = tens2.size()[0] - tens1.size()[0]
		tens1 = F.pad(tens1, (0, 0, 0, diff), value = 1)

	return tens1, tens2
		


	