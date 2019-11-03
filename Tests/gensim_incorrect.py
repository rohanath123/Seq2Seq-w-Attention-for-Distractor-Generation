import gensim 
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors

glove_file = 'D:/Deep Learning Trained Models/embeddings/glove.6B/glove.6B.300d.txt'
tmp_file = 'D:/Deep Learning Trained Models/embeddings/glove.6B/word2vec-glove.6B.300d.txt'

from gensim.scripts.glove2word2vec import glove2word2vec
glove2word2vec(glove_file, tmp_file)
model = KeyedVectors.load_word2vec_format(tmp_file)


print(model.most_similar(positive = ['hello my name is'], topn = 3))