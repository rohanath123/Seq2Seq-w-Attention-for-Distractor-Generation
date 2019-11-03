import torch
from transformers import BertTokenizer, BertModel, BertForMaskedML

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

text = "[CLS] Who is Rohan Athawade ? [SEP] Rohan Athawade was an inventor [SEP]"
tokenized_text = tokenizer.tokenize(text)

print(tokenized_text)