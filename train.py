from torchtext.data import Field, TabularDataset, BucketIterator
from torchtext.vocab import GloVe
import pandas as pd
import numpy as np
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import re 
import torch.nn as nn
import torch.optim as optim
from model import *

spacy_en = spacy.load('en')
def tokenizer(txt):
    """Simple tokenizer
    txt: Text to tokenize, as String

    First, remove all quotations, stars, +, minuses, etc. 
    Then, substitute certain punctuations with the correct one
    Finally, tokenize using spacy_en and remove stop words  
    """
    txt = re.sub(
        r"[\*\"“”\n\\…\+\-\/\=\(\)‘•:\[\]\|’\!;]", " ", 
        str(txt))
    txt = re.sub(r"[ ]+", " ", txt)
    txt = re.sub(r"\!+", "!", txt)
    txt = re.sub(r"\,+", ",", txt)
    txt = re.sub(r"\?+", "?", txt)
    tokenized = [tok.text for tok in spacy_en.tokenizer(txt)]
    processed = [tok for tok in tokenized if tok not in STOP_WORDS]
    return(processed)


text = Field(sequential=True, use_vocab=True, tokenize = tokenizer)
label = Field(sequential=False, use_vocab=False)

fields = {
    ("label", label),
    ("text", text),
}

# TSV imported weirdly 
train, val, test = TabularDataset.splits(path = "data/", train = "train.tsv", validation = "valid.tsv", test = "test.tsv", format = "tsv", fields = fields)

text.build_vocab(train, min_freq = 2, vectors = GloVe("6B", "100"))
label.build_vocab(train, min_freq = 2, vectors = GloVe("6B", "100"))

train_iter, val_iter, test_iter = BucketIterator.splits( (train,test), 
            batch_size = 3)

model = DAN(2, text.vocab, pretrain=True, n_hidden = 256)
def train(iterator, model, n_epoch, lr):
    # binary cross entropy loss 
    criterion = nn.BCELoss()
    opt = optim.Adam(model.parameters(), lr = lr)
    for batch in iterator:
        model.train()
        # zero out the gradients per batch 
        opt.zero_grad()
        # forward pass 
        out = model(batch)
        loss = criterion(out, batch.label)
        # backprop 
        loss.backward()
        opt.step()
    

