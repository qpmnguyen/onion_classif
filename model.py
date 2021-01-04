import torch.nn as nn
from torchtext.data import Field, TabularDataset, BucketIterator
from torchtext.vocab import GloVe
import pandas as pd
import numpy as np
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import re 

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

# Define the model using nn.module 
# Here we use the Deep Averaging Network, a very simple neural network model using an embedding layer  
class DAN(nn.Module):
    def __init__(self, n_classes, vocab, emb_dim = 100, n_hidden = 100):
        super(DAN, self).__init__()
        self.n_classes = n_classes
        self.vocab_size = len(vocab)
        self.emb_dim = emb_dim, 
        self.n_hidden = n_hidden
        self.embed = nn.EmbeddingBag(self.vocab_size, self.emb_dim, mode = "mean")
        self.embed.weight.data.copy_(vocab.vectors)
        self.classifier = nn.Sequential(
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.ReLu(), 
            nn.Linear(self.n_hidden, self.n_classes)
        )
        self._softmax = nn.Softmax()
    def forward(self, batch, probs=False):




