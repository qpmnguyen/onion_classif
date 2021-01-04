import pandas as pd  
import numpy as np  

# Splitting data  

def split(prop, data):
    idx = np.arange(data.shape[0])
    np.random.shuffle(idx)
    test_idx = int(prop * len(idx))
    test = data.iloc[idx[:test_idx]]
    train = data.iloc[idx[test_idx:]]
    return(train, test)

data = pd.read_csv("data/combined.tsv", sep = "\t")


train, test = split(prop = 0.2, data = data)

train, valid = split(prop = 0.2, data = train)

test.to_csv("data/test.tsv", sep = "\t", index = False)
train.to_csv("data/train.tsv", sep = "\t", index = False)
valid.to_csv("data/valid.tsv", sep = "\t", index = False)