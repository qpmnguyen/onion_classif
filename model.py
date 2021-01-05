import torch.nn as nn

# Define the model using nn.module 
# Here we use the Deep Averaging Network, a very simple neural network model using an embedding layer  
class DAN(nn.Module):
    def __init__(self, n_classes, vocab, pretrain = True, emb_dim = 100, n_hidden = 100):
        super(DAN, self).__init__()
        self.n_classes = n_classes
        self.vocab_size = len(vocab)
        self.emb_dim = emb_dim, 
        self.n_hidden = n_hidden
        self.embed = nn.EmbeddingBag(self.vocab_size, self.emb_dim, mode = "mean")
        if pretrain == True:
            self.embed.weight.data.copy_(vocab.vectors)
            self.embed.weight.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.ReLu(), 
            nn.Linear(self.n_hidden, self.n_classes)
        )
        self._softmax = nn.Softmax()
    def forward(self, batch, probs=False):
        text = batch.text
        # pass embedding to text layer 
        x = self.embed(text)
        logit = self.classifier(x)
        if probs:
            return self._softmax(logit)
        else:
            return(logit)







