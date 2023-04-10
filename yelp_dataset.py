import numpy as np
import torch
from torch.utils.data import Dataset

def tokenize_review(review, token_dict, maxlen):
    tokenization = np.zeros(maxlen)
    review = review.lower()
    review = review.split(' ')
    for i in range(maxlen):
        while True:
            try:
                word = review.pop()
                token = token_dict[word]
                tokenization[i] = (token)
                break
            except:
                if not review:
                    break
    #tokenization = torch.tensor(tokenization, dtype = torch.long)
    return tokenization

class YelpData(Dataset):
    def __init__(self, data = None, maxlen = None, token_dict = None, stop = None, path = None):
        if path is None:
            tokens = np.empty((len(data), maxlen))
            labels = np.empty(len(data))
            for idx, review in enumerate(data):
                
                tokens[idx] = tokenize_review(review['text'], token_dict, maxlen)
                labels[idx] = review['label']
                if stop and idx > stop:
                    break
            self.tokens = torch.tensor(tokens, dtype = torch.long)
            self.labels = torch.tensor(labels, dtype = torch.long)
            self.length = self.labels.shape[0]
        if path is not None:
            self.tokens = torch.load(path + '_tokens')
            self.labels = torch.load(path + '_labels')
            self.length = self.labels.shape[0]

    def __getitem__(self, idx):
        return self.tokens[idx], self.labels[idx]

    def __len__(self):
        return self.length
    
    def save(self, pathname):
        torch.save(self.tokens, pathname + '_tokens')
        torch.save(self.labels, pathname + '_labels')