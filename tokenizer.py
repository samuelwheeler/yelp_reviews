import torch
import numpy as np
from datasets import load_dataset
import yelp_dataset as yd


dataset = load_dataset("yelp_review_full")
train_data = dataset['train']
test_data = dataset['test']


# File path
embedding_file = '/content/drive/MyDrive/language_classification/glove.twitter.27B.50d.txt'

# read in embeddings
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(embedding_file))


words = []
idx = 0
word2idx = {}
embedding_weights = [np.zeros(50)]
with open(embedding_file) as f:
    for l in (f):
        word, vect = get_coefs(*l.strip().split())
        try:
            assert len(vect) == 50
            idx += 1
            embedding_weights.append(vect)
            words.append(word)
            word2idx[word] = idx
            
        except:
            print(idx)
            continue

embedding_weights = np.array(embedding_weights)

embedding_weights = torch.tensor(embedding_weights)


train_set = yd.YelpData(data = train_data, maxlen = 100, token_dict = word2idx)
test_set = yd.YelpData(data = test_data, maxlen = 100, token_dict = word2idx)


torch.save(embedding_weights, 'embedding_weights')
train_set.save('train_set')
test_set.save('test_set')

