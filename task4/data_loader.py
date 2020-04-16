import torch
import os
from tqdm import tqdm

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        word = word.lower()
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __len__(self):
        return len(self.word2idx)


class Corpus(object):
    def __init__(self):
        self.dictionary = Dictionary()

    def get_data(self, path, batch_size=20):
        print('Start Collecting Words...')
        with open(path, 'r') as f:
            tmp = f.readlines()

        tmp = tmp[:100000]

        tokens = 0
        for line in tqdm(tmp):
            words = line.split() + ['<eos>']
            tokens += len(words)
            for word in words: 
                self.dictionary.add_word(word.lower())  

        print('Start ids....')
        ids = torch.LongTensor(tokens)
        token = 0
        print(ids.shape)
        for line in tqdm(tmp):
            words = line.split() + ['<eos>']
            for word in words:
                ids[token] = self.dictionary.word2idx[word.lower()]
                token += 1
        num_batches = ids.size(0) // batch_size
        ids = ids[:num_batches*batch_size]
        return ids.view(batch_size, -1)