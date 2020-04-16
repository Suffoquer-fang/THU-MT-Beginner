import pickle 
import numpy as np 
class WordTable():
    def __init__(self):
        self.word2id = {}
        self.id2word = {}
        self.id = 0
        self.add_word('<pad>')
        self.add_word('<eos>')

    def load(self, path1, path2):
        with open(path1, 'rb') as f:
            self.word2id = pickle.load(f)
        with open(path2, 'rb') as f:
            self.id2word = pickle.load(f)
    
    def add_word(self, word):
        if not word in self.word2id: #如果词到索引的映射字典中 不包含该词 则添加
            self.word2id[word] = self.id
            self.id2word[self.id] = word 
            self.id += 1


class Corpus(object):
    def __init__(self, datapath):
        self.wordTable = WordTable() 
        self.path = datapath
        with open(self.path, 'r') as f:
            self.data = f.readlines()
 
    def get_data(self):
        
        tokens = 0
        for line in self.data: 
            words = line.split() + ['<eos>'] 
            tokens += len(words)
            for word in words:
                self.wordTable.add_word(word)  
        
    def get_vector(self, id, seqlen=10):
        words = self.data[id].split()
        x = []
        y = []
        for word in words:
            x.append(self.wordTable.word2id[word])
        
        y = x.copy()
        y.pop(0)
        y.append(self.wordTable.word2id['<eos>'])
        
        tmp = [0 for i in range(seqlen)]
        x += tmp
        y += tmp
        x = x[:seqlen]
        y = y[:seqlen]

        return np.array(x), np.array(y)

    def get_words(self, vec):
        ret = []
        for idx in vec:
            if idx in self.wordTable.id2word:
                ret.append(self.wordTable.id2word[idx])
            else:
                ret.append('<unknown>')
        return ret

        
            