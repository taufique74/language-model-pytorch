import os
import torch
import pickle
from .dictionary import Dictionary

class DataLoader(object):
    def __init__(self, path, min_freq=2, add_eos=False, vocab_cache=None):
        self.dictionary = Dictionary(vocab_cache, min_freq, add_eos)
        train_path = os.path.join(path, 'train.txt')
        test_path = os.path.join(path, 'test.txt')
        valid_path = os.path.join(path, 'valid.txt')
        
        if vocab_cache is not None:
            self.dictionary.load_from_cache()
        else:
            self.dictionary.build_vocab(train_path)
            self.dictionary.build_vocab(test_path)
            self.dictionary.build_vocab(valid_path)

            self.dictionary.sort_vocab()
        
        self.train = self.tokenize(train_path)
        self.valid = self.tokenize(valid_path)
        self.test = self.tokenize(test_path)
        
    def tokenize(self, path):
        assert os.path.exists(path)
        # self.dictionary.build_vocab(path, add_eos=False)
        with open(path, 'r', encoding="utf8") as f:
            idss = []
            for line in f:
                words = line.split()
                ids = []
                for word in words:
                    if word in self.dictionary.word2idx:
                        ids.append(self.dictionary.word2idx[word])
                    else:
                        ids.append(self.dictionary.word2idx['<unk>'])
                idss.append(torch.tensor(ids).type(torch.int64))
            ids = torch.cat(idss)
        
        return ids
    
    @staticmethod
    def batchify(data, bsz, device):
        # Data should be torch tensor
        nbatch = data.size(0) // bsz
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, nbatch * bsz)
        # Evenly divide the data across the bsz batches.
        data = data.view(bsz, -1).t().contiguous()
        return data.to(device)
    
    @staticmethod
    def get_batch(source, i, bptt):
        seq_len = min(bptt, len(source) - 1 - i)
        data = source[i:i+seq_len]
        target = source[i+1:i+1+seq_len].view(-1)
        return data, target

# corpus = DataLoader('data/wikitext-2/')
# print(corpus.dictionary.total_tokens)