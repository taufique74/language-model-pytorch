# coding: utf-8
import argparse
import time
import math
import os
import torch
import torch.nn as nn
import torch.onnx
import pickle
import wandb
import json
from dataloader import DataLoader
from args import args

from utils import millify
from models import RNNModel, AdaptiveSoftmaxRNN
from trainer import Trainer


adaptive = True

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")

# save the arguments in a json file
if not os.path.exists(args.save):
    os.mkdir(args.save)
with open(os.path.join(args.save, 'options.json'), 'w') as f:
    json.dump(args.__dict__, f)

###############################################################################
# Load data and vocabulary
###############################################################################
vocab_cache = f'{args.data}/vocab.pickle'

if(os.path.exists(vocab_cache)):
    print('[#] Found vocab cache in the corpus directory')
    print('[#] Loading the vocab cache...')
    with open(vocab_cache, 'rb') as f:
        vocab_cache = pickle.load(f)
    print('[#] Loading the corpus..')
    corpus = DataLoader(args.data, args.min_freq, args.add_eos, vocab_cache)
    cache = {
    'idx2word': corpus.dictionary.idx2word,
    'word2idx': corpus.dictionary.word2idx,
    'total_tokens': corpus.dictionary.total_tokens
    }
    
else:
    print('[#] No vocab cache found!')
    print('[#] Loading the corpus and building the vocabulary...')
    corpus = DataLoader(args.data, args.min_freq, args.add_eos)
    print('[#] Saving the vocabulary cache..')
    with open(vocab_cache, 'wb') as f:
        cache = {
            'idx2word': corpus.dictionary.idx2word,
            'word2idx': corpus.dictionary.word2idx,
            'total_tokens': corpus.dictionary.total_tokens
        }
        pickle.dump(cache, f)


eval_batch_size = 10
train_data = DataLoader.batchify(corpus.train, args.batch_size, device)
val_data = DataLoader.batchify(corpus.valid, args.eval_batch_size, device)
test_data = DataLoader.batchify(corpus.test, args.eval_batch_size, device)

###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)
cutoffs = [int(cutoff) for cutoff in args.cutoffs.split()]
if not args.adaptive:
    model = RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.emb_dropout, args.tied).to(device)
elif args.model == 'LSTM':
    model = AdaptiveSoftmaxRNN(
        ntokens,
        args.emsize,
        args.nhid,
        args.nlayers,
        emb_dropout = args.emb_dropout,
        rnn_dropout = args.rnn_dropout,
        tail_dropout = args.tail_dropout,
        cutoffs = cutoffs,
#         tie_weights = args.tied,
#         adaptive_input=True
    ).to(device)

criterion = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer = optimizer,
    mode = 'min',
    factor = 0.5,
    patience = args.patience,
    verbose = True,
    min_lr = 1.0
)

total_tokens = corpus.dictionary.total_tokens
vocabulary = len(corpus.dictionary)
print(f'[#] total tokens: {total_tokens} ({millify(total_tokens)})')
print(f'[#] vocabulary size: {vocabulary} ({millify(vocabulary)})')
print('-' * 89)
print(model)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'[#] total params: {total_params} ({millify(total_params)})')
print(f'[#] trainable params: {trainable_params} ({millify(trainable_params)})')
print('-' * 89)

###############################################################################
# Training code
###############################################################################


trainer = Trainer(
        corpus, model, args, criterion, optimizer, scheduler, train_data, test_data, val_data, cache
    )

trainer.fit()
