import argparse

parser = argparse.ArgumentParser(description='PyTorch Language Model')
parser.add_argument('--data', type=str, default='data',
                    help='location of the data corpus')
parser.add_argument('--min_freq', type=int, default=2,
                    help='minimum frequency for a word in the corpus to get included in the vocabulary')
parser.add_argument('--add_eos', action='store_true',
                    help='whether to add and <eos> token after a training sample')                    
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU, Transformer)')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=40,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='batch size')
parser.add_argument('--eval_batch_size', type=int, default=10, 
                    help='eval batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--emb_dropout', type=float, default=0.2,
                    help='dropout applied to embedding')
parser.add_argument('--rnn_dropout', type=float, default=0.2,
                    help='dropout applied to rnn layers')
parser.add_argument('--out_dropout', type=float, default=0.5,
                    help='dropout on the output of rnn')
parser.add_argument('--tail_dropout', type=float, default=0.3,
                    help='dropout applied to tail clusters')
parser.add_argument('--cutoffs', type=str, default='20000 50000',
                    help='Cutoff values for adaptive input and adaptive softmax')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--no_log', action='store_true',
                    help='don\'t log')
parser.add_argument('--adaptive', action='store_true',
                    help='use adaptive softmax')
parser.add_argument('--no_save', action='store_true',
                    help='don\'t save any model or checkpoints')                                      
parser.add_argument('--save', type=str, default='checkpoints',
                    help='path to save the final model')
parser.add_argument('--patience', type=int, default=0,
                    help='LR Scheduler patience')


args = parser.parse_args()