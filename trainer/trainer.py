import wandb
import time
import torch
import math
import os
import warnings
warnings.filterwarnings('ignore')

class Trainer():
    def __init__(
        self,
        corpus,
        model,
        args,
        criterion,
        optimizer,
        scheduler,
        train_data,
        test_data,
        val_data,
        vocab_cache
    ):
        self.corpus = corpus
        self.model = model
        self.args = args
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_data = train_data
        self.test_data = test_data,
        self.val_data = val_data
        self.vocab_cache = vocab_cache
        
        
        
    def _train(self, epoch):
        self.model.train()
        total_loss = 0
        start_time = time.time()
        ntokens = len(self.corpus.dictionary)
        args = self.args
        
        hidden = self.model.init_hidden(args.batch_size)
        
        for batch, i in enumerate(range(0, self.train_data.size(0) - 1, args.bptt)):
            data, targets = self.corpus.get_batch(self.train_data, i, args.bptt)
            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            self.optimizer.zero_grad()

            hidden = self._repackage_hidden(hidden)
            if not args.adaptive:
                output, hidden = self.model(data, hidden)
                loss = self.criterion(output, targets)
            else:
                output, hidden, loss = self.model(data, hidden, targets)

            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clip)
            # for p in model.parameters():
            #     p.data.add_(-lr, p.grad.data)

            self.optimizer.step()

            total_loss += loss.item()

            if batch % args.log_interval == 0 and batch > 0:
                cur_loss = total_loss / args.log_interval
                elapsed = time.time() - start_time
                ppl = math.exp(loss)

                print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                        'loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, batch, len(self.train_data) // args.bptt, args.lr,
                    elapsed * 1000 / args.log_interval, cur_loss, ppl))

                if not args.no_log:
                    wandb.log({'Perplexity': ppl, 'Loss': cur_loss})

                total_loss = 0
                start_time = time.time()

    def _evaluate(self, data_source):
        # Turn on evaluation mode which disables dropout.
        self.model.eval()
        args = self.args
        total_loss = 0.
        ntokens = len(self.corpus.dictionary)
        hidden = self.model.init_hidden(args.eval_batch_size)

        with torch.no_grad():
            for i in range(0, data_source.size(0) - 1, args.bptt):
                data, targets = self.corpus.get_batch(data_source, i, args.bptt)
                if args.adaptive:
                    output, hidden, loss = self.model(data, hidden, targets)
                else:
                    output, hidden = self.model(data, hidden)

                hidden = self._repackage_hidden(hidden)

                if args.adaptive:
                    total_loss += len(data) * loss

                else:
                    total_loss += len(data) * self.criterion(output, targets).item()
        
        return total_loss / (len(data_source) - 1)
    
    def _repackage_hidden(self, h):
        """Wraps hidden states in new Tensors, to detach them from their history."""
        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(self._repackage_hidden(v) for v in h)
    
    def fit(self):
        args = self.args
        best_val_loss = None
        
        try:
            if not args.no_log:
                name = f'b{args.batch_size}_lr{args.lr}_L{args.nlayers}_h{args.nhid}_em{args.emsize}_drp{args.rnn_dropout}_bptt{args.bptt}'
                wandb.init(name=name, project="5m_line_shuffled")
                wandb.config.update(args)

            for epoch in range(1, args.epochs+1):
                epoch_start_time = time.time()
                self._train(epoch)
                val_loss = self._evaluate(self.val_data)
                print(val_loss)
                val_ppl = math.exp(val_loss)
                print('-' * 89)
                print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                        'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                                   val_loss, val_ppl))
                print('-' * 89)

                if not args.no_log:
                    wandb.log({'val_ppl': val_ppl, 'val_loss': val_loss})

                # Save the model if the validation loss is the best we've seen so far.
                if not args.no_save:
                    # create the destination directory if it doesn't exist
                    if not os.path.exists(args.save):
                        os.mkdir(args.save)

                    # check if the current loss is the best validation loss
                    if not best_val_loss or val_loss < best_val_loss:
                        best_val_loss = val_loss

                        # save the best model
                        print('saving model...')
                        torch.save(self.model, f'{args.save}/best_model.pt')

                        # also save the checkpoint for the best model
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'val_loss': val_loss,
                            'val_ppl': val_ppl,
                            'vocabulary': self.vocab_cache
                        }, f'{args.save}/best_model_checkpoint.pt')
                    else:
                        # this saves the checkpoint for every epoch
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'val_loss': val_loss,
                            'val_ppl':val_ppl,
                            'vocabulary': self.vocab_cache
                        }, f'{args.save}/checkpoint.pt')

                print('-'*70)
                print()
                self.scheduler.step(val_loss)

        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')
            
            # Run on test data.
            # this is not working for some reason :(
            # test_loss = self._evaluate(self.test_data[0])
            # print('=' * 89)
            # print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
            #    test_loss, math.exp(test_loss)))
            # print('=' * 89)
