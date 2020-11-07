# Language Model PyTorch
Train language models from scratch and finetune them.

## How to train
```
python main.py --cutoffs='5000 15000' --cuda --data='data/wikitext-2' --save='wikitext_ckpt' --batch_size=60 --bptt=50 --lr=30 --no_log --no_save
```
