# ConveRT for Response Selection

Transformer-based model for selecting response based on similarity - where I have made modifications based on [codertimo's work](https://github.com/codertimo/ConveRT-pytorch)
This repository is an unofficial implementation for PolyAI's [ConveRT model](https://github.com/PolyAI-LDN/polyai-models)
You can use BERT instead of scratch Transformer Encoder, where I have applied [Korean DistilBERT](https://github.com/monologg/DistilKoBERT)

## Process
### 1. Preprocess
- Place train.txt file and test.txt file under '/data' 
- Sample files for both train.txt and test.txt can be found under the same directory
- The label column is a unique index for each response
- The tokenizer used here is for Korean so you will need to change if necessary
- Then, run preprocess.py 

### 2. Train
- After preprocess, run train.py 
- Model or train config such as number of encoder layer, learning rate and batch size should be defined in '/src/constant.py'
- If you want to use Korean DistilBERT, with 6 layers, give a flag `-kobert True`
- Default similarity metric is `dot-product` or you can change to cosine similarity with `-loss_type cosine`