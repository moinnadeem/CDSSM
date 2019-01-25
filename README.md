# Convolutional Deep Semantic Similarity Model

This repository implements [CDSSM](http://www.iro.umontreal.ca/~lisa/pointeurs/ir0895-he-2.pdf), which seeks to rank documents given some inputs.

Be sure to read the paper before continuing, as my implementation has begun to differ from the paper.

## Instructions
*Entry point:* clsm_pytorch.py

### Relevant Files
A lot of these evidences have been preprocessed into pickle format; the following files / parameters are useful to speed up compute / training time.

- __claims_dict.pkl__ is used to get a mapping of claims to preprocessed representations, similarly.
- __feature_encoder.pkl__ and __encoder.pkl__ are used to preprocess text on the fly. They contain a mapping of the trigrams to one-hot vectors, and characters to trigrams respectively. Make sure you have these.
- The __data__ folder contains all data input needed to run the documents.

These documents are stored in _/usr/users/mnadeem/CDSSM_github_. The following commands should be able to copy them locally:

```
cp /usr/users/mnadeem/CDSSM_github/claims_dict.pkl .
cp /usr/users/mnadeem/CDSSM_github/feature_encoder.pkl .
cp /usr/users/mnadeem/CDSSM_github/encoder.pkl .
cp -r /usr/users/mnadeem/CDSSM_github/data/ .
```

To run: `python3 clsm_pytorch.py --data data/large ARGS`

### Speedups
- Running it with the _--sparse-evidences_ flag: this loads a dictionary of preprocessed matricies rather than building it on runtime; that speeds up training significantly.
- claims_dict.pkl is used to get a mapping of claims to preprocessed representations, similarly.

### Notes:
- I normally run it on a Titan X, and each 2% of the batch takes 20-30s, or around 16 minutes per epoch.
- I normally use 1 GPU, and haven't noticed a performance speedup with several GPUs.
