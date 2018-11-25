#!/usr/bin/env python
# coding: utf-8

# # Implementing CLSM - Keras

# ## Purpose
# The purpose of this notebook is to implement Microsoft's [Convolutional Latent Semantic Model](http://www.iro.umontreal.ca/~lisa/pointeurs/ir0895-he-2.pdf) in Keras, and evaluate it on our dataset.
# 
# ## Inputs
# - This notebook requires *wiki-pages* from the FEVER dataset as an input.

# ## Preprocessing Data

# In[1]:


import numpy as np
import nltk
import utils
import pickle
import gc

from scipy import sparse
from joblib import Parallel, delayed
from multiprocessing import cpu_count
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from tqdm import tqdm_notebook


claims, labels, article_list, claim_set, claim_to_article = utils.extract_fever_jsonl_data("../train.jsonl")

with open("train.pkl", "rb") as f:
    train_dict = pickle.load(f)

def generate_all_tokens(arr):
    all_tokens = []
    for unprocessed_claim in tqdm_notebook(arr):
        c = utils.preprocess_article_name(unprocessed_claim)
        c = "! {} !".format(c)
        for word in c.split():
            letter_tuples = list(nltk.ngrams("#" + word + "#", 3))
            letter_grams = []
            for l in letter_tuples:
                letter_grams.append("".join(l))
            all_tokens.extend(letter_grams)
    return all_tokens

processed_claims = generate_all_tokens(claims)
all_evidence = []
for query in train_dict:
    all_evidence.extend(query['evidence'])
    
processed_claims.extend(generate_all_tokens(list(set(all_evidence))))

possible_tokens = list(set(processed_claims))

encoder = LabelEncoder()
encoder.fit(np.array(sorted(possible_tokens)))

del processed_claims
del possible_tokens
gc.collect()

feature_encoder = {}
for idx, e in tqdm_notebook(enumerate(encoder.classes_)):
    feature_encoder[e] = idx


# In[12]:


def tokenize_claim(c, enc):
    """
    Input: a string that represents a single claim
    Output: a list of 3x|vocabulary| arrays that has a 1 where the letter-gram exists.
    """
    encoded_vector = []
    c = utils.preprocess_article_name(c)
    c = "! {} !".format(c)
    for ngram in nltk.ngrams(nltk.word_tokenize(c), 3):
        arr = sparse.lil_matrix((3, len(enc.__dict__['classes_'])))
        for idx, word in enumerate(ngram):
            for letter_gram in nltk.ngrams("#" + word + "#", 3):
                s = "".join(letter_gram)
                letter_idx = feature_encoder[s]
                arr[idx, letter_idx] = 1
        encoded_vector.append(arr)
    return encoded_vector


all_data = []

article_set = set(article_list)

def process_claim(idx):
    J = 399
    data = {}
    articles = [utils.preprocess_article_name(i.split("http://wikipedia.org/wiki/")[1]) for i in train_dict[idx]['evidence']]
    data['claim'] = tokenize_claim(train_dict[idx]['claim'], encoder)
    true_article = claim_to_article[train_dict[idx]['claim']][0]
    true_article_idx = articles.index(true_article)
    data['positive_article'] = tokenize_claim(true_article, encoder)
    negative_articles = articles[:true_article_idx] + articles[true_article_idx+1:]
    negative_articles = [tokenize_claim(i, encoder) for i in negative_articles]
    for i in range(J):
        data['negative_article_{}'.format(i)] = negative_articles[i]
    return data

all_data = utils.parallel_process(range(len(train_dict)), process_claim, n_jobs=12)

with open("all_data.pkl_lucene", "wb") as f:
    pickle.dump(all_data, f)
