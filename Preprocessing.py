import gc
import glob
import itertools
import os
import pickle

import joblib
import keras
import numpy as np
from scipy import sparse
from tqdm import tqdm_notebook
from deep_semantic_similarity_model import create_model

import utils

train = joblib.load("train.pkl")

encoder = utils.ClaimEncoder()

claims, labels, article_list, claim_set, claim_to_article = utils.extract_fever_jsonl_data("../train.jsonl")

def stack_uneven(arrays, fill_value=0.):
        '''
        Fits arrays into a single numpy array, even if they are
        different sizes. `fill_value` is the default value.

        Args:
                arrays: list of np arrays of various sizes
                    (must be same rank, but not necessarily same size)
                fill_value (float, optional):

        Returns:
                np.ndarray
        '''
        sizes = [a.shape for a in arrays]
        max_sizes = np.max(list(zip(*sizes)), -1)
        # The resultant array has stacked on the first dimension
        result = np.full((len(arrays),) + tuple(max_sizes), fill_value)
        for i, a in enumerate(arrays):
          # The shape of this array `a`, turned into slices
          slices = tuple(slice(0,s) for s in sizes[i])
          # Overwrite a block slice of `result` with this array `a`
          result[i][slices] = a
        return result

class DataGenerator(keras.utils.Sequence):
    """
    Generates data with batch size of 1 sample for the purposes of training our model.
    """
    def __init__(self, data, batch_size=32, split=None):
        """
            Sets the initial arguments and creates
            an indicies array to randomize the dataset
            between epochs
        """
        if split:            
            self.indicies = split
        else:
            self.indicies = list(range(len(data)))
        self.data = data
        encoder = utils.ClaimEncoder()
        self.batch_size = batch_size
        _, _, _, _, self.claim_to_article = utils.extract_fever_jsonl_data("../train.jsonl")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.get_item(index)
    
    def get_item(self, index):            
        d = self.data[index]
        claim = sparse.vstack(encoder.tokenize_claim(d['claim'])).toarray()
        evidences = []
        ys = []
        for e in d['evidence']:
            processed = utils.preprocess_article_name(e.split("http://wikipedia.org/wiki/")[1])
            evidences.append(sparse.vstack(encoder.tokenize_claim(processed)).toarray())
            if processed in self.claim_to_article[d['claim']]:
                ys.append(1)
            else:
                ys.append(0)

        evidences = stack_uneven(evidences)
        gc.collect()
        return {"claim":np.repeat(claim[np.newaxis, :, :], len(evidences), axis=0), "document":evidences}, np.array(ys)
    
    def on_epoch_end(self):
        #np.random.shuffle(self.indicies)
        pass

gen = DataGenerator(train)
model = create_model()

import gc
gc.collect()

model.fit_generator(gen, workers=1, max_queue_size=10, use_multiprocessing=False)

model.save("preprocessed_model.h5")
