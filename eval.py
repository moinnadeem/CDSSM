# # Implementing CLSM

# ## Purpose
# The purpose of this notebook is to implement Microsoft's [Convolutional Latent Semantic Model](http://www.iro.umontreal.ca/~lisa/pointeurs/ir0895-he-2.pdf) on our dataset.
# 
# ## Inputs
# - This notebook requires *wiki-pages* from the FEVER dataset as an input.

# ## Preprocessing Data

import pickle
from multiprocessing import cpu_count

import joblib
import nltk
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from joblib import Parallel, delayed
from logger import Logger
from scipy import sparse
from sys import argv
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm, tqdm_notebook

import cdssm
import pytorch_data_loader
import utils

torch.backends.cudnn.benchmark=True
nltk.data.path.append('/usr/users/mnadeem/nltk_data/')

def run():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    BATCH_SIZE = 8 
    NUM_EPOCHS = 1

    print("Created model...")
    model = cdssm.CDSSM()
    model = model.cuda()
    model = model.to(device)
    model.load_state_dict(torch.load("saved_model_more_examples"))
    #if torch.cuda.device_count() > 0:
    #  print("Let's use", torch.cuda.device_count(), "GPU(s)!")
    #  model = nn.DataParallel(model)

    print("Created dataset...")
    train_size = int(len(train) * 0.8)
    dataset = pytorch_data_loader.WikiDataset(train[train_size:], claims_dict, data_batch_size=8) 
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=3, shuffle=True, collate_fn=pytorch_data_loader.variable_collate)

    LETTER_GRAM_SIZE = 3 # See section 3.2.
    WINDOW_SIZE = 3 # See section 3.2.
    TOTAL_LETTER_GRAMS = int(29243) # Determined from data. See section 3.2.
    WORD_DEPTH = WINDOW_SIZE * TOTAL_LETTER_GRAMS # See equation (1).

    # WORD_DEPTH = 1000
    K = 300 # Dimensionality of the max-pooling layer. See section 3.4.
    L = 128 # Dimensionality of latent semantic space. See section 3.5.
    J = 4 # Number of random unclicked documents serving as negative examples for a query. See section 4.
    FILTER_LENGTH = 1 # We only consider one time step for convolutions.

    OUTPUT_FREQ = int((len(dataset)/BATCH_SIZE)*0.02) 

    true = []
    pred = []
    print("Training...")
    running_accuracy = 0.0
    num_batches = 0
    for batch_num, inputs in tqdm(enumerate(dataloader), total=(len(dataset)/BATCH_SIZE)):
        num_batches += 1
        claims, evidences, labels = inputs  

        claims = claims.to(device) 
        evidences = evidences.to(device) 
        claims = claims.cuda()
        evidences = evidences.cuda()
        labels = labels.to(device)
        labels = labels.cuda()

        y_pred = model(claims, evidences)

        y = (labels)
        y_pred = y_pred.squeeze()
        y = y.squeeze()
        y = y.view(-1)
        y_pred = y_pred.view(-1)
        bin_acc = F.sigmoid(y_pred).round()

        true.extend(y.tolist())
        pred.extend(bin_acc.tolist())

        accuracy = (y==bin_acc).float().mean()
        running_accuracy += accuracy
    print("Final accuracy: {}".format(running_accuracy / num_batches))
    joblib.dump({"true": true, "pred": pred}, "predicted_labels.pkl")

if __name__=="__main__":
    try:
        train 
    except:
        s = ""
        f = ""
        if len(argv)==1:
            s = "Loading large training data.."
            f = "train.pkl"
        else:
            if argv[1]=="-m":
                s = "Loading medium training data..."
                f = "train_medium.pkl"
            elif argv[1]=="-m":
                s = "Loading small training data..."
                f = "train_small.pkl"

        print(s)
        train = joblib.load(f)

    try:
        claims_dict
    except:
        print("Loading claims data...")
        claims_dict = joblib.load("claims_dict.pkl")

    torch.multiprocessing.set_start_method("fork", force=True)
    run()
