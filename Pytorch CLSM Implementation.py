# # Implementing CLSM

# ## Purpose
# The purpose of this notebook is to implement Microsoft's [Convolutional Latent Semantic Model](http://www.iro.umontreal.ca/~lisa/pointeurs/ir0895-he-2.pdf) on our dataset.
# 
# ## Inputs
# - This notebook requires *wiki-pages* from the FEVER dataset as an input.

# ## Preprocessing Data

import cdssm
import numpy as np
import nltk
import utils
import pickle
from tqdm import tqdm

from scipy import sparse
from joblib import Parallel, delayed
from multiprocessing import cpu_count
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from tqdm import tqdm_notebook

import imp
import torch 
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_data_loader
import cdssm
import joblib
torch.backends.cudnn.benchmark=True
nltk.data.path.append('/usr/users/mnadeem/nltk_data/')


def run():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    BATCH_SIZE = 8

    imp.reload(pytorch_data_loader)
    imp.reload(cdssm)

    print("Created model...")
    model = cdssm.CDSSM()
    model = model.cuda()
    model = model.to(device)
    #if torch.cuda.device_count() > 0:
    #  print("Let's use", torch.cuda.device_count(), "GPU(s)!")
    #  model = nn.DataParallel(model)

    print("Created dataset...")
    dataset = pytorch_data_loader.WikiDataset(train, claims_dict, data_batch_size=2) 
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

    # Loss and optimizer
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)

    OUTPUT_FREQ = 250 

    print("Training...")
    running_loss = 0.0
    for batch_num, inputs in tqdm(enumerate(dataloader), total=len(dataset)/BATCH_SIZE):
        claims, evidences, labels = inputs  

        claims = claims.to(device) 
        evidences = evidences.to(device) 
        claims = claims.cuda()
        evidences = evidences.cuda()
        labels = labels.to(device)
        labels = labels.cuda()

        #claim = claim.to(device) 
        #evidence = evidence.to(device) 
        #labels = torch.FloatTensor(labels).to(device) 

        #claim = claim.to_dense()
        #evidence = evidence.to_dense()
        #print(claim.device)
        #print(evidence.device)
        y_pred = model(claims, evidences)

        #y = Variable(torch.from_numpy(np.array(labels[i])).long())
        y = (labels)
        y_pred = y_pred.squeeze()
        y = y.squeeze()
        y = y.view(-1)
        y_pred = y_pred.view(-1)
        loss = criterion(y_pred, y)

        running_loss += loss.item()
        if (batch_num % OUTPUT_FREQ)==0:
            #print(y_pred, y)
            print(batch_num, running_loss/OUTPUT_FREQ)
            running_loss = 0.0
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

if __name__=="__main__":
    try:
        train 
    except:
        print("Loading training data..., (Line 40)")
        train = joblib.load("train.pkl")

    try:
        claims_dict
    except:
        print("Loading claims data...")
        claims_dict = joblib.load("claims_dict.pkl")
    torch.multiprocessing.set_start_method("fork", force=True)
    print("Testing...")
    run()
