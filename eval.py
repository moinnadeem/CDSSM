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
from hyperdash import Experiment, monitor
from scipy import sparse
from sys import argv
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm, tqdm_notebook

import cdssm
import pytorch_data_loader
import argparse
import utils

torch.backends.cudnn.benchmark=True
nltk.data.path.append('/usr/users/mnadeem/nltk_data/')

def parse_args():
    parser = argparse.ArgumentParser(description='Learning the optimal convolution for network.')
    parser.add_argument("--batch-size", type=int, help="Number of queries per batch.", default=1)
    parser.add_argument("--data-batch-size", type=int, help="Number of examples per query.", default=8)
    parser.add_argument("--learning-rate", type=float, help="Learning rate for model.", default=1e-3)
    parser.add_argument("--epochs", type=int, help="Number of epochs to learn for.", default=3)
    parser.add_argument("--data", help="Training dataset to load file from.", default="shared_task_dev.pkl")
    parser.add_argument("--model", help="Model to evaluate.") 
    return parser.parse_args()

@monitor("CLSM Test")
def run():
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.learning_rate
    DATA_BATCH_SIZE = args.data_batch_size
    NUM_EPOCHS = args.epochs
    MODEL = args.model

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    logger = Logger('./logs/{}'.format(time.localtime()))

    print("Created model...")
    model = cdssm.CDSSM()
    model = model.cuda()
    model = model.to(device)
    if torch.cuda.device_count() > 0:
      print("Let's use", torch.cuda.device_count(), "GPU(s)!")
      model = nn.DataParallel(model)
    model.load_state_dict(torch.load(MODEL))

    print("Created dataset...")
    dataset = pytorch_data_loader.WikiDataset(test, claims_dict, data_batch_size=DATA_BATCH_SIZE, testFile="shared_task_dev.jsonl") 
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=3, shuffle=True, collate_fn=pytorch_data_loader.variable_collate)

    OUTPUT_FREQ = int((len(dataset)/BATCH_SIZE)*0.02) 
    criterion = torch.nn.BCEWithLogitsLoss()

    parameters = {"batch size": BATCH_SIZE, "loss": criterion.__class__.__name__, "data batch size": DATA_BATCH_SIZE, "data": args.data}
    exp_params = {}
    exp = Experiment("CLSM V2")
    for key, value in parameters.items():
       exp_params[key] = exp.param(key, value) 

    true = []
    pred = []
    print("Training...")
    test_running_accuracy = 0.0
    test_running_loss = 0.0
    num_batches = 0
    for batch_num, inputs in enumerate(dataloader):
        num_batches += 1
        claims, evidences, labels = inputs  

        claims = claims.to(device) 
        evidences = evidences.to(device) 
        claims = claims.cuda()
        evidences = evidences.cuda()
        labels = labels.to(device)
        labels = labels.cuda()

        y_pred = model(claims, evidences)

        y = (labels).float()
        y_pred = y_pred.squeeze()
        y = y.squeeze()
        y = y.view(-1)
        y_pred = y_pred.view(-1)
        bin_acc = F.sigmoid(y_pred).round()

        loss = criterion(y_pred, y)

        true.extend(y.tolist())
        pred.extend(bin_acc.tolist())

        accuracy = (y==bin_acc).float().mean()
        test_running_accuracy += accuracy.item()
        test_running_loss += loss.item() 

        
        if batch_num % OUTPUT_FREQ==0 and batch_num>0:
            print("[{}]: {}".format(batch_num, test_running_accuracy / num_batches))

            # 1. Log scalar values (scalar summary)
            info = { 'test_loss': test_running_loss/OUTPUT_FREQ, 'test_accuracy': test_running_accuracy/OUTPUT_FREQ }

            for tag, value in info.items():
                exp.metric(tag, value, log=False)
                logger.scalar_summary(tag, value, batch_num+1)

            # 2. Log values and gradients of the parameters (histogram summary)
            for tag, value in model.named_parameters():
                tag = tag.replace('.', '/')
                logger.histo_summary(tag, value.data.cpu().numpy(), batch_num+1)

            test_running_loss = 0.0
            test_running_accuracy = 0.0

    print("Final accuracy: {}".format(test_running_accuracy / num_batches))
    filename = "predicted_labels"
    for key, value in parameters.items():
        filename += "_{}-{}".format(key.replace(" ", "_"), value)

    joblib.dump({"true": true, "pred": pred}, filename)

if __name__=="__main__":
    args = parse_args()

    print("Loading {}".format(args.data))
    test = joblib.load(args.data)

    try:
        claims_dict
    except:
        print("Loading validation claims data...")
        claims_dict = joblib.load("val_dict.pkl")

    torch.multiprocessing.set_start_method("fork", force=True)
    run()
