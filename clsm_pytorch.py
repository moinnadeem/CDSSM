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
    parser.add_argument("--data", help="Training dataset to load file from.", default="train.pkl")
    return parser.parse_args()

@monitor("CLSM Test")
def run():
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.learning_rate
    DATA_BATCH_SIZE = args.data_batch_size
    NUM_EPOCHS = args.epochs
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    logger = Logger('./logs/{}'.format(time.localtime()))

    print("Created model...")
    model = cdssm.CDSSM()
    model = model.cuda()
    model = model.to(device)
    #model.load_state_dict(torch.load("saved_model"))
    #if torch.cuda.device_count() > 0:
    #  print("Let's use", torch.cuda.device_count(), "GPU(s)!")
    #  model = nn.DataParallel(model)

    print("Created dataset...")
    train_size = int(len(train) * 0.8)
    #test = int(len(train) * 0.5)
    train_dataset = pytorch_data_loader.WikiDataset(train[:train_size], claims_dict, data_batch_size=DATA_BATCH_SIZE) 
    val_dataset = pytorch_data_loader.WikiDataset(train[train_size:], claims_dict, data_batch_size=DATA_BATCH_SIZE) 
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=5, shuffle=True, collate_fn=pytorch_data_loader.variable_collate)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=5, shuffle=True, collate_fn=pytorch_data_loader.variable_collate)

    # Loss and optimizer
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    OUTPUT_FREQ = int((len(train_dataset)/BATCH_SIZE)*0.02) 
    parameters = {"batch size": BATCH_SIZE, "epochs": NUM_EPOCHS, "learning rate": LEARNING_RATE, "optimizer": optimizer.__class__.__name__, "loss": criterion.__class__.__name__, "training size": train_size, "data batch size": DATA_BATCH_SIZE, "data": args.data}
    exp_params = {}
    exp = Experiment("CLSM V2")
    for key, value in parameters.items():
       exp_params[key] = exp.param(key, value) 

    print("Training...")
    train_running_loss = 0.0
    train_running_accuracy = 0.0
    val_running_loss = 0.0
    val_running_accuracy = 0.0

    for epoch in range(NUM_EPOCHS):
        for train_batch_num, inputs in enumerate(train_dataloader):
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

            loss = criterion(y_pred, y)

            bin_acc = F.sigmoid(y_pred).round()
            accuracy = (y==bin_acc).float().mean()
            train_running_accuracy += accuracy.item()
            train_running_loss += loss.item()

            if (train_batch_num % OUTPUT_FREQ)==0 and train_batch_num>0:
                print("[{}:{}] training loss: {}, training accuracy: {}".format(epoch, train_batch_num / (len(train_dataset)/BATCH_SIZE), train_running_loss/OUTPUT_FREQ, train_running_accuracy/OUTPUT_FREQ))

                # 1. Log scalar values (scalar summary)
                info = { 'train_loss': train_running_loss/OUTPUT_FREQ, 'train_accuracy': train_running_accuracy/OUTPUT_FREQ }

                for tag, value in info.items():
                    exp.metric(tag, value, log=False)
                    logger.scalar_summary(tag, value, train_batch_num+1)

                # 2. Log values and gradients of the parameters (histogram summary)
                for tag, value in model.named_parameters():
                    tag = tag.replace('.', '/')
                    logger.histo_summary(tag, value.data.cpu().numpy(), train_batch_num+1)
                    logger.histo_summary(tag+'/grad', value.grad.data.cpu().numpy(), train_batch_num+1)

                train_running_loss = 0.0
                train_running_accuracy = 0.0
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("Running validation...")
        for val_batch_num, val_inputs in enumerate(val_dataloader):
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
            accuracy = (y==bin_acc).float().mean()
            val_running_accuracy += accuracy.item()
            val_running_loss += loss.item()

            if (val_batch_num % OUTPUT_FREQ)==0 and val_batch_num>0:
                print("[{}:{}] loss: {}, accuracy: {}".format(epoch, val_batch_num / (len(val_dataset)/BATCH_SIZE), val_running_loss/OUTPUT_FREQ, val_running_accuracy/OUTPUT_FREQ))

                # 1. Log scalar values (scalar summary)
                info = { 'val_loss': val_running_loss/OUTPUT_FREQ, 'val_accuracy': val_running_accuracy/OUTPUT_FREQ }

                for tag, value in info.items():
                    exp.metric(tag, value, log=False)
                    logger.scalar_summary(tag, value, val_batch_num+1)

                # 2. Log values and gradients of the parameters (histogram summary)
                for tag, value in model.named_parameters():
                    tag = tag.replace('.', '/')
                    logger.histo_summary(tag, value.data.cpu().numpy(), val_batch_num+1)
                    logger.histo_summary(tag+'/grad', value.grad.data.cpu().numpy(), val_batch_num+1)

                val_running_loss = 0.0
                val_running_accuracy = 0.0

    model_str = "saved_model" 
    for key, value in parameters.items():
        model_str += "_{}-{}".format(key.replace(" ", "_"), value)

    torch.save(model.state_dict(), model_str)

if __name__=="__main__":
    args = parse_args()

    print("Loading {}".format(args.data))
    train = joblib.load(args.data)

    try:
        claims_dict
    except:
        print("Loading claims data...")
        claims_dict = joblib.load("new_claims_dict.pkl")

    torch.multiprocessing.set_start_method("fork", force=True)
    run()
