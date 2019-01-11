# # Implementing CLSM

# ## Purpose
# The purpose of this notebook is to implement Microsoft's [Convolutional Latent Semantic Model](http://www.iro.umontreal.ca/~lisa/pointeurs/ir0895-he-2.pdf) on our dataset.
# 
# ## Inputs
# - This notebook requires *wiki-pages* from the FEVER dataset as an input.

# ## Preprocessing Data

import argparse
import os
import pickle
import time
from multiprocessing import cpu_count
from sys import argv
from parallel import DataParallelModel, DataParallelCriterion
import parallel

import joblib
import pytorch_utils as putils
import nltk
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from hyperdash import Experiment, monitor
from joblib import Parallel, delayed
from scipy import sparse
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import recall_score
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm, tqdm_notebook

import cdssm
import pytorch_data_loader
import utils
from logger import Logger

torch.backends.cudnn.benchmark=True
nltk.data.path.append('/usr/users/mnadeem/nltk_data/')

def parse_args():
    parser = argparse.ArgumentParser(description='Learning the optimal convolution for network.')
    parser.add_argument("--batch-size", type=int, help="Number of queries per batch.", default=1)
    parser.add_argument("--model", help="Loading a pretrained model.", default=None)
    parser.add_argument("--data-batch-size", type=int, help="Number of examples per query.", default=8)
    parser.add_argument("--learning-rate", type=float, help="Learning rate for model.", default=1e-3)
    parser.add_argument("--epochs", type=int, help="Number of epochs to learn for.", default=3)
    parser.add_argument("--data", help="Folder dataset to load file from.", default="data/large/train.pkl")
    parser.add_argument("--sparse-evidences", default=False, action="store_true")
    return parser.parse_args()

def run(args, train, sparse_evidences, claims_dict):
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.learning_rate
    DATA_BATCH_SIZE = args.data_batch_size
    NUM_EPOCHS = args.epochs
    MODEL = args.model
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    logger = Logger('./logs/{}'.format(time.localtime()))

    model = cdssm.CDSSM()
    model = model.to(device)
    if torch.cuda.device_count() > 0:
      print("Let's use", torch.cuda.device_count(), "GPU(s)!")
      model = DataParallelModel(model)
    print("Created model with {:,} parameters.".format(putils.count_parameters(model)))

    if MODEL:
      model.load_state_dict(torch.load(MODEL))

    print("Created dataset...")
    train_size = int(len(train) * 0.8)
    #test = int(len(train) * 0.5)
    train_dataset = pytorch_data_loader.WikiDataset(train[:train_size], claims_dict, data_batch_size=DATA_BATCH_SIZE, sparse_evidences=sparse_evidences) 
    val_dataset = pytorch_data_loader.WikiDataset(train[train_size:], claims_dict, data_batch_size=DATA_BATCH_SIZE, sparse_evidences=sparse_evidences) 
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=0, shuffle=True, collate_fn=pytorch_data_loader.PadCollate())
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=0, shuffle=True, collate_fn=pytorch_data_loader.PadCollate())

    # Loss and optimizer
    criterion = torch.nn.BCEWithLogitsLoss()
    if torch.cuda.device_count() > 0:
        print("Let's parallelize the backward pass...")
        criterion = DataParallelCriterion(criterion)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    OUTPUT_FREQ = max(int((len(train_dataset)/BATCH_SIZE)*0.02), 20) 
    parameters = {"batch size": BATCH_SIZE, "epochs": NUM_EPOCHS, "learning rate": LEARNING_RATE, "optimizer": optimizer.__class__.__name__, "loss": criterion.__class__.__name__, "training size": train_size, "data batch size": DATA_BATCH_SIZE, "data": args.data, "sparse_evidences": args.sparse_evidences}
    exp_params = {}
    exp = Experiment("CLSM V2")

    model_checkpoint_dir = "models/saved_model" 
    for key, value in parameters.items():
        exp_params[key] = exp.param(key, value) 

        if type(value)==str:
            value = value.replace("/", "-")
        model_checkpoint_dir += "_{}-{}".format(key.replace(" ", "_"), value)

    print("Training...")
    beginning_time = 0.0
    best_accuracy = torch.tensor(0.0, dtype=torch.float)

    for epoch in range(NUM_EPOCHS):
        beginning_time = time.time()
        mean_train_acc = 0.0
        train_running_loss = 0.0
        train_running_accuracy = 0.0
        val_running_accuracy = 0.0
        model.train()

        for train_batch_num, inputs in enumerate(train_dataloader):
            claims_tensors, claims_text, evidences_tensors, evidences_text, labels = inputs  

            #claims = claims.to(device).float()
            #evidences = evidences.to(device).float()
            #labels = labels.to(device)

            y_pred = model(claims_tensors, evidences_tensors)
            y = (labels).float()
            y = y.unsqueeze(0)
            y = y.unsqueeze(0)
            print(y_pred[0].shape, y.shape)
            loss = criterion(y_pred, y)
            y_pred = parallel.gather(y_pred, 0)

            y_pred = y_pred.squeeze()
            y = y.squeeze()
            y = y.view(-1)
            y_pred = y_pred.view(-1)


            predictions = torch.sigmoid(y_pred).round()
            accuracy = (y==predictions).cuda().float()
            accuracy = accuracy.mean()
            train_running_accuracy += accuracy.item()
            mean_train_acc += accuracy.item()
            train_running_loss += loss.item()

            if (train_batch_num % OUTPUT_FREQ)==0 and train_batch_num>0:
                elapsed_time = time.time() - beginning_time
                print("[{}:{}:{:3f}s] training loss: {}, training accuracy: {}, training recall: {}".format(epoch, train_batch_num / (len(train_dataset)/BATCH_SIZE), elapsed_time, train_running_loss/OUTPUT_FREQ, train_running_accuracy/OUTPUT_FREQ, recall_score(y.cpu().detach().numpy(), predictions.cpu().detach().numpy())))

                # 1. Log scalar values (scalar summary)
                info = { 'train_loss': train_running_loss/OUTPUT_FREQ, 'train_accuracy': train_running_accuracy/OUTPUT_FREQ }

                #for tag, value in info.items():
                #    exp.metric(tag, value, log=False)
                #    logger.scalar_summary(tag, value, train_batch_num+1)

                ## 2. Log values and gradients of the parameters (histogram summary)
                #for tag, value in model.named_parameters():
                #    tag = tag.replace('.', '/')
                #    logger.histo_summary(tag, value.detach().cpu().numpy(), train_batch_num+1)
                #    logger.histo_summary(tag+'/grad', value.grad.detach().cpu().numpy(), train_batch_num+1)

                train_running_loss = 0.0
                beginning_time = time.time() 
                train_running_accuracy = 0.0
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            del loss
            del accuracy

        print("Running validation...")
        model.eval()
        for val_batch_num, val_inputs in enumerate(val_dataloader):
            claims, evidences, labels = inputs  

            claims = claims.to(device) 
            evidences = evidences.to(device).float()
            claims = claims.cuda().float()
            evidences = evidences.cuda().float()
            labels = labels.to(device)
            labels = labels.cuda()

            y_pred = model(claims, evidences)

            y = (labels)
            y_pred = y_pred.squeeze()
            y = y.squeeze()
            y = y.view(-1)
            y_pred = y_pred.view(-1)

            predictions = F.sigmoid(y_pred).round()
            accuracy = (y==predictions).to(device)
            accuracy = accuracy.float().mean()
            val_running_accuracy += accuracy.item()

            if (val_batch_num % OUTPUT_FREQ)==0 and val_batch_num>0:
                print("[{}:{}]  accuracy: {}, recall: {}".format(epoch, val_batch_num / (len(val_dataset)/BATCH_SIZE), val_running_accuracy/OUTPUT_FREQ, recall_score(y.detach().cpu().numpy(), predictions.detach().cpu().numpy())))

                # 1. Log scalar values (scalar summary)
                #info = { 'val_accuracy': val_running_accuracy/OUTPUT_FREQ }

                #for tag, value in info.items():
                #    exp.metric(tag, value, log=False)
                #    logger.scalar_summary(tag, value, val_batch_num+1)

                ## 2. Log values and gradients of the parameters (histogram summary)
                #for tag, value in model.named_parameters():
                #    tag = tag.replace('.', '/')
                #    logger.histo_summary(tag, value.detach().cpu().numpy(), val_batch_num+1)
                #    logger.histo_summary(tag+'/grad', value.grad.detach().cpu().numpy(), val_batch_num+1)

                val_running_accuracy = 0.0

        train_acc = torch.tensor((mean_train_acc)/len(train_dataloader), dtype=torch.float)
        print("[{}] mean accuracy: {}".format(epoch, train_acc))
        best_accuracy = torch.tensor(max(train_acc.cpu().numpy(), best_accuracy.cpu().numpy()))
        is_best = bool(train_acc >= best_accuracy)
        
        putils.save_checkpoint({"epoch": epoch, "state_dict": model.state_dict(), "best_accuracy": best_accuracy}, is_best, filename="accuracy_{}".format(train_acc.cpu().numpy()) + model_checkpoint_dir)

if __name__=="__main__":
    args = parse_args()

    print("Loading {}".format(args.data))
    
    fname = os.path.join(args.data,"train.pkl")
    train = joblib.load(fname)

    if args.sparse_evidences:
        print("Loading sparse evidences...")
        fname = os.path.join(args.data, "evidence.pkl")
        sparse_evidences = joblib.load(fname)
    else:
        sparse_evidences = None

    try:
        claims_dict
    except:
        print("Loading claims data...")
        claims_dict = joblib.load("claims_dict.pkl")

    torch.multiprocessing.set_start_method("fork", force=True)
    run(args, train, sparse_evidences, claims_dict)
