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
from comet_ml import Experiment
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

from joblib import Parallel, delayed
from scipy import sparse
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import recall_score, classification_report, accuracy_score
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
    parser.add_argument("--batch-size", type=int, help="Number of queries per batch.", default=10)
    parser.add_argument("--model", help="Loading a pretrained model.", default=None)
    parser.add_argument("--data-sampling", type=int, help="Number of examples per query.", default=3)
    parser.add_argument("--no-randomize", default=True, action="store_false")
    parser.add_argument("--learning-rate", type=float, help="Learning rate for model.", default=1e-3)
    parser.add_argument("--epochs", type=int, help="Number of epochs to learn for.", default=15)
    parser.add_argument("--data", help="Folder dataset to load file from.", default="data/large")
    parser.add_argument("--print", default=False, action="store_true", help="Whether to print predicted labels or not.")
    parser.add_argument("--sparse-evidences", default=False, action="store_true")
    return parser.parse_args()

def run(args, train, sparse_evidences, claims_dict):
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.learning_rate
    DATA_SAMPLING = args.data_sampling
    NUM_EPOCHS = args.epochs
    MODEL = args.model
    RANDOMIZE = args.no_randomize
    PRINT = args.print
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    logger = Logger('./logs/{}'.format(time.localtime()))

    if MODEL:
        print("TEMPORARY change to loading!")
        model = torch.load(MODEL)
        model.load_state_dict(torch.load(MODEL).state_dict())
    else:
        model = cdssm.CDSSM()
        model = model.cuda()
        model = model.to(device)

    # model = cdssm.CDSSM()
    # model = model.cuda()
    # model = model.to(device)

    if torch.cuda.device_count() > 0:
      print("Let's use", torch.cuda.device_count(), "GPU(s)!")
      model = nn.DataParallel(model)
    
    print("Created model with {:,} parameters.".format(putils.count_parameters(model)))

    # if MODEL:
        # print("TEMPORARY change to loading!")
        # model.load_state_dict(torch.load(MODEL).state_dict())

    print("Created dataset...")
    train_size = int(len(train) * 0.80)
    #test = int(len(train) * 0.5)
    train_dataset = pytorch_data_loader.WikiDataset(train[:train_size], claims_dict, data_sampling=DATA_SAMPLING, sparse_evidences=sparse_evidences, randomize=RANDOMIZE) 
    val_dataset = pytorch_data_loader.WikiDataset(train[train_size:], claims_dict, data_sampling=DATA_SAMPLING, sparse_evidences=sparse_evidences, randomize=RANDOMIZE) 

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=0, shuffle=True, collate_fn=pytorch_data_loader.PadCollate())
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=0, shuffle=True, collate_fn=pytorch_data_loader.PadCollate())

    # Loss and optimizer
    #criterion = torch.nn.NLLLoss()
    # criterion = torch.nn.SoftMarginLoss()
    criterion = putils.ContrastiveLoss(margin=1)
    # if torch.cuda.device_count() > 0:
        # print("Let's parallelize the backward pass...")
        # criterion = DataParallelCriterion(criterion)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)

    OUTPUT_FREQ = max(int((len(train_dataset)/BATCH_SIZE)*0.02), 20) 
    parameters = {"batch size": BATCH_SIZE, "epochs": NUM_EPOCHS, "learning rate": LEARNING_RATE, "optimizer": optimizer.__class__.__name__, "loss": criterion.__class__.__name__, "training size": train_size, "data sampling rate": DATA_SAMPLING, "data": args.data, "sparse_evidences": args.sparse_evidences, "randomize": RANDOMIZE, "model": MODEL}
    experiment = Experiment(api_key="YLsW4AvRTYGxzdDqlWRGCOhee", project_name="clsm", workspace="moinnadeem")
    experiment.add_tag("train")
    experiment.log_asset("cdssm.py")
    experiment.log_dataset_info(name=args.data)
    experiment.log_parameters(parameters)

    model_checkpoint_dir = "models/saved_model" 
    for key, value in parameters.items():
        if type(value)==str:
            value = value.replace("/", "-")
        if key!="model":
            model_checkpoint_dir += "_{}-{}".format(key.replace(" ", "_"), value)

    print("Training...")
    beginning_time = time.time() 
    best_loss = torch.tensor(float("inf"), dtype=torch.float)  # begin loss at infinity

    for epoch in range(NUM_EPOCHS):
        beginning_time = time.time()
        mean_train_acc = 0.0
        train_running_loss = 0.0
        train_running_accuracy = 0.0
        model.train()
        experiment.log_current_epoch(epoch)

        with experiment.train():
            for train_batch_num, inputs in enumerate(train_dataloader):
                claims_tensors, claims_text, evidences_tensors, evidences_text, labels = inputs  

                claims_tensors = claims_tensors.cuda()
                evidences_tensors = evidences_tensors.cuda()
                labels = labels.cuda()
                #claims = claims.to(device).float()
                #evidences = evidences.to(device).float()
                #labels = labels.to(device)

                y_pred = model(claims_tensors, evidences_tensors)

                y = (labels)
                y = 1 - y
                # y = y.unsqueeze(0)
                # y = y.unsqueeze(0)
                # y_pred = parallel.gather(y_pred, 0)

                y_pred = y_pred.squeeze()
                # y = y.squeeze()

                # loss = criterion(y_pred, torch.max(y,1)[1])
                loss = criterion(y_pred, y)

                y = y.float()
                classifications = torch.norm(y_pred, p="fro", dim=1) >= 1
                classifications = classifications.to(device).float() 
                accuracy = (classifications==y).float()
                accuracy = accuracy.mean()

                train_running_accuracy += accuracy.item()
                mean_train_acc += accuracy.item()
                train_running_loss += loss.item()

                if PRINT:
                    for idx in range(len(y)): 
                        print("Claim: {}, Evidence: {}, Prediction: {}, Label: {}".format(claims_text[0], evidences_text[idx], torch.exp(y_pred[idx]), y[idx])) 

                if (train_batch_num % OUTPUT_FREQ)==0 and train_batch_num>0:
                    elapsed_time = time.time() - beginning_time
                    print("[{}:{}:{:3f}s] training loss: {}, training accuracy: {}, training recall: {}".format(epoch, train_batch_num / (len(train_dataset)/BATCH_SIZE), elapsed_time, train_running_loss/OUTPUT_FREQ, train_running_accuracy/OUTPUT_FREQ, recall_score(y.cpu().detach().numpy(), classifications.cpu().detach().numpy())))

                    # 1. Log scalar values (scalar summary)
                    info = { 'train_loss': train_running_loss/OUTPUT_FREQ, 'train_accuracy': train_running_accuracy/OUTPUT_FREQ }

                    for tag, value in info.items():
                       experiment.log_metric(tag, value, step=train_batch_num*(epoch+1))
                       logger.scalar_summary(tag, value, train_batch_num+1)

                    ## 2. Log values and gradients of the parameters (histogram summary)
                    for tag, value in model.named_parameters():
                        tag = tag.replace('.', '/')
                        logger.histo_summary(tag, value.detach().cpu().numpy(), train_batch_num+1)
                        logger.histo_summary(tag+'/grad', value.grad.detach().cpu().numpy(), train_batch_num+1)

                    train_running_loss = 0.0
                    beginning_time = time.time() 
                    train_running_accuracy = 0.0
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # del loss
        # del accuracy
        # del claims_tensors
        # del claims_text
        # del evidences_tensors
        # del evidences_text
        # del labels 
        # del y
        # del y_pred
        # torch.cuda.empty_cache()


        print("Running validation...")
        model.eval()
        pred = []
        true = []
        avg_loss = 0.0
        val_running_accuracy = 0.0
        val_running_loss = 0.0
        beginning_time = time.time()
        with experiment.validate():
            for val_batch_num, val_inputs in enumerate(val_dataloader):
                claims_tensors, claims_text, evidences_tensors, evidences_text, labels = val_inputs  

                claims_tensors = claims_tensors.cuda()
                evidences_tensors = evidences_tensors.cuda()
                labels = labels.cuda()

                y_pred = model(claims_tensors, evidences_tensors)

                y = (labels)
                y = 1-y
                # y_pred = parallel.gather(y_pred, 0)

                y_pred = y_pred.squeeze()

                #loss = criterion(y_pred, torch.max(y,1)[1])
                loss = criterion(y_pred, y)

                y = y.float()

                classifications = torch.norm(y_pred, p="fro", dim=1) >= 1.0
                classifications = classifications.to(device).float() 

                accuracy = (classifications==y).float()
                accuracy = accuracy.mean()

                true.extend(y.tolist())
                pred.extend(classifications.tolist())

                val_running_accuracy += accuracy.item()
                val_running_loss += loss.item() 
                avg_loss += loss.item()

                if (val_batch_num % OUTPUT_FREQ)==0 and val_batch_num>0:
                    elapsed_time = time.time() - beginning_time
                    print("[{}:{}:{:3f}s] validation loss: {}, accuracy: {}, recall: {}".format(epoch, val_batch_num / (len(val_dataset)/BATCH_SIZE), elapsed_time, val_running_loss/OUTPUT_FREQ, val_running_accuracy/OUTPUT_FREQ, recall_score(classifications.cpu().detach().numpy(), y.cpu().detach().numpy())))

                    # 1. Log scalar values (scalar summary)
                    info = { 'val_accuracy': val_running_accuracy/OUTPUT_FREQ }

                    for tag, value in info.items():
                       experiment.log_metric(tag, value, step=val_batch_num*(epoch+1))
                       logger.scalar_summary(tag, value, val_batch_num+1)

                    ## 2. Log values and gradients of the parameters (histogram summary)
                    for tag, value in model.named_parameters():
                       tag = tag.replace('.', '/')
                       logger.histo_summary(tag, value.detach().cpu().numpy(), val_batch_num+1)
                       logger.histo_summary(tag+'/grad', value.grad.detach().cpu().numpy(), val_batch_num+1)

                    val_running_accuracy = 0.0
                    val_running_loss = 0.0
                    beginning_time = time.time()

        # del loss
        # del accuracy
        # del claims_tensors
        # del claims_text
        # del evidences_tensors
        # del evidences_text
        # del labels 
        # del y
        # del y_pred
        # torch.cuda.empty_cache()

        accuracy = accuracy_score(true, pred) 
        print("[{}] mean accuracy: {}, mean loss: {}".format(epoch, accuracy, avg_loss / len(val_dataloader)))

        true = np.array(true).astype("int") 
        pred = np.array(pred).astype("int") 
        print(classification_report(true, pred))

        best_loss = torch.tensor(min(avg_loss / len(val_dataloader), best_loss.cpu().numpy()))
        is_best = bool((avg_loss / len(val_dataloader)) <= best_loss)
        
        putils.save_checkpoint({"epoch": epoch, "model": model, "best_loss": best_loss}, is_best, filename="{}_loss_{}".format(model_checkpoint_dir, best_loss.cpu().numpy()))

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

    # torch.multiprocessing.set_start_method("spawn", force=True)
    run(args, train, sparse_evidences, claims_dict)
