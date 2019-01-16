# # Implementing CLSM

# ## Purpose
# The purpose of this notebook is to implement Microsoft's [Convolutional Latent Semantic Model](http://www.iro.umontreal.ca/~lisa/pointeurs/ir0895-he-2.pdf) on our dataset.
# 
# ## Inputs
# - This notebook requires *wiki-pages* from the FEVER dataset as an input.

# ## Preprocessing Data

import pickle
from multiprocessing import cpu_count
import os
from parallel import DataParallelModel, DataParallelCriterion
import parallel

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
from sklearn.metrics import classification_report, accuracy_score
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
    parser.add_argument("--data-sampling", type=int, help="Number of examples per query.", default=8)
    parser.add_argument("--learning-rate", type=float, help="Learning rate for model.", default=1e-3)
    parser.add_argument("--epochs", type=int, help="Number of epochs to learn for.", default=3)
    parser.add_argument("--randomize", default=False, action="store_true")
    parser.add_argument("--data", help="Training dataset to load file from.", default="data/validation")
    parser.add_argument("--model", help="Model to evaluate.") 
    parser.add_argument("--sparse-evidences", default=False, action="store_true")
    return parser.parse_args()

@monitor("CLSM Test")
def run():
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.learning_rate
    DATA_SAMPLING = args.data_sampling
    NUM_EPOCHS = args.epochs
    MODEL = args.model
    RANDOMIZE = args.randomize

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    logger = Logger('./logs/{}'.format(time.localtime()))

    print("Created model...")
    if MODEL:
        model = torch.load(MODEL).module
    else:
        model = cdssm.CDSSM()
        model = model.cuda()
        model = model.to(device)
    if torch.cuda.device_count() > 0:
      print("Let's use", torch.cuda.device_count(), "GPU(s)!")
      model = nn.DataParallel(model)

    print("Created dataset...")
    dataset = pytorch_data_loader.ValWikiDataset(test, claims_dict, testFile="shared_task_dev.jsonl", sparse_evidences=sparse_evidences, batch_size=BATCH_SIZE) 
    dataloader = DataLoader(dataset, num_workers=0, collate_fn=pytorch_data_loader.PadCollate())

    OUTPUT_FREQ = int((len(dataset))*0.10) 
    
    parameters = {"batch size": BATCH_SIZE, "data sampling rate": DATA_SAMPLING, "data": args.data}
    exp_params = {}
    exp = Experiment("CLSM V2")
    for key, value in parameters.items():
       exp_params[key] = exp.param(key, value) 

    true = []
    pred = []
    model.eval()
    test_running_accuracy = 0.0
    test_running_recall_at_ten = 0.0

    recall_intervals = [1,2,5,10,20,30,40,50]
    recall = {}
    for i in recall_intervals:
        recall[i] = []

    num_batches = 0

    print("Evaluating...")
    beginning_time = time.time() 

    prev_claim = None
    for batch_num, inputs in enumerate(dataloader):
        num_batches += 1
        claims_tensors, claims_text, evidences_tensors, evidences_text, labels = inputs  

        claims_tensors = claims_tensors.cuda()
        evidences_tensors = evidences_tensors.cuda()
        labels = labels.cuda()

        y_pred = model(claims_tensors, evidences_tensors)

        y = (labels).float()

        y_pred = y_pred.squeeze()
        y = y.squeeze()

        # flatten tensors
        y = y.view(-1)
        y_pred = y_pred.view(-1)

        bin_acc = torch.sigmoid(y_pred).to("cuda")

        if prev_claim is None:
            all_y = y 
            all_evidences = evidences_text 
            all_bin_acc = bin_acc 
        elif prev_claim!=claims_text[0]:
            # handle ranking here!
            sorted_idxs = torch.sort(all_bin_acc, descending=True)[1]

            relevant_evidences = []
            for idx in range(all_y.shape[0]):
                try:
                    if int(all_y[idx]):
                        relevant_evidences.append(all_evidences[idx])
                except Exception as e:
                    print(all_y, all_y[idx], idx)
                    raise e

            # if len(relevant_evidences)==0:
                # print("Zero relevant", y.sum())

            retrieved_evidences = []
            for idx in sorted_idxs:
                retrieved_evidences.append(all_evidences[idx])

            for k in recall_intervals:
                if len(relevant_evidences)==0:
                    recall[k].append(0)
                else:
                    recall[k].append(calculate_recall(retrieved_evidences, relevant_evidences, k=k))

            if len(relevant_evidences)==0:
                test_running_recall_at_ten += 0.0
            else:
                test_running_recall_at_ten += calculate_recall(retrieved_evidences, relevant_evidences, k=50)

            # reset tensors
            all_y = y 
            all_evidences = evidences_text 
            all_bin_acc = bin_acc 
        else:
            all_bin_acc = torch.cat([all_bin_acc, bin_acc])
            all_evidences.extend(evidences_text)
            all_y = torch.cat([all_y, y]) 
        prev_claim = claims_text[0]

        # for idx in range(len(y)):
          # print("Claim: {}, Evidence: {}, Prediction: {}, Label: {}".format(claims_text[idx], evidences_text[idx], bin_acc[idx], y[idx])) 
        
        # compute recall
        # assuming only one claim, this creates a list of all relevant evidences
        y = y.round()
        bin_acc = bin_acc.round()
        true.extend(y.tolist())
        pred.extend(bin_acc.tolist())

        accuracy = (y==bin_acc)
        accuracy = accuracy.float().mean()
        test_running_accuracy += accuracy.item()

        if batch_num % OUTPUT_FREQ==0 and batch_num>0:
            elapsed_time = time.time() - beginning_time
            print("[{}:{}:{:3f}s]: accuracy: {}, recall@50: {}".format(epoch, batch_num / len(val_dataset), elapsed_time, test_running_accuracy / OUTPUT_FREQ, test_running_recall_at_ten / OUTPUT_FREQ))
            for k, v in recall.items():
                print("recall@{}: {}".format(k, np.mean(v)))

            # 1. Log scalar values (scalar summary)
            info = { 'test_accuracy': test_running_accuracy/OUTPUT_FREQ }

            for tag, value in info.items():
                exp.metric(tag, value, log=False)
            #     logger.scalar_summary(tag, value, batch_num+1)

            # 2. Log values and gradients of the parameters (histogram summary)
            # for tag, value in model.named_parameters():
            #     tag = tag.replace('.', '/')
            #     logger.histo_summary(tag, value.data.cpu().numpy(), batch_num+1)

            test_running_accuracy = 0.0
            test_running_recall_at_ten = 0.0
            beginning_time = time.time()

        del claims_tensors
        del claims_text
        del evidences_tensors
        del evidences_text
        del labels 
        del y
        del y_pred
        torch.cuda.empty_cache()

    final_accuracy = accuracy_score(true, pred)
    print("Final accuracy: {}".format(final_accuracy))
    true = [int(i) for i in true]
    pred = [int(i) for i in pred]
    print(classification_report(true, pred))

    for k, v in recall.items():
        print("Recall@{}: {}".format(k, np.mean(v)))

    filename = "predicted_labels/predicted_labels"
    for key, value in parameters.items():
        key = key.replace(" ", "_")
        key = key.replace("/", "_")
        if type(value)==str:
            value = value.replace("/", "_")
        filename += "_{}-{}".format(key, value)

    joblib.dump({"true": true, "pred": pred}, filename)

def calculate_precision(retrieved, relevant, k=None):
    """
        retrieved: a list of sorted documents that were retrieved
        relevant: a list of sorted documents that are relevant
        k: how many documents to consider, all by default.
    """
    if k==None:
        k = len(retrieved)
    return len(set(retrieved[:k]).intersection(set(relevant))) / len(set(retrieved))

def calculate_recall(retrieved, relevant, k=None):
    """
        retrieved: a list of sorted documents that were retrieved
        relevant: a list of sorted documents that are relevant
        k: how many documents to consider, all by default.
    """
    if k==None:
        k = len(retrieved)
    return len(set(retrieved[:k]).intersection(set(relevant))) / len(set(relevant))

if __name__=="__main__":
    args = parse_args()

    print("Loading {}".format(args.data))
    
    fname = os.path.join(args.data,"train.pkl")
    test = joblib.load(fname)

    if args.sparse_evidences:
        print("Loading sparse evidences...")
        fname = os.path.join(args.data, "evidence.pkl")
        sparse_evidences = joblib.load(fname)
    else:
        sparse_evidences = None

    try:
        claims_dict
    except:
        print("Loading validation claims data...")
        claims_dict = joblib.load("claims_dict.pkl")

    test[2576]['evidence'] = test[0]['evidence']
    test[4176]['evidence'] = test[0]['evidence']
    test[4936]['evidence'] = test[0]['evidence']
    test[9835]['evidence'] = test[0]['evidence']
    test[10857]['evidence'] = test[0]['evidence']
    test[12177]['evidence'] = test[0]['evidence']
    test[12478]['evidence'] = test[0]['evidence']
    test[14404]['evidence'] = test[0]['evidence']

    torch.multiprocessing.set_start_method("spawn", force=True)
    run()
