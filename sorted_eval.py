# # Implementing CLSM

# ## Purpose
# The purpose of this notebook is to implement Microsoft's [Convolutional Latent Semantic Model](http://www.iro.umontreal.ca/~lisa/pointeurs/ir0895-he-2.pdf) on our dataset.
# 
# ## Inputs
# - This notebook requires *wiki-pages* from the FEVER dataset as an input.

# ## Preprocessing Data

import pickle
from multiprocessing import cpu_count
from comet_ml import Experiment
import os
from parallel import DataParallelModel, DataParallelCriterion
import pytorch_utils as putils
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
    parser.add_argument("--batch-size", type=int, help="Number of queries per batch.", default=20)
    parser.add_argument("--learning-rate", type=float, help="Learning rate for model.", default=1e-3)
    parser.add_argument("--epochs", type=int, help="Number of epochs to learn for.", default=3)
    parser.add_argument("--randomize", default=False, action="store_true")
    parser.add_argument("--data", help="Training dataset to load file from.", default="data/validation")
    parser.add_argument("--model", help="Model to evaluate.") 
    parser.add_argument("--sparse-evidences", default=False, action="store_true")
    parser.add_argument("--print", default=False, action="store_true", help="Whether to print predicted labels or not.")
    return parser.parse_args()

# @monitor("CLSM Test")
def run():
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.learning_rate
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
    dataloader = DataLoader(dataset, num_workers=0, collate_fn=pytorch_data_loader.PadCollate(), shuffle=False)

    OUTPUT_FREQ = int((len(dataset))*0.02) 
    
    parameters = {"batch size": BATCH_SIZE, "data": args.data, "model": args.model}
    experiment = Experiment(api_key="YLsW4AvRTYGxzdDqlWRGCOhee", project_name="clsm", workspace="moinnadeem")
    experiment.add_tag("test")
    experiment.log_parameters(parameters)
    experiment.log_asset("cdssm.py")

    true = []
    pred = []
    model.eval()
    test_running_accuracy = 0.0
    test_running_loss = 0.0
    test_running_recall_at_ten = 0.0

    recall_intervals = [1,2,5,10,20]
    recall = {}
    for i in recall_intervals:
        recall[i] = []

    num_batches = 0

    print("Evaluating...")
    beginning_time = time.time() 
    #criterion = torch.nn.NLLLoss()
    # criterion = torch.nn.SoftMarginLoss()
    criterion = putils.ContrastiveLoss()

    with experiment.test():
        for batch_num, inputs in enumerate(dataloader):
            num_batches += 1
            claims_tensors, claims_text, evidences_tensors, evidences_text, labels = inputs  

            claims_tensors = claims_tensors.cuda()
            evidences_tensors = evidences_tensors.cuda()
            labels = labels.cuda()

            try:
                y_pred = model(claims_tensors, evidences_tensors)
            except Exception as e:
                continue

            y = (labels).float()
            y = 1-y

            y_pred = y_pred.squeeze()
            # loss = criterion(y_pred, torch.max(y,1)[1])
            loss = criterion(y_pred, y)
            test_running_loss += loss.item()

            classifications = torch.norm(y_pred, p="fro", dim=1)
            binary_classifications = classifications >= 1

            accuracy = (classifications==y).to(device)
            accuracy = accuracy.float().mean()
            # bin_acc = y_pred

            # handle ranking here!
            sorted_idxs = torch.sort(classifications, descending=False)[1]

            relevant_evidences = []
            for idx in range(y.shape[0]):
                try:
                    if not int(y[idx]):
                        relevant_evidences.append(evidences_text[idx])
                except Exception as e:
                    print(y, y[idx], idx)
                    raise e

            # if len(relevant_evidences)==0:
                # print("Zero relevant", y.sum())

            retrieved_evidences = []
            for idx in sorted_idxs:
                retrieved_evidences.append(evidences_text[idx])

            for k in recall_intervals:
                if len(relevant_evidences)==0:
                    # recall[k].append(0)
                    pass
                else:
                    recall[k].append(calculate_recall(retrieved_evidences, relevant_evidences, k=k))

            if len(relevant_evidences)==0:
                #test_running_recall_at_ten += 0.0
                pass
            else:
                test_running_recall_at_ten += calculate_recall(retrieved_evidences, relevant_evidences, k=20)


            if args.print:      
                for idx in sorted_idxs: 
                    print("Claim: {}, Evidence: {}, Prediction: {}, Label: {}".format(claims_text[0], evidences_text[idx], classifications[idx], y[idx])) 

            # compute recall
            # assuming only one claim, this creates a list of all relevant evidences
            true.extend(y.tolist())
            pred.extend(classifications.tolist())

            test_running_accuracy += accuracy.item()

            if batch_num % OUTPUT_FREQ==0 and batch_num>0:
                elapsed_time = time.time() - beginning_time
                print("[{}:{:3f}s]: accuracy: {}, loss: {}, recall@20: {}".format(batch_num / len(dataloader), elapsed_time, test_running_accuracy / OUTPUT_FREQ, test_running_loss / OUTPUT_FREQ, test_running_recall_at_ten / OUTPUT_FREQ))
                for k in sorted(recall.keys()):
                    v = recall[k]
                    print("recall@{}: {}".format(k, np.mean(v)))

                # 1. Log scalar values (scalar summary)
                info = { 'test_accuracy': test_running_accuracy/OUTPUT_FREQ }

                true = [int(i) for i in true]
                pred = [int(i) for i in pred]
                print(classification_report(true, pred))

                for tag, value in info.items():
                   experiment.log_metric(tag, value, step=batch_num)

                # 2. Log values and gradients of the parameters (histogram summary)
                # for tag, value in model.named_parameters():
                #     tag = tag.replace('.', '/')
                #     logger.histo_summary(tag, value.data.cpu().numpy(), batch_num+1)

                test_running_accuracy = 0.0
                test_running_recall_at_ten = 0.0
                test_running_loss = 0.0
                beginning_time = 0.0
                # beginning_time = time.time()

        # del claims_tensors
        # del claims_text
        # del evidences_tensors
        # del evidences_text
        # del labels 
        # del y
        # del y_pred
        # torch.cuda.empty_cache()

    true = [int(i) for i in true]
    pred = [int(i) for i in pred]
    final_accuracy = accuracy_score(true, pred)
    print("Final accuracy: {}".format(final_accuracy))
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

    # torch.multiprocessing.set_start_method("spawn", force=True)
    run()
