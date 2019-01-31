# # Implementing CLSM

# ## Purpose
# The purpose of this notebook is to implement Microsoft's [Convolutional Latent Semantic Model](http://www.iro.umontreal.ca/~lisa/pointeurs/ir0895-he-2.pdf) on our dataset.
# 
# ## Inputs
# - This notebook requires *wiki-pages* from the FEVER dataset as an input.

# ## Preprocessing Data

import pickle
from multiprocessing import cpu_count
# from comet_ml import Experiment
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
from metrics import calculate_precision, calculate_recall

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
    parser.add_argument("--vocab-path", type=str, default='/data/sls/temp/weifang/fact_checking/processed/fever_uncased_40000_vocab.txt')
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
        dirname = os.path.dirname(args.vocab_path)
        base = os.path.splitext(os.path.basename(args.vocab_path))[0]
        base = base.rsplit('_', 1)[0]
        vec_path = os.path.join(dirname, f'{base}_wordvecs.pt')
        wordvecs = torch.load(vec_path)

        model = cdssm.CDSSM(wordvecs=wordvecs)
        model = model.cuda()
        model = model.to(device)
    if torch.cuda.device_count() > 0:
      print("Let's use", torch.cuda.device_count(), "GPU(s)!")
      model = nn.DataParallel(model)

    print("Created dataset...")
    dataset = pytorch_data_loader.ValWikiDataset(test, claims_dict, vocab_path=args.vocab_path, testFile="shared_task_dev.jsonl", sparse_evidences=sparse_evidences, batch_size=BATCH_SIZE) 
    dataloader = DataLoader(dataset, num_workers=0, collate_fn=pytorch_data_loader.PadCollate(), shuffle=False)

    OUTPUT_FREQ = int((len(dataset))*0.02) 
    
    parameters = {"batch size": BATCH_SIZE, "data": args.data, "model": args.model}
    # experiment = Experiment(api_key="YLsW4AvRTYGxzdDqlWRGCOhee", project_name="clsm", workspace="moinnadeem")
    # experiment.add_tag("test")
    # experiment.log_parameters(parameters)
    # experiment.log_asset("cdssm.py")

    true = []
    pred = []
    model.eval()
    test_running_accuracy = 0.0
    test_running_loss = 0.0
    test_running_recall_at_ten = 0.0

    recall_intervals = [1,2,5,10,20]
    recall = {}
    overall_recall = {}
    for i in recall_intervals:
        recall[i] = []
        overall_recall[i] = []

    num_batches = 0

    print("Evaluating...")
    beginning_time = time.time() 
    criterion = torch.nn.NLLLoss()

    # with experiment.test():
    with torch.no_grad():
        pbar = tqdm(dataloader)
        for batch_num, inputs in enumerate(pbar):
            num_batches += 1
            claims_tensors, claims_text, evidences_tensors, evidences_text, labels = inputs  

            claims_tensors = claims_tensors.cuda()
            evidences_tensors = evidences_tensors.cuda()
            labels = labels.cuda()

            y_pred = model(claims_tensors, evidences_tensors)

            y = (labels).float()

            y_pred = y_pred.squeeze()
            loss = criterion(y_pred, torch.max(y,1)[1])
            test_running_loss += loss.item()

            y_pred = torch.exp(y_pred)
            binary_y = torch.max(y, 1)[1]
            binary_y_pred = torch.max(y_pred, 1)[1]
            accuracy = (binary_y==binary_y_pred).to(device)
            bin_acc = y_pred[:,1]
            accuracy = accuracy.float().mean()
            # bin_acc = y_pred

            # handle ranking here!
            sorted_idxs = torch.sort(bin_acc, descending=True)[1]

            relevant_evidences = []
            for idx in range(y.shape[0]):
                try:
                    if int(y[idx][1]):
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
                    overall_recall[k].append(0.)
                else:
                    rec_k = calculate_recall(retrieved_evidences, relevant_evidences, k=k)
                    recall[k].append(rec_k)
                    overall_recall[k].append(rec_k)

            if len(relevant_evidences)==0:
                #test_running_recall_at_ten += 0.0
                pass
            else:
                test_running_recall_at_ten += calculate_recall(retrieved_evidences, relevant_evidences, k=20)


            if args.print:      
                for idx in sorted_idxs: 
                    print("Claim: {}, Evidence: {}, Prediction: {}, Label: {}".format(claims_text[0], evidences_text[idx], y_pred[idx], y[idx])) 

            # compute recall
            # assuming only one claim, this creates a list of all relevant evidences
            true.extend(binary_y.tolist())
            pred.extend(binary_y_pred.tolist())

            test_running_accuracy += accuracy.item()

            # if batch_num % OUTPUT_FREQ==0 and batch_num>0:
                # elapsed_time = time.time() - beginning_time
                # print("[{}:{:3f}s]: accuracy: {}, loss: {}, recall@20: {}".format(batch_num / len(dataloader), elapsed_time, test_running_accuracy / OUTPUT_FREQ, test_running_loss / OUTPUT_FREQ, test_running_recall_at_ten / OUTPUT_FREQ))
                # for k in sorted(recall.keys()):
                    # v = recall[k]
                    # print("recall@{}: {}".format(k, np.mean(v)))

                # # 1. Log scalar values (scalar summary)
                # info = { 'test_accuracy': test_running_accuracy/OUTPUT_FREQ }

                # true = [int(i) for i in true]
                # pred = [int(i) for i in pred]
                # print(classification_report(true, pred))

                # # for tag, value in info.items():
                   # # experiment.log_metric(tag, value, step=batch_num)

                # # 2. Log values and gradients of the parameters (histogram summary)
                # # for tag, value in model.named_parameters():
                # #     tag = tag.replace('.', '/')
                # #     logger.histo_summary(tag, value.data.cpu().numpy(), batch_num+1)

                # test_running_accuracy = 0.0
                # test_running_recall_at_ten = 0.0
                # test_running_loss = 0.0
                # beginning_time = time.time()

    true = [int(i) for i in true]
    pred = [int(i) for i in pred]
    final_accuracy = accuracy_score(true, pred)
    print("Final accuracy: {}".format(final_accuracy))
    print(classification_report(true, pred))

    recall_at_k = [(k, np.mean(v)) for k, v in recall.items()]
    overall_recall_at_k = [(k, np.mean(v)) for k, v in overall_recall.items()]
    print("           {} | {}".format("nm_only", "overall"))
    for (k, v1), (_, v2) in zip(recall_at_k, overall_recall_at_k):
        print("Recall@{:>2}: {:.5f} | {:.5f}".format(k, v1, v2))

    # filename = "predicted_labels/predicted_labels"
    # for key, value in parameters.items():
        # key = key.replace(" ", "_")
        # key = key.replace("/", "_")
        # if type(value)==str:
            # value = value.replace("/", "_")
        # filename += "_{}-{}".format(key, value)

    # joblib.dump({"true": true, "pred": pred}, filename)

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
