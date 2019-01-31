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
# from comet_ml import Experiment
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
from metrics import calculate_precision, calculate_recall

torch.backends.cudnn.benchmark=True
nltk.data.path.append('/usr/users/mnadeem/nltk_data/')

def parse_args():
    parser = argparse.ArgumentParser(description='Learning the optimal convolution for network.')
    parser.add_argument("--batch-size", type=int, help="Number of queries per batch.", default=10)
    parser.add_argument("--model", help="Loading a pretrained model.", default=None)
    parser.add_argument("--data-sampling", type=int, help="Number of examples per query.", default=3)
    parser.add_argument("--no-randomize", default=True, action="store_false", help="Disables randomly selecting documents from the data loader.")
    parser.add_argument("--learning-rate", type=float, help="Learning rate for model.", default=1e-4)
    parser.add_argument("--epochs", type=int, help="Number of epochs to learn for.", default=15)
    parser.add_argument("--data", help="Folder dataset to load file from.", default="data/large")
    parser.add_argument("--print", default=False, action="store_true", help="Whether to print predicted labels or not.")
    parser.add_argument("--sparse-evidences", default=False, action="store_true")
    parser.add_argument("--vocab-path", type=str, default='/data/sls/temp/weifang/fact_checking/processed/fever_uncased_40000_vocab.txt')
    parser.add_argument("--eval-train", action="store_true")
    parser.add_argument("--eval-test", action="store_true")
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

    dirname = os.path.dirname(args.vocab_path)
    base = os.path.splitext(os.path.basename(args.vocab_path))[0]
    base = base.rsplit('_', 1)[0]
    vec_path = os.path.join(dirname, f'{base}_wordvecs.pt')
    wordvecs = torch.load(vec_path)

    if MODEL:
        print("Loading pretrained model...")
        model = torch.load(MODEL)
        model.load_state_dict(torch.load(MODEL).state_dict())
    else:
        model = cdssm.CDSSM(wordvecs=wordvecs)
        model = model.cuda()
        model = model.to(device)

    # model = cdssm.CDSSM()
    # model = model.cuda()
    # model = model.to(device)

    if torch.cuda.device_count() > 0:
      print("Let's use", torch.cuda.device_count(), "GPU(s)!")
      model = nn.DataParallel(model)
    
    print(model)
    print("Created model with {:,} parameters.".format(putils.count_parameters(model)))

    # if MODEL:
        # print("TEMPORARY change to loading!")
        # model.load_state_dict(torch.load(MODEL).state_dict())

    print("Created dataset...")

    # use an 80/20 train/validate split!
    train_size = int(len(train) * 0.80)
    #test = int(len(train) * 0.5)
    train_dataset = pytorch_data_loader.WikiDataset(train[:train_size], claims_dict, vocab_path=args.vocab_path, data_sampling=DATA_SAMPLING, sparse_evidences=sparse_evidences, randomize=RANDOMIZE) 
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=0, shuffle=True, drop_last=True, collate_fn=pytorch_data_loader.PadCollate())

    val_dataset = pytorch_data_loader.ValWikiDataset(train[train_size:], claims_dict, vocab_path=args.vocab_path, sparse_evidences=sparse_evidences, batch_size=20) 
    val_dataloader = DataLoader(val_dataset, batch_size=1, num_workers=0, shuffle=False, collate_fn=pytorch_data_loader.PadCollate())

    if args.eval_train:
        train_eval_dataset = pytorch_data_loader.ValWikiDataset(train[:train_size], claims_dict, vocab_path=args.vocab_path, sparse_evidences=sparse_evidences, batch_size=20) 

        train_eval_dataloader = DataLoader(train_eval_dataset, batch_size=1, num_workers=0, shuffle=False, collate_fn=pytorch_data_loader.PadCollate())

    if args.eval_test:
        fname = os.path.join("data/validation","train.pkl")
        test = joblib.load(fname)
        test_dataset = pytorch_data_loader.ValWikiDataset(test, claims_dict, vocab_path=args.vocab_path, testFile="shared_task_dev.jsonl", sparse_evidences=sparse_evidences, batch_size=20) 
        test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=0, collate_fn=pytorch_data_loader.PadCollate(), shuffle=False)

    # Loss and optimizer
    criterion = torch.nn.NLLLoss()
    # criterion = torch.nn.SoftMarginLoss()
    # if torch.cuda.device_count() > 0:
        # print("Let's parallelize the backward pass...")
        # criterion = DataParallelCriterion(criterion)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)

    OUTPUT_FREQ = max(int((len(train_dataset)/BATCH_SIZE)*0.0025), 20) 
    parameters = {"batch": BATCH_SIZE, "ep": NUM_EPOCHS, "lr": LEARNING_RATE, "opt": optimizer.__class__.__name__, "loss": criterion.__class__.__name__, "train size": train_size, "samp": DATA_SAMPLING, "data": args.data, "model": MODEL}
    # experiment = Experiment(api_key="YLsW4AvRTYGxzdDqlWRGCOhee", project_name="clsm", workspace="moinnadeem")
    # experiment.add_tag("train")
    # experiment.log_asset("cdssm.py")
    # experiment.log_dataset_info(name=args.data)
    # experiment.log_parameters(parameters)

    model_checkpoint_dir = "models/saved_model" 
    for key, value in parameters.items():
        if type(value)==str:
            value = value.replace("/", "-")
        if key!="model":
            model_checkpoint_dir += "_{}-{}".format(key.replace(" ", "_"), value)

    model_checkpoint_dir += '_' + str(time.time())

    if not os.path.isdir(model_checkpoint_dir):
        print("Created directory '" + model_checkpoint_dir + "'")
        os.makedirs(model_checkpoint_dir)

    print("Training...")
    beginning_time = time.time() 
    # best_loss = torch.tensor(float("inf"), dtype=torch.float)  # begin loss at infinity
    best_metric = 0.

    for epoch in range(NUM_EPOCHS):
        beginning_time = time.time()
        mean_train_acc = 0.0
        train_running_loss = 0.0
        train_running_accuracy = 0.0
        model.train()
        # experiment.log_current_epoch(epoch)

        # with experiment.train():
        if True:
            print('-'*80)
            pbar = tqdm(train_dataloader)
            # print('Epoch {}'.format(epoch+1))
            for train_batch_num, inputs in enumerate(pbar):
                claims_tensors, claims_text, evidences_tensors, evidences_text, labels = inputs  

                claims_tensors = claims_tensors.cuda()
                evidences_tensors = evidences_tensors.cuda()
                labels = labels.cuda()
                #claims = claims.to(device).float()
                #evidences = evidences.to(device).float()
                #labels = labels.to(device)

                optimizer.zero_grad()
                y_pred = model(claims_tensors, evidences_tensors)

                y = (labels)
                # y = y.unsqueeze(0)
                # y = y.unsqueeze(0)
                # y_pred = parallel.gather(y_pred, 0)

                y_pred = y_pred.squeeze()
                # y = y.squeeze()

                loss = criterion(y_pred, torch.max(y,1)[1])
                # loss = criterion(y_pred, y)

                y = y.float()
                binary_y = torch.max(y, 1)[1]
                binary_pred = torch.max(y_pred, 1)[1]
                accuracy = (binary_y==binary_pred).to("cuda")
                accuracy = accuracy.float()
                accuracy = accuracy.mean()
                train_running_accuracy += accuracy.item()
                mean_train_acc += accuracy.item()
                train_running_loss += loss.item()


                if PRINT:
                    for idx in range(len(y)): 
                        print("Claim: {}, Evidence: {}, Prediction: {}, Label: {}".format(claims_text[0], evidences_text[idx], torch.exp(y_pred[idx]), y[idx])) 

                if (train_batch_num % OUTPUT_FREQ)==0 and train_batch_num>0:
                    elapsed_time = time.time() - beginning_time
                    binary_y = torch.max(y, 1)[1]
                    binary_pred = torch.max(y_pred, 1)[1]
                    pbar.set_description("[Epoch {:>2}] loss: {:.3f}, acc: {:.3f}".format(epoch+1, train_running_loss/OUTPUT_FREQ, train_running_accuracy/OUTPUT_FREQ))
                    # print("[{}:{}:{:3f}s] training loss: {}, training accuracy: {}, training recall: {}".format(epoch, train_batch_num / (len(train_dataset)/BATCH_SIZE), elapsed_time, train_running_loss/OUTPUT_FREQ, train_running_accuracy/OUTPUT_FREQ, recall_score(binary_y.cpu().detach().numpy(), binary_pred.cpu().detach().numpy())))

                    # 1. Log scalar values (scalar summary)
                    info = { 'train_loss': train_running_loss/OUTPUT_FREQ, 'train_accuracy': train_running_accuracy/OUTPUT_FREQ }

                    for tag, value in info.items():
                       # experiment.log_metric(tag, value, step=train_batch_num*(epoch+1))
                       logger.scalar_summary(tag, value, train_batch_num+1)

                    ## 2. Log values and gradients of the parameters (histogram summary)
                    # for tag, value in model.named_parameters():
                        # tag = tag.replace('.', '/')
                        # logger.histo_summary(tag, value.detach().cpu().numpy(), train_batch_num+1)
                        # logger.histo_summary(tag+'/grad', value.grad.detach().cpu().numpy(), train_batch_num+1)

                    train_running_loss = 0.0
                    beginning_time = time.time() 
                    train_running_accuracy = 0.0
                loss.backward()
                optimizer.step()

        if args.eval_train:
            print("Running validation on train...")
            evaluate(model, train_eval_dataloader, criterion)

        print("Running validation on dev...")
        acc, loss, recall_at_k, overall_recall_at_k = evaluate(model, val_dataloader, criterion)
        metric = overall_recall_at_k[1][1] # recall @ 2
        best_metric = max(metric, best_metric)
        is_best = (metric >= best_metric)
        
        putils.save_checkpoint({"epoch": epoch, "model": model}, is_best, 
                               filename=os.path.join(model_checkpoint_dir, "ep-{}_r@2-{:.6f}.pth".format(epoch+1, best_metric)))

        if args.eval_test:
            print("Running validation on test...")
            evaluate(model, test_dataloader, criterion)


def evaluate(model, val_dataloader, criterion):
    model.eval()
    pred = []
    true = []
    avg_loss = 0.0
    val_running_accuracy = 0.0
    val_running_loss = 0.0
    beginning_time = time.time()
    # with experiment.validate():

    recall_intervals = [1,2,5,10,20]
    recall = {}
    overall_recall = {}
    for i in recall_intervals:
        recall[i] = []
        overall_recall[i] = []

    with torch.no_grad():
        for val_batch_num, val_inputs in enumerate(tqdm(val_dataloader)):
            claims_tensors, claims_text, evidences_tensors, evidences_text, labels = val_inputs  

            claims_tensors = claims_tensors.cuda()
            evidences_tensors = evidences_tensors.cuda()
            labels = labels.cuda()

            y_pred = model(claims_tensors, evidences_tensors)

            y = (labels)
            # y_pred = parallel.gather(y_pred, 0)

            y_pred = y_pred.squeeze()

            loss = criterion(y_pred, torch.max(y,1)[1])

            y = y.float()

            binary_y = torch.max(y, 1)[1]
            y_pred = torch.exp(y_pred)
            binary_pred = torch.max(y_pred, 1)[1]
            true.extend(binary_y.tolist())
            pred.extend(binary_pred.tolist())

            accuracy = (binary_y==binary_pred).to("cuda")

            accuracy = accuracy.float().mean()
            val_running_accuracy += accuracy.item()
            val_running_loss += loss.item() 
            avg_loss += loss.item()

            bin_acc = y_pred[:,1]
            sorted_idxs = torch.sort(bin_acc, descending=True)[1]

            relevant_evidences = []
            for idx in range(y.shape[0]):
                try:
                    if int(y[idx][1]):
                        relevant_evidences.append(evidences_text[idx])
                except Exception as e:
                    print(y, y[idx], idx)
                    raise e

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


            # if (val_batch_num % OUTPUT_FREQ)==0 and val_batch_num>0:
                # # elapsed_time = time.time() - beginning_time
                # # print("[{}:{}:{:3f}s] validation loss: {}, accuracy: {}, recall: {}".format(epoch, val_batch_num / (len(val_dataset)/BATCH_SIZE), elapsed_time, val_running_loss/OUTPUT_FREQ, val_running_accuracy/OUTPUT_FREQ, recall_score(binary_y.cpu().detach().numpy(), binary_pred.cpu().detach().numpy())))

                # # 1. Log scalar values (scalar summary)
                # info = { 'val_accuracy': val_running_accuracy/OUTPUT_FREQ }

                # for tag, value in info.items():
                   # # experiment.log_metric(tag, value, step=val_batch_num*(epoch+1))
                   # logger.scalar_summary(tag, value, val_batch_num+1)

                # ## 2. Log values and gradients of the parameters (histogram summary)
                # for tag, value in model.named_parameters():
                   # tag = tag.replace('.', '/')
                   # # logger.histo_summary(tag, value.detach().cpu().numpy(), val_batch_num+1)
                   # # logger.histo_summary(tag+'/grad', value.grad.detach().cpu().numpy(), val_batch_num+1)

                # val_running_accuracy = 0.0
                # val_running_loss = 0.0
                # beginning_time = time.time()


    accuracy = accuracy_score(true, pred) 
    avg_loss /= len(val_dataloader)
    print("Mean accuracy: {:.5f}, mean loss: {:.5f}".format(accuracy, avg_loss))

    true = np.array(true).astype("int") 
    pred = np.array(pred).astype("int") 
    print(classification_report(true, pred))

    print('True: #0={}, #1={}'.format((true==0).sum(), (true==1).sum()))
    print('Pred: #0={}, #1={}'.format((pred==0).sum(), (pred==1).sum()))

    recall_at_k = [(k, np.mean(v)) for k, v in recall.items()]
    overall_recall_at_k = [(k, np.mean(v)) for k, v in overall_recall.items()]
    print("           {} | {}".format("nm_only", "overall"))
    for (k, v1), (_, v2) in zip(recall_at_k, overall_recall_at_k):
        print("Recall@{:>2}: {:.5f} | {:.5f}".format(k, v1, v2))

    return accuracy, avg_loss, recall_at_k, overall_recall_at_k


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
