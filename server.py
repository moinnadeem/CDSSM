import importlib
import joblib
import torch
import os
import argparse
import clsm_pytorch as clsm

def parse_args():
    parser = argparse.ArgumentParser(description='Learning the optimal convolution for network.')
    parser.add_argument("--batch-size", type=int, help="Number of queries per batch.", default=1)
    parser.add_argument("--data-batch-size", type=int, help="Number of examples per query.", default=8)
    parser.add_argument("--learning-rate", type=float, help="Learning rate for model.", default=1e-3)
    parser.add_argument("--epochs", type=int, help="Number of epochs to learn for.", default=3)
    parser.add_argument("--data", help="Folder dataset to load file from.", default="data/large/train.pkl")
    parser.add_argument("--sparse-evidences", default=False, action="store_true")
    return parser.parse_args()

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
        claims_dict = joblib.load("new_claims_dict.pkl")

    #torch.multiprocessing.set_start_method("fork", force=True)

    try:
        clsm.run(args, train, sparse_evidences, claims_dict) 
    except KeyboardInterrupt:
        importlib.reload(clsm)
        clsm.run(args, train, sparse_evidences, claims_dict) 
