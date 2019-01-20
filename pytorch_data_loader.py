import joblib
import numpy as np
import torch
from scipy import sparse
from torch.utils.data import DataLoader, Dataset

import utils

#torch.multiprocessing.set_start_method("spawn")

def variable_collate(batch):
    #claims = []
    #evidences = []
    #labels = []
    #for item in batch:
    #    for c in item[0]:
    #        claims.append(c)
    #    for e in item[1]:
    #        evidences.append(e)
    #    for l in item[2]:
    #        labels.append(l)

    claims_tensors = []
    claims_text = []
    evidences_tensors = []
    evidences_text = []
    labels = []
    for item in batch:
        for c in item[0]:
            claims_tensors.append(c)

        for c in item[1]:
            claims_text.append(c)

        for c in item[2]:
            evidences_tensors.append(c)

        for c in item[3]:
            evidences_text.append(c)

        for c in item[4]:
            labels.append(c)

    claims_tensors = stack_uneven(claims_tensors)
    evidences_tensors = stack_uneven(evidences_tensors)
    labels = torch.LongTensor(labels).cuda()

    claims_tensors = torch.from_numpy(claims_tensors).cuda()
    evidences_tensors = torch.from_numpy(evidences_tensors).cuda()

    return [claims_tensors, claims_text, evidences_tensors, evidences_text, labels]
    
def pad_tensor(vec, pad, dim):
    """
    args:
        vec - tensor to pad
        pad - the size to pad to
        dim - dimension to pad

    return:
        a new tensor padded to 'pad' in dimension 'dim'
    """
    # pad_size = list(vec.shape)
    # pad_size[dim] = pad - vec.size(dim)
    # if vec.is_cuda:
        # zeros_tensor = torch.cuda.FloatTensor(*pad_size)
        # # zeros_tensor[dim] = vec
    # else:
        # zeros_tensor = torch.FloatTensor(*pad_size)

    # return torch.cat([vec, zeros_tensor], dim=dim)

    # pad_size = list(vec.shape)
    # pad_size[dim] = pad 
    padding = torch.nn.ZeroPad2d((0, 0, 0, pad - vec.size(dim)))
    return padding(vec).cuda()
    
    # return zeros_tensor
    # return torch.cat([vec, zeros_tensor], dim=dim)

class PadCollate:
    """
    a variant of callate_fn that pads according to the longest sequence in
    a batch of sequences
    """

    def __init__(self, dim=0):
        """
        args:
            dim - the dimension to be padded (dimension of time in sequences)
        """
        self.dim = dim
        use_cuda = True
        self.device = torch.device("cuda" if use_cuda else "cpu")

    def pad_collate(self, batch):
        """
        args:
            batch - list of (claim (tensor), claim (text), evidence (tensor), evidence (text), label)

        reutrn:
            xs - a tensor of all examples in 'batch' after padding
            ys - a LongTensor of all labels in batch
        """
        claims_tensors = []
        claims_text = []
        evidences_tensors = []
        evidences_text = []
        labels = []
        for item in batch:
            claims_tensors.extend(item[0])
            claims_text.extend(item[1])
            evidences_tensors.extend(item[2])
            evidences_text.extend(item[3])
            labels.extend(item[4])

        batched_items = []

        for tensor in [claims_tensors, evidences_tensors]:
            # find longest sequence
            max_len = max(map(lambda x: x.shape[self.dim], tensor))

            # pad according to max_len
            batched_items.append(list(map(lambda x: pad_tensor(x, pad=max_len, dim=self.dim), tensor)))

        # stack all
        claims_tensors = torch.stack(batched_items[0], dim=0).cuda()
        evidences_tensors = torch.stack(batched_items[1], dim=0).cuda()
        labels = torch.tensor(labels, dtype=torch.float).cuda()
        return [claims_tensors, claims_text, evidences_tensors, evidences_text, labels] 

    def __call__(self, batch):
        return self.pad_collate(batch)

class WikiDataset(Dataset):
    """
    Generates data with batch size of 1 sample for the purposes of training our model.
    """
    def __init__(self, data, claims_dict, data_sampling=10, batch_size=32, split=None, randomize=True, testFile="train.jsonl", sparse_evidences=None):
        """
            Sets the initial arguments and creates
            an indicies array to randomize the dataset
            between epochs
        """
        if split:            
            self.indicies = split
        else:
            self.indicies = list(range(len(data)))
        self.data = data[::-1]
        self.randomize = randomize
        if sparse_evidences:
            self.evidence_to_sparse = sparse_evidences 
        else:
            self.evidence_to_sparse = None
        use_cuda = True 
        self.device = torch.device("cuda:0" if use_cuda else "cpu")
        self.data_sampling = data_sampling 
        self.encoder = utils.ClaimEncoder()
        self.claims_dict = claims_dict
        self.batch_size = batch_size
        self.collate_fn = PadCollate()
        _, _, _, _, self.claim_to_article = utils.extract_fever_jsonl_data(testFile)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.get_item(index)
    
    def get_item(self, index):            
        #data_index = index % self.data_sampling
        item_index = index 
        
        d = self.data[item_index]  # get training item 
        #claim = utils.preprocess_article_name(d['claim'])  # preprocess the claim
        #claim = self.encoder.tokenize_claim(claim)
        #claim = sparse.vstack(claim).toarray()  # turn it into a array
        claim = self.claims_dict[d['claim']]
        claim = claim.toarray()
        claim = torch.from_numpy(claim).cuda().float()
        claim_text = d['claim']
        #claim = sparse.vstack(self.encoder.tokenize_claim(utils.preprocess_article_name(d['claim']))).toarray()

        claims_tensors = []
        claims_text = []
        evidence_tensors = []
        evidence_text = []
        labels = []

        num_positive_articles = min(len(self.claim_to_article[d['claim']]), 4)  # get all positive articles 
        for idx in range(num_positive_articles):
            processed = self.claim_to_article[d['claim']][idx]

            if self.evidence_to_sparse:
                if processed in self.evidence_to_sparse: 
                    evidence = self.evidence_to_sparse[processed]
                else:
                    print("Claim: {}, evidence {} is missing!".format(d['claim'], processed)) 
                    return self.get_item(index+1)
            else:
                evidence = self.encoder.tokenize_claim(processed)
                evidence = sparse.vstack(evidence)

            evidence = evidence.toarray()
            evidence = torch.from_numpy(evidence).cuda().float()

            evidence_text.append(processed)
            evidence_tensors.append(evidence)

            claims_text.append(claim_text) 
            claims_tensors.append(claim)
            #print("{}, Evidence: {}, label: {}".format(claim_text, processed, 1.0))
            labels.append([0,1])

        for j in range(num_positive_articles, self.data_sampling):
            if not self.randomize:
                e = d['evidence'][j]
            else:
                e = np.random.choice(d['evidence'])

            processed = utils.preprocess_article_name(e.split("http://wikipedia.org/wiki/")[1])
            #evidence = articles_dict[processed]

            if processed=="":  # handle empty string case
                evidence = sparse.coo_matrix((10, 29244))  # randomly chosen shape for array
            else:
                if self.evidence_to_sparse:
                    if processed in self.evidence_to_sparse:
                        evidence = self.evidence_to_sparse[processed]
                    else:
                        print(e)
                        raise Exception("You fucked up somewhere")

                else: 
                    evidence = self.encoder.tokenize_claim(processed)
                    if len(evidence)>0:
                        evidence = sparse.vstack(evidence)

            if evidence.shape[0]>0:
                evidence = evidence.toarray()
                evidence = torch.from_numpy(evidence).cuda().float()
                evidence_tensors.append(evidence)
                evidence_text.append(processed)

                claims_text.append(claim_text)
                claims_tensors.append(claim)

                if processed in self.claim_to_article[d['claim']]:
                    labels.append([0,1])
                else:
                    labels.append([1,0])
            else:
                print(d['claim'], e)
                raise Exception("SKipping append")

        #claim = claim.expand(evidences.shape[0], claim.shape[0], claim.shape[1])
        # claims_tensors = self.collate_fn(claims_tensors)
        # evidence_tensors = self.collate_fn(evidence_tensors)
        return claims_tensors, claims_text, evidence_tensors, evidence_text, labels 

    def on_epoch_end(self):
        #np.random.shuffle(self.indicies)
        pass

def to_torch_sparse_tensor(M, device="cuda"):
    M = M.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((M.row, M.col))).cuda().long()
    values = torch.from_numpy(M.data).cuda()
    shape = torch.Size(M.shape)
    T = torch.cuda.sparse.FloatTensor(indices, values, shape, device=device)
    return T

def stack_uneven(arrays, fill_value=0.):
    '''
    Fits arrays into a single numpy array, even if they are
    different sizes. `fill_value` is the default value.

    Args:
            arrays: list of np arrays of various sizes
                (must be same rank, but not necessarily same size)
            fill_value (float, optional):

    Returns:
            np.ndarray
    '''
    sizes = [a.shape for a in arrays]
    max_sizes = np.max(list(zip(*sizes)), -1)

    # The resultant array has stacked on the first dimension
    result = np.full((len(arrays),) + tuple(max_sizes), fill_value, dtype=np.float)
    for i, a in enumerate(arrays):
      # The shape of this array `a`, turned into slices
      slices = tuple(slice(0,s) for s in sizes[i])

      # Overwrite a block slice of `result` with this array `a`
      result[i][slices] = a
    return result

class ValWikiDataset(Dataset):
    def __init__(self, data, claims_dict, batch_size=1, split=None, testFile="train.jsonl", sparse_evidences=None):
        """
        Initializes the class.
        """

        if split:
            self.indicies = split
        else:
            self.indicies = list(range(len(data)))

        self.data = data[::-1]

        if sparse_evidences:
            self.evidence_to_sparse = sparse_evidences
        else:
            self.evidence_to_sparse = None

        use_cuda = True
        self.device = torch.device("cuda:0" if use_cuda else "cpu")
        self.encoder = utils.ClaimEncoder()
        self.claims_dict = claims_dict
        self.batch_size = batch_size
        _, _, _, _, self.claim_to_article = utils.extract_fever_jsonl_data(testFile)

    def __len__(self):
        return (len(self.data)*20)//self.batch_size

    def __getitem__(self, index):
        return self.get_item(index)

    def get_item(self, index):
        claim_index = (index*self.batch_size)//20
        evidences_idx = (index*self.batch_size)%20

        d = self.data[claim_index]
        claim = self.claims_dict[d['claim']]
        claim = claim.toarray()
        claim = torch.from_numpy(claim).cuda().float()
        claim_text = d['claim']

        claim_tensors = []
        claim_texts = []
        evidence_tensors = []
        evidence_text = []
        labels = []

        for j in range(evidences_idx, evidences_idx+self.batch_size):
            try:
                e = d['evidence'][j]
            except:
                raise Exception("Out of range, evidence idx is {}, claim_idx is {}".format(j, claim_index))

            processed = utils.preprocess_article_name(e.split("http://wikipedia.org/wiki/")[1])

            if processed=="":
                evidence = sparse.coo_matrix((10, 29244))
                print("Zero length evidence encountered. Be careful!")
            else:
                if self.evidence_to_sparse:
                    if processed in self.evidence_to_sparse:
                        evidence = self.evidence_to_sparse[processed]
                    else:
                        print(e)
                        raise Exception("Some item has not been found in the sparse dataset")
                else:
                    evidence = self.encoder.tokenize_claim(processed)
                    if len(evidence)>0:
                        evidence = sparse.vstack(evidence)

            if evidence.shape[0]>0:
                evidence = evidence.toarray()
                evidence = torch.from_numpy(evidence).cuda().float()
                evidence_tensors.append(evidence)
                evidence_text.append(processed)

                claim_texts.append(claim_text)
                claim_tensors.append(claim)
                # TODO: This isn't really necessary. 
                # You could probably steal some performance gains by copying on the GPU.

                if processed in self.claim_to_article[d['claim']]:
                    labels.append([0,1])
                else:
                    labels.append([1,0])
            else:
                print(d['claim'], e, "is not positive length!")

        return [claim_tensors, claim_texts, evidence_tensors, evidence_text, labels]

    def on_epoch_end(self):
        np.random.shuffle(self.indicies)
