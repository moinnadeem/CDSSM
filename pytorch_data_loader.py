from scipy import sparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import utils

#torch.multiprocessing.set_start_method("spawn")

def variable_collate(batch):
    claims = []
    evidences = []
    labels = []
    for item in batch:
        for c in item[0]:
            claims.append(c)
        for e in item[1]:
            evidences.append(e)
        for l in item[2]:
            labels.append(l)
    claims = stack_uneven(claims)
    evidences = stack_uneven(evidences)
    labels = torch.Tensor(labels).long()

    claims= torch.from_numpy(claims).float()
    evidences = torch.from_numpy(evidences).float()

    return [claims, evidences, labels]
    

class WikiDataset(Dataset):
    """
    Generates data with batch size of 1 sample for the purposes of training our model.
    """
    def __init__(self, data, claims_dict, data_batch_size=10, batch_size=32, split=None, randomize=True, testFile="train.jsonl"):
        """
            Sets the initial arguments and creates
            an indicies array to randomize the dataset
            between epochs
        """
        if split:            
            self.indicies = split
        else:
            self.indicies = list(range(len(data)))
        self.data = data
        self.randomize = randomize
        use_cuda = True
        self.device = torch.device("cuda:0" if use_cuda else "cpu")
        self.data_batch_size = data_batch_size
        self.encoder = utils.ClaimEncoder()
        self.claims_dict = claims_dict
        self.batch_size = batch_size
        _, _, _, _, self.claim_to_article = utils.extract_fever_jsonl_data(testFile)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.get_item(index)
    
    def get_item(self, index):            
        #data_index = index % self.data_batch_size
        item_index = index 
        
        d = self.data[item_index]
        claim = utils.preprocess_article_name(d['claim'])
        claim = self.encoder.tokenize_claim(claim)
        claim = sparse.vstack(claim).toarray()
        #claim = (self.claims_dict[utils.preprocess_article_name(d['claim'])]).toarray()
        #claim = sparse.vstack(self.encoder.tokenize_claim(utils.preprocess_article_name(d['claim']))).toarray()

        evidences = []
        labels = []
        claims = []
        num_pos = min(len(self.claim_to_article[d['claim']]), 4)
        for idx in range(num_pos):
            processed = self.claim_to_article[d['claim']][idx]
            evidence = self.encoder.tokenize_claim(processed)
            evidence = sparse.vstack(evidence)
            evidence = evidence.toarray()
            evidences.append(evidence)
            claims.append(claim)
            labels.append([0,1])

        for j in range(num_pos, self.data_batch_size):
            if self.randomize:
                e = np.random.choice(d['evidence'])
            else:
                e = d['evidence'][j]
            processed = utils.preprocess_article_name(e.split("http://wikipedia.org/wiki/")[1])
            #evidence = articles_dict[processed]
            evidence = self.encoder.tokenize_claim(processed)
            if len(evidence)>0:
                evidence = sparse.vstack(evidence).toarray() 
                evidences.append(evidence)
                claims.append(claim)
                if processed in self.claim_to_article[d['claim']]:
                    labels.append([0,1])
                else:
                    labels.append([1,0])

        #claim = claim.expand(evidences.shape[0], claim.shape[0], claim.shape[1])
        return claims, evidences, labels 

    def on_epoch_end(self):
        #np.random.shuffle(self.indicies)
        pass

def to_torch_sparse_tensor(M, device="cuda:0"):
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
    result = np.full((len(arrays),) + tuple(max_sizes), fill_value)
    for i, a in enumerate(arrays):
      # The shape of this array `a`, turned into slices
      slices = tuple(slice(0,s) for s in sizes[i])

      # Overwrite a block slice of `result` with this array `a`
      result[i][slices] = a
    return result
