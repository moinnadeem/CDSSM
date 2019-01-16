# Nishant Nikhil (i.nishantnikhil@gmail.com)
# An implementation of the Deep Semantic Similarity Model (DSSM) found in [1].
# [1] Shen, Y., He, X., Gao, J., Deng, L., and Mesnil, G. 2014. A latent semantic model
#         with convolutional-pooling structure for information retrieval. In CIKM, pp. 101-110.
#         http://research.microsoft.com/pubs/226585/cikm2014_cdssm_final.pdf
# [2] http://research.microsoft.com/en-us/projects/dssm/
# [3] http://research.microsoft.com/pubs/238873/wsdm2015.v3.pdf

import torch 
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F


LETTER_GRAM_SIZE = 3 # See section 3.2.
WINDOW_SIZE = 3 # See section 3.2.
TOTAL_LETTER_GRAMS = 8246 # Determined from data. See section 3.2.
WORD_DEPTH = WINDOW_SIZE * TOTAL_LETTER_GRAMS # See equation (1).
WORD_DEPTH = 29244 # See equation (1).
# Uncomment it, if testing
# WORD_DEPTH = 1000
K = 300 # Dimensionality of the max-pooling layer. See section 3.4.
L = 128 # Dimensionality of latent semantic space. See section 3.5.
J = 4 # Number of random unclicked documents serving as negative examples for a query. See section 4.
FILTER_LENGTH = 3 # We only consider one time step for convolutions.


def kmax_pooling(x, dim, k):
    index = x.topk(k, dim = dim)[1].sort(dim = dim)[0]
    return x.gather(dim, index)

class CDSSM(nn.Module):
    def __init__(self):
        super(CDSSM, self).__init__()
        # layers for query
        self.query_conv = nn.Conv1d(WORD_DEPTH, K, FILTER_LENGTH)
        # adding Xavier-He initialization
        torch.nn.init.xavier_uniform_(self.query_conv.weight)

        # learn the semantic representation
        self.query_sem = nn.Linear(K, L)
        torch.nn.init.xavier_uniform_(self.query_sem.weight)

        # adding a hidden layer
        self.query_hidden = nn.Linear(L, L//2)
        self.document_hidden = nn.Linear(L, L//2)

        # dropout for regularization
        print("Model with dropout and tanh.")
        self.query_dropout = nn.Dropout(0.2)
        self.document_dropout = nn.Dropout(0.2)

        # layers for docs
        self.doc_conv = nn.Conv1d(WORD_DEPTH, K, FILTER_LENGTH)
        torch.nn.init.xavier_uniform_(self.doc_conv.weight)
        self.doc_sem = nn.Linear(K, L)
        torch.nn.init.xavier_uniform_(self.doc_sem.weight)

        # learning gamma
        self.learn_gamma = nn.Conv1d(1, 1, 1)
        torch.nn.init.xavier_uniform_(self.learn_gamma.weight)

        # adding batch norm
        self.q_norm = nn.BatchNorm1d(K)
        self.doc_norm = nn.BatchNorm1d(K)

    def forward(self, q, pos):
        # Query model. The paper uses separate neural nets for queries and documents (see section 5.2).
        # To make it compatible with Conv layer we reshape it to: (batch_size, WORD_DEPTH, query_len)
        q = q.transpose(1,2)

        # In this step, we transform each word vector with WORD_DEPTH dimensions into its
        # convolved representation with K dimensions. K is the number of kernels/filters
        # being used in the operation. Essentially, the operation is taking the dot product
        # of a single weight matrix (W_c) with each of the word vectors (l_t) from the
        # query matrix (l_Q), adding a bias vector (b_c), and then applying the tanh activation.
        # That is, h_Q = tanh(W_c • l_Q + b_c). Note: the paper does not include bias units.
        q_c = torch.tanh(self.query_conv(q))
        q_c = self.q_norm(q_c)

        # Next, we apply a max-pooling layer to the convolved query matrix.
        q_k = kmax_pooling(q_c, 2, 1)
        q_k = q_k.transpose(1,2)
        q_k = self.query_dropout(q_k)

        # In this step, we generate the semantic vector represenation of the query. This
        # is a standard neural network dense layer, i.e., y = tanh(W_s • v + b_s). Again,
        # the paper does not include bias units.
        q_s = torch.tanh(self.query_sem(q_k))

        #q_s = q_s.resize(L)
        # # The document equivalent of the above query model for positive document
        pos = pos.squeeze()
        pos = pos.transpose(1,2)

        pos_c = torch.tanh(self.doc_conv(pos))
        pos_c = self.doc_norm(pos_c)

        pos_k = kmax_pooling(pos_c, 2, 1)
        pos_k = pos_k.transpose(1,2)
        pos_k = self.document_dropout(pos_k)
        pos_s = torch.tanh(self.doc_sem(pos_k))
        #pos_s = pos_s.resize(L)
        # # The document equivalent of the above query model for negative documents

        # Now let us calculates the cosine similarity between the semantic representations of
        # a queries and documents
        # dots[0] is the dot-product for positive document, this is necessary to remember
        # because we set the target label accordingly

        q_s = F.tanh(self.query_hidden(q_s))
        pos_s = F.tanh(self.document_hidden(pos_s))
        q_s = self.query_dropout(q_s)
        pos_s = self.document_dropout(pos_s)
        
        q_s = q_s.squeeze()
        pos_s = pos_s.squeeze()

        dots = torch.mm(q_s, pos_s.transpose(0,1)).diag() 
        dots = dots / (torch.norm(q_s)*torch.norm(pos_s))  # divide by the norm to make it cosine distance

        # dots is a list as of now, lets convert it to torch variable
        #dots = torch.stack(dots)

        # In this step, we multiply each dot product value by gamma. In the paper, gamma is
        # described as a smoothing factor for the softmax function, and it's set empirically
        # on a held-out data set. We're going to learn gamma's value by pretending it's
        # a single 1 x 1 kernel.

        #dots = torch.stack([dots, dots],1)
        #dots = dots.unsqueeze(2)

        # We transform a scalar into a 3D vector
        dots = dots.unsqueeze(0)
        dots = dots.unsqueeze(0)
        with_gamma = self.learn_gamma(dots)
        
        # Finally, we use the softmax function to calculate P(D+|Q).
        #prob = F.logsigmoid(with_gamma)
        #prob = F.softmax(with_gamma)
        return with_gamma 
