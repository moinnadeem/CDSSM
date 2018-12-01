import unicodedata
import json
import unicodedata
import joblib
import string
import pickle
import nltk
import numpy as np
from scipy import sparse
from sklearn.preprocessing import LabelEncoder
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm.autonotebook import tqdm 

FEVER_LABELS = {'SUPPORTS': 0, 'REFUTES': 1}

def tokenize_helper(inp):
    print(inp)
    return tokenize_claim(inp[0], inp[1], inp[2])

class ClaimEncoder(object):
    def __init__(self):
        with open("feature_encoder.pkl", "rb") as f:
            self.feature_encoder = pickle.load(f)
            
        with open("encoder.pkl", "rb") as f:
            self.encoder = pickle.load(f)
            
    def tokenize_claim(self, c):
        """
        Input: a string that represents a single claim
        Output: a list of 3x|vocabulary| arrays that has a 1 where the letter-gram exists.
        """
        encoded_vector = []
        c = preprocess_article_name(c)
        c = "! {} !".format(c)
        for ngram in nltk.ngrams(nltk.word_tokenize(c), 3):
            arr = sparse.lil_matrix((3, len(self.encoder.__dict__['classes_'])))
            for idx, word in enumerate(ngram):
                for letter_gram in nltk.ngrams("#" + word + "#", 3):
                    s = "".join(letter_gram)
                    letter_idx = self.feature_encoder[s]
                    arr[idx, letter_idx] = 1
            encoded_vector.append(arr)
        return encoded_vector
    
    def create_encodings(self, claims, train_dict, write_to_file=False):
        processed_claims = generate_all_tokens(claims)
        all_evidence = []

        for query in tqdm(train_dict):
            all_evidence.extend([preprocess_article_name(i) for i in query['evidence']])

        processed_claims.extend(generate_all_tokens(list(set(all_evidence))))

        possible_tokens = list(set(processed_claims))
        possible_tokens.append("OOV")

        self.encoder = LabelEncoder()
        self.encoder.fit(np.array(sorted(possible_tokens)))
        
        self.feature_encoder = {}
        for idx, e in tqdm(enumerate(self.encoder.classes_)):
            self.feature_encoder[e] = idx
            
        if write_to_file:
            joblib.dump(self.feature_encoder, "feature_encoder.pkl")
            joblib.dump(self.encoder, "encoder.pkl")
            
def generate_all_tokens(arr):
    all_tokens = []
    for unprocessed_claim in tqdm(arr):
        c = preprocess_article_name(unprocessed_claim)
        c = "! {} !".format(c)
        for word in c.split():
            letter_tuples = list(nltk.ngrams("#" + word + "#", 3))
            letter_grams = []
            for l in letter_tuples:
                letter_grams.append("".join(l))
            all_tokens.extend(letter_grams)
    return all_tokens

def extract_fever_jsonl_data(path):
    '''
    HELPER FUNCTION

    Extracts lists of headlines, labels, articles, and a set of
    all distinct claims from a given FEVER jsonl file.

    Inputs:
      path: path to FEVER jsonl file
    Outputs:
      claims: list of claims for each data point
      labels: list of labels for each claim (see FEVER_LABELS in
        var.py)
      article_list: list of names of articles corresponding to
        each claim
      claim_set: set of distinct claim
    '''
    num_train = 0
    total_ev = 0

    claims = []
    labels = []
    article_list = []
    claim_set = set()
    claim_to_article = {}
    with open(path, 'r') as f:
        for item in f:
            data = json.loads(item)
            claim_set.add(data["claim"])
            if data["verifiable"] == "VERIFIABLE":
                evidence_articles = set()
                for evidence in data["evidence"][0]:
                    article_name = unicodedata.normalize('NFC', evidence[2])
                    article_name = preprocess_article_name(article_name)
                    
                    # Ignore evidence if the same article has
                    # already been used before as we are using
                    # the entire article and not the specified
                    # sentence.
                    if article_name in evidence_articles:
                        continue
                    else:
                        article_list.append(article_name)
                        evidence_articles.add(article_name)
                        claims.append(data["claim"])
                        labels.append(FEVER_LABELS[data["label"]])
                        if data['claim'] not in claim_to_article:
                            claim_to_article[data['claim']] = [article_name]
                        else:
                            claim_to_article[data['claim']].append(article_name)

                    total_ev += 1
                num_train += 1

    print("Num Distinct Claims", num_train)
    print("Num Data Points", total_ev)

    return claims, labels, article_list, claim_set, claim_to_article

def preprocess_article_name(s):
    s = s.replace("_", " ")
    s = s.replace("-LRB-", "(")
    s = s.replace("-RRB-", ")")
    s = s.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))
    s = strip_accents(s)
    s = s.replace("’", "")
    s = s.replace("“", '')
    s = ''.join([i if ord(i) < 128 else ' ' for i in s])
    s = ' '.join(s.split())
    return s.lower().rstrip()

def strip_accents(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                  if unicodedata.category(c) != 'Mn')

def char_ngrams(s, n):
    s = "#" + s + "#"
    return [s[i:i+n] for i in range(len(s) - 2)]

def parallel_process(array, function, n_jobs=12, use_kwargs=False, front_num=3):
    """
        A parallel version of the map function with a progress bar. 

        Args:
            array (array-like): An array to iterate over.
            function (function): A python function to apply to the elements of array
            n_jobs (int, default=16): The number of cores to use
            use_kwargs (boolean, default=False): Whether to consider the elements of array as dictionaries of 
                keyword arguments to function 
            front_num (int, default=3): The number of iterations to run serially before kicking off the parallel job. 
                Useful for catching bugs
        Returns:
            [function(array[0]), function(array[1]), ...]
    """
    #We run the first few iterations serially to catch bugs
    if front_num > 0:
        front = [function(**a) if use_kwargs else function(a) for a in array[:front_num]]
    #If we set n_jobs to 1, just run a list comprehension. This is useful for benchmarking and debugging.
    if n_jobs==1:
        return front + [function(**a) if use_kwargs else function(a) for a in tqdm(array[front_num:])]
    #Assemble the workers
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        #Pass the elements of array into function
        if use_kwargs:
            futures = [pool.submit(function, **a) for a in array[front_num:]]
        else:
            futures = [pool.submit(function, a) for a in array[front_num:]]
        kwargs = {
            'total': len(futures),
            'unit': 'it',
            'unit_scale': True,
            'leave': True
        }
        #Print out the progress as tasks complete
        for f in tqdm(as_completed(futures), **kwargs):
            pass
    out = []
    #Get the results from the futures. 
    for i, future in tqdm(enumerate(futures)):
        try:
            out.append(future.result())
        except Exception as e:
            out.append(e)
    return front + out
