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

