import imp
import sys
sys.modules["sqlite"] = imp.new_module("sqlite")
sys.modules["sqlite3.dbapi2"] = imp.new_module("sqlite.dbapi2")
import nltk
import os
import json
import numpy as np
import tensorflow as tf
import unicodedata
import var

from csv import DictReader, DictWriter
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

STOP_WORDS = set(nltk.corpus.stopwords.words('english'))


def open_csv(path):
    '''
    HELPER FUNCTION

    Opens a CSV and returns a list of rows representing the CSV.

    Inputs:
      path: path to the csv file
    Outputs:
      rows: list of rows representing the original CSV file.
    '''
    with open(path, "r", encoding='mac_roman') as table:
        rows = [row for row in DictReader(table)]
        return rows


def get_fnc_data(stances_path, bodies_path):
    '''
    Parses the FNC data and returns the information in a usable
    format.

    Inputs:
      stances_path: path to FNC stances CSV
      bodies_path: path to FNC bodies CSV
    Outputs:
      headlines: list of FNC headlines
      bodies: list of FNC bodies corresponding to the headlines
      labels: list of labels for the corresponding headline (see
        FNC_LABELS in var.py)
      body_ids: list of body_ids corresponding to the bodies
    '''
    stances_file = open_csv(stances_path)
    bodies_file = open_csv(bodies_path)

    headlines = [row['Headline'] for row in stances_file]
    body_id_to_article = {
        int(row['Body ID']): row['articleBody'] for row in bodies_file}
    bodies = [body_id_to_article[int(row['Body ID'])] for row in stances_file]
    labels = [var.FNC_LABELS[row['Stance']] for row in stances_file]
    body_ids = [int(row['Body ID']) for row in stances_file]

    return headlines, bodies, labels, body_ids


def extract_tokens_from_binary_parse(parse):
    '''
    HELPER FUNCTION

    Remove the parenthesis from the parsed sentences and gets
    text tokens.

    Inputs:
      parse: parenthesis delimited parse sentences
    Output:
      list of text tokens from the input
    '''
    return parse.replace('(', ' ').replace(')', ' ').replace(
        '-LRB-', '(').replace('-RRB-', ')').split()


def get_snli_examples(jsonl_path, skip_no_majority=True,
                      limit=None, use_neutral=True):
    '''
    HELPER FUNCTION

    Given the SNLI data, gets the sentences for each data point
    as well as the label and saves it in a usable form.

    Inputs:
      jsonl_path: path to SNLI jsonl CSV
      skip_no_majority: boolean indicating whether to skip data
        point if there's no majority label agreement
      limit: limit on amount of data to extract from SNLI
      use_neutral: boolean indicating whether to get data with
        neutral labels or not.
    Outputs:
      examples: list of tuples where each tuple contains the
        label and both the sentences corresponding to a data point.
    '''
    examples = []
    skipped = 0
    with open(jsonl_path) as f:
        for i, line in enumerate(f):
            if limit is not None and i - skipped >= limit:
                break
            data = json.loads(line)
            label = data['gold_label']
            if label == "neutral" and not use_neutral:
                skipped += 1
                continue
            s1 = ' '.join(
                extract_tokens_from_binary_parse(
                    data['sentence1_binary_parse']))
            s2 = ' '.join(
                extract_tokens_from_binary_parse(
                    data['sentence2_binary_parse']))
            if skip_no_majority and label == '-':
                skipped += 1
                continue
            examples.append((label, s1, s2))
    return examples


def get_snli_data(jsonl_path, limit=None, use_neutral=True):
    '''
    Extracts data from a the SNLI jsonl data file and returns
    the left sentence, right sentence, and corresponding label
    in 3 seperate lists.

    Inputs:
      jsonl_path: path to SNLI jsonl file
      limit: limit on the number of SNLI data points to get
      use_neutral: boolean indicating whether to get data points
        labeled neutral or not
    Outputs:
      left: list of sentences representing the first sentence
      right: list of sentences representing the second sentence
      labels: list of labels for the corresponding left and right
        sentences (see SNLI_LABELS in var.py)
    '''
    data = get_snli_examples(
        jsonl_path=jsonl_path,
        limit=limit,
        use_neutral=use_neutral)
    left = [s1 for _, s1, _ in data]
    right = [s2 for _, _, s2 in data]
    labels = [var.SNLI_LABELS[l] for l, _, _ in data]
    return left, right, labels


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

    with open(path, 'r') as f:
        for item in f:
            data = json.loads(item)
            claim_set.add(data["claim"])
            if data["verifiable"] == "VERIFIABLE":
                evidence_articles = set()
                for evidence in data["all_evidence"]:
                    article_name = unicodedata.normalize('NFC', evidence[2])

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
                        labels.append(var.FEVER_LABELS[data["label"]])

                    total_ev += 1
                num_train += 1

    print("Num Distinct Claims", num_train)
    print("Num Data Points", total_ev)

    return claims, labels, article_list, claim_set


def get_relevant_articles(wikidata_path, article_list):
    '''
    HELPER FUNCTION

    Given a article_list containing names of articles, get the
    wikipedia text for each of the articles.

    Inputs:
      wikidata_path: path to wiki dump
      article_list: list of article names to find
    Outputs:
      bodies: list of full text articles corresponding to the
        original article_list
    '''
    article_dict = {article: None for article in article_list}

    wiki_files = [os.path.join(wikidata_path, f)
                  for f in os.listdir(wikidata_path)]

    total_num_files = 0
    for file in wiki_files:
        print(file)
        with open(file, 'r') as f:
            for item in f:
                data = json.loads(item)
                key = unicodedata.normalize('NFC', data["id"])
                if key in article_dict:
                    article_dict[key] = data["text"]
                total_num_files += 1

    print("Total Num Wiki Articles", total_num_files)

    bodies = []
    for article in article_list:
        bodies.append(article_dict[article])

    return bodies


def get_fever_data(jsonl_path, wikidata_path):
    '''
    Extracts claims, article text, corresponding labels, and
    a set of unique claims from the input FEVER jsonl data and
    provided wiki dump.

    Inputs:
      jsonl_path: path to FEVER jsonl file
      wikidata_path: path to Wiki dump
    Outputs:
      claims: list of claims for each data point
      bodies: list of article text for each data point
      labels: label for each data point (see FEVER_LABELS in
        var.py)
      claim_set: set of unique claims
    '''
    claims, labels, article_list, claim_set = extract_fever_jsonl_data(
        jsonl_path)
    print(claims[:20], article_list[:20])
    bodies = get_relevant_articles(wikidata_path, article_list)
    return claims, bodies, labels, claim_set


def get_vectorizers(train_data, MAX_FEATURES):
    '''
    Given training data, create bag of words, tf, and tfidf
    vectorizers.

    Input:
      train_data: data to create vectorizers with
      MAX_FEATURES: the maximum number of terms to consider in
        vocabulary
    Outputs:
      bow_vectorizer: bag of words vectorizer
      tfreq_vectorizer: TF vectorizer
      tfidf_vectorizer: TFIDF vectorizer
    '''
    train_data = list(set(train_data))

    bow_vectorizer = CountVectorizer(
        max_features=MAX_FEATURES,
        stop_words=STOP_WORDS)
    bow = bow_vectorizer.fit_transform(train_data)

    tfreq_vectorizer = TfidfTransformer(use_idf=False).fit(bow)

    tfidf_vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES,
        stop_words=STOP_WORDS).fit(train_data)

    return bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer


def get_feature_vectors(headlines, bodies, bow_vectorizer,
                        tfreq_vectorizer, tfidf_vectorizer):
    '''
    Convert data into feature vectors where the first
    NUM_FEATURES elements is the TF vector for the first
    document and the next NUM_FEATURES elements is the TF
    vector for the second document. The cosine distance
    between the TFIDF values of the vectors are then appended
    to this vector.

    Inputs:
      headlines: the first document to get tf values from
      bodies: the second document to get tf values from
      bow_vectorizer: trained bow vectorizer
      tfreq_vectorizer: trained tfreq vectorizer
      tfidf_vectorizer: trained tfidf vectorizer
    Outputs:
      len(headlines) by (2*NUM_FEATURES+1) sized vector for
      each headline, body pair.
    '''
    feature_vectors = []

    for i in range(len(headlines)):
        if i % 5000 == 0:
            print("    Processed", i, "out of", len(headlines))

        headline = headlines[i]
        body = bodies[i]

        headline_bow = bow_vectorizer.transform([headline]).toarray()
        headline_tf = tfreq_vectorizer.transform(
            headline_bow).toarray()[0].reshape(1, -1)
        headline_tfidf = tfidf_vectorizer.transform(
            [headline]).toarray().reshape(1, -1)

        body_bow = bow_vectorizer.transform([body]).toarray()
        body_tf = tfreq_vectorizer.transform(body_bow).toarray()[
            0].reshape(1, -1)
        body_tfidf = tfidf_vectorizer.transform(
            [body]).toarray().reshape(1, -1)

        tfidf_cos = cosine_similarity(
            headline_tfidf,
            body_tfidf)[0].reshape(
            1,
            1)
        feature_vector = np.squeeze(np.c_[headline_tf, body_tf, tfidf_cos])

        feature_vectors.append(feature_vector)

    print("    Number of Feature Vectors:", len(feature_vectors))

    feature_vectors = np.asarray(feature_vectors)

    return feature_vectors


def get_relational_feature_vectors(feature_vectors):
    '''
    Create relational feature vectors where the first NUM_FEATURES
    elements are the square of the difference between the feature
    vectors of the TF values for the two documents at that
    corresponding vertex. The next NUM_FEATURES elements represent
    the product of the two corresponding TF values. The value of the
    cosine distance of the TFIDF values are then appended to this
    vector.

    Inputs:
      feature_vectors: feature_vectors extracted from
        the above get_feature_vectors
    Outputs:
      len(feature_vectors) by (2*NUM_FEATURES + 1) vectors
      corresponding to each input feature vector
    '''
    # Calculate number of features per document tf vector
    NUM_FEATURES = len(feature_vectors[0]) // 2

    relational_feature_vectors = np.zeros((len(feature_vectors), 10001))

    for i in range(len(feature_vectors)):
        if (i % 5000) == 0:
            print("    Processed", i, "out of", len(feature_vectors))

        tf_vector_1 = feature_vectors[i][:NUM_FEATURES]
        tf_vector_2 = feature_vectors[i][NUM_FEATURES:2 * NUM_FEATURES]
        tfidf = feature_vectors[i][2 * NUM_FEATURES:]

        dist_vector = np.power(tf_vector_1 - tf_vector_2, 2)
        mag_vector = np.multiply(tf_vector_1, tf_vector_2)

        relational_vector = np.concatenate([dist_vector, mag_vector, tfidf])

        relational_feature_vectors[i] = relational_vector

    print("    Number of Relational Feature Vectors:",
          len(relational_feature_vectors))

    return relational_feature_vectors


def save_predictions(pred, actual, save_file):
    '''
    Saves predictions of model to CSV file. Uses FNC_LABELS_REV
    to translate from numbers to string labels.

    Inputs:
        pred: numpy array of numeric predictions
        save_file: path to file to save predictions
    '''

    with open(save_file, 'w') as csvfile:
        fieldnames = ['Stance', 'Actual']
        writer = DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for i in range(len(pred)):
            writer.writerow(
                {'Stance': var.FNC_LABELS_REV[pred[i]], 'Actual': var.FNC_LABELS_REV[actual[i]]})


def get_composite_score(pred, labels):
    '''
    Calculates FNC composite score given predictions and labels.
    The scoring is as follows:
      0.25 points for a correct related/unrelated label where
        related means agree, disagree, or discuss
      0.75 additional points for every correct related label

    Inputs:
      pred: list of predictions (see FNC_LABELS in var.py for
        mapping)
      labels: list of expected predictions (see FNC_LABELS in
        var.py for mapping)
    '''
    score = 0
    for i in range(len(pred)):
        # Unrelated label
        if labels[i] == var.FNC_LABELS['unrelated'] and pred[i] == var.FNC_LABELS['unrelated']:
            score += 0.25

        # Related label
        if labels[i] != var.FNC_LABELS['unrelated'] and pred[i] != var.FNC_LABELS['unrelated']:
            score += 0.25
            if labels[i] == pred[i]:
                score += 0.75
    return score


def get_prediction_accuracies(pred, labels, num_labels):
    '''
    Calculates the accuracy of the predictions for each label.
    Uses FNC_LABELS in var.py as the possible labels to
    consider.

    Inputs:
      pred: model predictions
      labels: expected labels
    Outputs:
      len(var.FNC_LABELS) sized list with accuracies for each
      label.
    '''
    correct = [0 for _ in range(num_labels)]
    total = [0 for _ in range(num_labels)]

    for i in range(len(pred)):
        total[labels[i]] += 1
        if pred[i] == labels[i]:
            correct[labels[i]] += 1

    # Avoid dividing by 0 case
    for label in range(len(total)):
        if total[label] == 0:
            total[label] += 1

    return [correct[i] / total[i] for i in range(len(total))]


def remove_stop_words(sentences):
    '''
    Removes stopwords from a sentence.

    Inputs:
      sentences: list of sentences to tokenize
    Outputs:
      sentences: list of list of tokens corresponding to each
        original sentence
    '''
    sentences = [[word for word in nltk.word_tokenize(
        sentence.lower()) if word not in STOP_WORDS] for sentence in sentences]
    sentences = [' '.join(word for word in sentence) for sentence in sentences]
    return sentences


def get_average_embeddings(sentences, embeddings, embedding_size=300):
    '''
    Given a list of texts, and word2vec embeddings, produce a 300
    length avg embedding vector for each text.

    Inputs:
      sentences: list of sentences to get average embeddings for
      embeddings: word2vec embeddings
      embedding_size: size of embeddings for each word
    Outputs:
      avg_embeddings: list of embedding_size vectors where each
        vector corresponds to an original sentence.
    '''
    sentences = [nltk.word_tokenize(sentence.lower())
                 for sentence in sentences]
    avg_embeddings = np.zeros((len(sentences), embedding_size))

    for i, sentence in enumerate(sentences):
        if len(sentence) == 0:
            continue

        if i % 5000 == 0:
            print("    Processed", i, "out of", len(sentences))

        count = 0.0
        for word in sentence:
            if word in embeddings.vocab:
                count += 1
                avg_embeddings[i] += embeddings[word]
        if count > 0:
            avg_embeddings[i] /= count

    return avg_embeddings


def print_model_results(f, set_name, pred, labels, d_pred,
                        d_labels, p_loss, d_loss, l2_loss, use_domains):
    '''
    Prints the results from the current training epoch.

    Inputs:
      f: file to write output training output to
      set_name: string indicating Train, Val or Test
      pred: list of label predictions
      labels: list of expected labels
      d_pred: list of domain predictions
      d_labels: list of expected domain labels
      p_loss: prediction loss
      d_loss: domain loss
      l2_loss: l2 regularization loss
      use_domains: boolean indicating whether to domain adaptation
        was used or not.
    '''
    print("\n    " + set_name + "  Label Loss =", p_loss)
    print("    " + set_name + "  Domain Loss =", d_loss)
    print("    " + set_name + "  Regularization Loss =", l2_loss)
    print("    " + set_name + "  Total Loss =", p_loss + d_loss + l2_loss)

    f.write("\n    " + set_name + "  Label Loss = " + str(p_loss) + "\n")
    f.write("    " + set_name + "  Domain Loss = " + str(d_loss) + "\n")
    f.write(
        "    " +
        set_name +
        "  Regularization Loss = " +
        str(l2_loss) +
        "\n")
    f.write("    " + set_name + "  Total Loss = " +
            str(p_loss + d_loss + l2_loss) + "\n")

    composite_score = get_composite_score(pred, labels)
    print("    " + set_name + "  Composite Score", composite_score)
    f.write(
        "    " +
        set_name +
        "  Composite Score " +
        str(composite_score) +
        "\n")

    pred_accuracies = get_prediction_accuracies(
        pred, labels, len(var.FNC_LABELS))
    print("    " + set_name + "  Label Accuracy", pred_accuracies)
    f.write(
        "    " +
        set_name +
        "  Label Accuracy [" +
        ', '.join(
            str(acc) for acc in pred_accuracies) +
        "]\n")

    if use_domains:
        domain_accuracies = get_prediction_accuracies(
            d_pred, d_labels, len(var.DOMAIN_MAPPING))
        print("    " + set_name + "  Domain Accuracy", domain_accuracies)
        f.write(
            "    " +
            set_name +
            "  Domain Accuracy [" +
            ', '.join(
                str(acc) for acc in domain_accuracies) +
            "]\n")


def remove_data_with_label(labels_to_remove, headlines,
                           bodies, labels, domains, additional=None):
    '''
    Removes all data points with specified labels

    Inputs:
      labels_to_remove: set of labels to remove from data
      headlines: list of headlines
      bodies: list of bodies
      labels: list of labels
      domains: list of domains
      additional (optional): any additional lists to
        remove data from
    outputs:
      headlines, bodies, labels, domains, and any additional
      with the data with labels in labels_to_remove removed.
    '''
    throwaway_indices = [i for i, x in enumerate(
        labels) if x in labels_to_remove]

    for i in sorted(throwaway_indices, reverse=True):
        del headlines[i]
        del bodies[i]
        del labels[i]
        del domains[i]
        if additional is not None:
            del additional[i]

    result = [headlines, bodies, labels, domains]
    if additional is not None:
        result.append(additional)
    return result


def get_body_sentences(bodies):
    '''
    Given text, gets the sentences from each body.

    Inputs:
      bodies: list of text bodies
    Outputs:
      result: list of list of sentences
    '''
    result = []
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    for body in bodies:
        sents = tokenizer.tokenize(body)
        result.append(sents)
    return result


def select_best_body_sentences(headlines, bodies, tfidf_vectorizer):
    '''
    Selects the sentence most closely related to headline
    from sentences in body according to tfidf similarity

    Inputs:
      headlines: list of headlines
      bodies: list of body texts
      tfidf_vectorizer: trained tfidf vectorizer
    Outputs:
      best_sents: list of sentences with each sentence
        being the best sentence for each of the given
        headlines
    '''
    best_sents = []

    for i, headline in enumerate(headlines):
        if i % 1000 == 0:
            print('Finished ' + str(i) + ' out of ' +
                  str(len(headlines)) + ' headlines')

        best_sent = None
        best_tfidf = -1

        for j, body_sent in enumerate(bodies[i]):
            headline_tfidf = tfidf_vectorizer.transform(
                [headline]).toarray().reshape(1, -1)
            body_tfidf = tfidf_vectorizer.transform(
                [body_sent]).toarray().reshape(1, -1)

            tfidf_cos = cosine_similarity(
                headline_tfidf,
                body_tfidf)[0].reshape(1, 1)

            if tfidf_cos > best_tfidf:
                best_tfidf = tfidf_cos
                best_sent = body_sent

        best_sents.append(best_sent)

    return best_sents
