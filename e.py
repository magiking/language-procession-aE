#!/usr/bin/env python3

import sys
import os
import re
import operator
import math
import numpy as np
from collections import Counter
sys.path.append(os.path.relpath("./"))
import irStub


def cosineSimilarity(d1, d2):
    '''
    compute cosine similarity of two vectors d1, d2
    d1 and d2 need to be the same length
    '''
    dp = sum(map(operator.mul, d1, d2))
    el = lambda d: math.sqrt(len(d)**2)
    return dp / (el(d1) * el(d2))


def unigramCounts(doc):
    ''' return a dict with word unigram counts from document 'doc'. '''
    # Jake's crappy tokenizer from d1
    tokens = re.split(r'[\.,?!\'\";:\-\s`()]+', doc.rstrip())
    tokens = [ x.lower() for x in tokens ]
    return dict(Counter(tokens))


def retrieve(queries, doc_dicts):
    top3sets = []
    for q in queries:
        # compute unigram counts
        q_counts = unigramCounts(q)  
        # normalized
        q_counts = { k:q_counts[k]/len(q_counts) for k in q_counts }
        # print(f"count for query \"{q}\": \n {q_counts}")
        # print()

        # doc_vectors should match the tokens in query
        doc_vectors = []
        for dict_ in doc_dicts:
            doc_vectors.append([ dict_[k] if k in dict_.keys() else 0 for k in q_counts ])
        # print(f"document vectors: {doc_vectors}")
        # print()

        similarities = np.array([ cosineSimilarity(q_counts.values(), d) for d in doc_vectors ])
        # print(f"cosine similarities: \n{similarities} ")
        # print()

        top3indices = similarities.argsort()[::-1][:3] # sort in desc order and take top 3
        # print(f"top 3 similar documents: \n{top3indices} ")
        top3sets.append(top3indices)  
    return top3sets

def computeIDF(queries, doc_dicts):
    top3sets = []
    for q in queries:
        # compute unigram counts
        tf_query = unigramCounts(q) # tf for the query
        df = { word:0 for word in tf_query }

        for term in tf_query.keys():
            for doc in doc_dicts:
                if term in doc:
                    df[term] += 1

        # idf = np.log(len(doc_dicts) / (np.array(df.values()) + 1))
        idf = [np.log(len(doc_dicts) / (x + 1)) for x in df.values()]

        tf_vectors = [] # term freqs for each document
        for dict_ in doc_dicts:
            tf_vectors.append([ dict_[k] if k in dict_.keys() else 0 for k in tf_query])

        for v in tf_vectors:
            for i in range(len(v)):
                v[i] *= idf[i]

        tf_idf_query = list(tf_query.values())

        for i in range(len(tf_idf_query)):
            tf_idf_query[i] *= idf[i]

        similarities = np.array([ cosineSimilarity(tf_idf_query, d) for d in tf_vectors ])
        top3indices = similarities.argsort()[::-1][:3] # sort in desc order and take top 3
        top3sets.append(top3indices)  

    return top3sets


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 e.py " +
              "<document file>" +
              "<queries file>")
        sys.exit()

    contents, names =  irStub.parseAlternatingLinesFile(sys.argv[1]) 
    
    print(f"read in pages for {names}")

    queries, targets = irStub.parseAlternatingLinesFile(sys.argv[2])
    targetIDs = irStub.targetNumbers(targets, names)

    # straight up word counts
    tf_dicts = [] # would be archine
    for c in contents:
        tf_dicts.append(unigramCounts(c)) # unigramCounts replaces computeFeatures from irStub

    
    normalized_tfs= [] # or this would be archive
    for d in tf_dicts:
        normalized = d.copy()
        for term in normalized.keys():
            normalized[term] = normalized[term]/len(normalized.keys())
        normalized_tfs.append(normalized)

    # function that takes the queries, tfs, and targets, and all that
    #  and returns a list of results for each query
    # print(f"tf_dicts: {tf_dicts}")
    results = retrieve(queries, normalized_tfs)
    print(f"results: {results}")


    modelName = "raw term frequency model"
    irStub.scoreAllResults(queries, results, targetIDs, 
            f"{modelName} on {sys.argv[1]}")


    modelName2 = "TF-IDF Results"
    tf_idf_results = computeIDF(queries, tf_dicts)
    irStub.scoreAllResults(queries, tf_idf_results, targetIDs, f"{modelName2} on {sys.argv[1]}")

    modelName3 = "Normalized TF-IDF Results"
    norm_tf_idf_results = computeIDF(queries, normalized_tfs)
    irStub.scoreAllResults(queries, norm_tf_idf_results, targetIDs, f"{modelName3} on {sys.argv[1]}")
  
        

    
