# Nigel Ward, UTEP, October 2018
# Updated by Angel Garcia, UTEP, July 2020
# Speech and Language Processing
# Assignment E: Information Retrieval

# This is just a skeleton that needs to be fleshed out.
# It is not intended as an example of good Python style

import numpy as np
import sys

def parseAlternatingLinesFile(file):
    """ read a sequence of pairs of lines
    e.g. text of webpage(s), name/URL
    """
    sequenceA = []
    sequenceB = []

    with open(file, mode="r", encoding="utf-8") as f:
        for i,line in enumerate(f):
            if i % 2:
                sequenceB.append(line.strip())
            else:
                sequenceA.append(line.strip())

    return sequenceA, sequenceB

def generateCharTrigrams(text):
    """Generate Character Trigrams from Text"""
    for i in range(len(text)-3+1):
        yield text[i:i+3]

def computeFeatures(text, trigramInventory):        
    """Computes the count of trigrams.
    Trigrams can catch some similarities
    (e.g. between  "social" and "societal" etc.)
    
    But really should be replaced with something better
    """
    counts = {}
    for trigram in generateCharTrigrams(text):
        if trigram in trigramInventory:
            counts[trigram] = (1 if trigram not in counts
                     else counts[trigram] + 1)   
    return counts
   

def computeSimilarity(dict1, dict2):
    """Compute the similarity between 2 dictionaries of trigtrams

    Ad-hoc and inefficient.
    """
    
    keys_d1 = set(dict1.keys())
    keys_d2 = set(dict2.keys())
    matches = keys_d1 & keys_d2
    
    similarity = len(matches) / len(dict2)
    #print(f"Similarity: {similarity:.3f}")

    return similarity

def retrieve(queries, trigramInventory, archive):     
    """returns an array: for each query, the top 3 results found"""
    top3sets = []
    for query in queries:
        #print(f"query is {query}")
        
        q = computeFeatures(query, trigramInventory)
        #print(f"query features are \n{q}")

        similarities = [computeSimilarity(q, d) for d in archive] 
        
        #print(similarities)
        top3indices = np.argsort(similarities)[0:3]
        #print(f"top three indices are {top3indices}")
        
        top3sets.append(top3indices)  
    return top3sets

def valueOfSuggestion(result, position, targets):
    weight = [1.0, .5, .25]
    if result in targets:
        return weight[max(position, targets.index(result))]
    else:
        return 0


def scoreResults(results, targets):   #-----------------------------
    '''
    give a score to the suggested documet based on the actual(labled from test/train data)
    relevance of that doc to the query.
    '''
    merits = [valueOfSuggestion(results[i], i, targets) for i in range(3)]
    return sum(merits)


def scoreAllResults(queries, results, targets, descriptor):   
    '''
    queries; list of lists [ [ results for q1 ], [ q2 results ], ... ]
    '''
    print()
    print(f"Scores for {descriptor}")
    scores = [(q, r, t, scoreResults(r, t)) 
            for q, r, t in zip(queries, results, targets)]
    for q, r, t, s in scores:
        print(f"for query: {q}")
        print(f"  results = \n{r}")
        print(f"  targets = \n{t}")
        print(f"  score = {s:.3f}")

    all_scores = [s for _,_,_,s in scores]
    overallScore = np.mean(all_scores)
    print(f"All Scores:\n{all_scores}")
    print(f"Overall Score: {overallScore:.3f}")

    return overallScore

def pruneUniqueNgrams(ngrams):
    twoOrMore = {} 
    print("Before pruning: " +
            f"{len(ngrams)} ngrams across all documents")

    twoOrMore = {k:v for k,v in ngrams.items() if ngrams[k] > 1}

    print("After pruning: " +
            f"{len(twoOrMore)} ngrams across all documents")
    
    return twoOrMore

def findAllNgrams(contents):
    allTrigrams = {}
    
    for text in contents:
        for tri in generateCharTrigrams(text):
            allTrigrams[tri] = (1 if tri not in allTrigrams
                    else allTrigrams[tri] + 1)
    return allTrigrams


def targetNumbers(targets, nameInventory):
    """targets is a list of strings, each a sequence of names"""
    targetIDs = []
    for target in targets: # string of 3 names
      threeNumbers = [] 
      for name in target.split():
          threeNumbers.append(nameInventory.index(name))
      targetIDs.append(threeNumbers)
    return targetIDs
          

if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("Usage: python irStub.py " +
              "<document file>" +
              "<queries file>")
        sys.exit()

    print("......... irStub .........")
    
    contents, names =  parseAlternatingLinesFile(sys.argv[1]) 
    
    print(f"read in pages for {names}")
    
    trigramInventory = pruneUniqueNgrams(findAllNgrams(contents))
    archive = [computeFeatures(line, trigramInventory) 
            for line in contents]

    queries, targets = parseAlternatingLinesFile(sys.argv[2])
    targetIDs = targetNumbers(targets, names)
    results = retrieve(queries, trigramInventory, archive)
    modelName = "silly character trigram model"
    
    scoreAllResults(queries, results, targetIDs, 
            f"{modelName} on {sys.argv[1]}")
