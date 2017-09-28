#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Perform some analysis on the top components of SVD.

This script takes articles from collected academic papers on school violence, then
applies LSA to them to create compact feature vectors.

@author: Aman Arya
"""

import os
import pickle
import numpy

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from pylab import *


# Load txt dataset

print('Loading data...')
os.chdir('../..')

vdata, labels = pickle.load(open(os.getcwd()+'/Data/B_data.pickle', 'rb'))

print("Number of training examples " + str((len(vdata))))

# Tf-idf Vectorizer
#   - Strips out “stop words”
#   - Filters out terms that occur in more than half of the docs (max_df=0.5)
#   - Filters out terms that occur in only one document (min_df=2)
#   - Selects the 10,000 most frequently occuring words in the corpus
#   - Normalizes the vector (L2 norm of 1.0) to normalize the effect of
#     document length on the tf-idf values
#   - Chooses only single words to be most important (ngram_range=(1,1))

vectorizer = TfidfVectorizer(max_df=0.6, max_features=10000,
                             min_df=2, stop_words='english',
                             use_idf=True, norm='l2', ngram_range=(1,1))

for file in vdata:
    file.decode('utf-8')

vdata_tfidf = vectorizer.fit_transform(vdata)

print("  Actual number of tfidf features: %d" % vdata_tfidf.get_shape()[1])

# Get the words that correspond to each of the features.
feat_names = vectorizer.get_feature_names()

print("\nPerforming dimensionality reduction using LSA")

# Project the tfidf vectors onto the first N principal components.
svd = TruncatedSVD(100)
lsa = svd.fit(vdata_tfidf)

t = []
sizes = [10, 15, 10, 10]
a = 0
k = 0
for s in sizes:
    s = s + a
    terms = []
    weights = []
    c = -1
    for compNum in range(k, s):
        comp = lsa.components_[compNum]

        # Sort the weights in the first component, and get the indeces
        indices = numpy.argsort(comp).tolist()

        # Reverse the indeces, so we have the largest weights first.
        indices.reverse()

        for weightIndex in indices[0:10]:

            if feat_names[weightIndex] in terms:
                # if feature already appears, add more weight to it
                weights[c] += comp[weightIndex]
            else:
                terms.append(feat_names[weightIndex])
                weights.append(comp[weightIndex])
                c += 1

    k = s + 1
    a = s
    t.append(zip(terms, weights))

featured_words = dict(zip(labels, t))

# Save features
d = os.getcwd() + '/Data/'
pickle.dump(featured_words, open(d+'features', 'wb'))

# Load features
f = pickle.load(open(d+'features', 'rb'))
print('\n' + str(f['Sexual Harrassment'][0])) # (u'coercive', 0.27306948044831902)

