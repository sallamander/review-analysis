"""A script for fitting non neural network models on text.

These models will simply fit on a term-frequency, inverse-document frequency 
matrix generated from the corpus.
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
from review_analysis.utils.eval_utils import intellegently_guess

if __name__ == '__main__':
    raw_reviews_fp = 'work/raw_reviews.pkl'
    ratios_fp = 'work/filtered_ratios.npy'

    with open(raw_reviews_fp, 'rb') as f: 
        raw_reviews = pickle.load(f) # `raw_reviews` are tokenized
    ratios = np.load(ratios_fp)
    review_txts = [' '.join(review) for review in raw_reviews]

    X_train, X_test, y_train, y_test = train_test_split(review_txts, ratios, 
                                                        test_size=0.2, 
                                                        random_state=609)
    
    vectorizer = TfidfVectorizer(max_features=10000)
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    print('Need to beat... {}'.format(intellegently_guess(y_train)))
