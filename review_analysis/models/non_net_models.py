"""A script for fitting non neural network models on text.

These models will simply fit on a term-frequency, inverse-document frequency 
matrix generated from the corpus.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split

if __name__ == '__main__':
    reviews_df_fp = 'work/reviews/amazon/filtered_food_reviews.csv'
    ratios_fp = 'work/reviews/amazon/filtered_ratios.npy'

    reviews_df = pd.read_csv(reviews_df_fp)
    ratios = np.load(ratios_fp)
    review_txts = reviews_df['text'].values
    
    X_train, X_test, y_train, y_test = train_test_split(review_txts, ratios, test_size=0.2, 
                                                        random_state=609)
    
    vectorizer = TfidfVectorizer(max_features=10000)
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)
