"""A script for fitting recurrent networks for predicting on amazon reviews."""

import pickle
import numpy as np
from review_analysis.utils.preprocessing import format_reviews

if __name__ == '__main__':
    wrd_idx_dct_fp = 'work/wrd_idx_dct.pkl'
    embedding_weights_fp = 'work/embedding_weights.npy'
    vectorized_reviews_fp = 'work/vec_reviews.npy'
    ratios_fp = 'work/reviews/amazon/filtered_ratios.npy'

    with open(wrd_idx_dct_fp, 'rb') as f: 
        wrd_idx_dct = pickle.load(f)
    embedding_weights = np.load(embedding_weights_fp)
    vectorized_reviews = np.load(vectorized_reviews_fp)
    ratios = np.load(ratios_fp)

    input_length = 50
    Xs = format_reviews(vectorized_reviews, maxlen=input_length)
    ys = np.array(ratios)
