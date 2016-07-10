"""A script for preparing reviews and helpfulness ratios to input into models."""

import sys
import numpy as np
import pickle
from review_analysis.utils.data_io import return_data
from review_analysis.utils.mappings import create_mapping_dicts, \
        gen_embedding_weights

def vectorize_txts(txts, wrd_idx_dct): 
    """Translate the reviews in `txts` into numbers using `word_idx_dct`.

    Args:
    ----
        txts: 1d np.ndarray (or array-like) of lists of strings
        word_idx_dct: dict

    Return:
    ------
        vec_txts: 1d np.ndarray
    """
    
    vec_txts = []
    for review in txts: 
        vec_words = [wrd_idx_dct.get(word, 0) for word in review]
        vec_txts.append(vec_words)

    vec_txts = np.array(vec_txts)
    return vec_txts

if __name__ == '__main__': 
    try: 
        embed_dim = sys.argv[1]
    except: 
        raise Exception("Usage: {} embed_dim".format(sys.argv[0]))
        
    wrd_embedding = return_data("word_embedding", embed_dim=embed_dim)
    reviews, ratios = return_data("reviews")
    reviews, ratios = reviews[:10], ratios[:10]
    wrd_idx_dct, idx_wrd_dct, wrd_vec_dct = \
            create_mapping_dicts(wrd_embedding, reviews)
    embedding_weights = gen_embedding_weights(wrd_idx_dct, wrd_vec_dct,
                                              int(embed_dim))
    vectorized_reviews = vectorize_txts(reviews, wrd_idx_dct)

    word_idx_dct_fp = 'work/word_idx_dct.pkl'
    embedding_weights_fp = 'work/embedding_weights.npy'
    vectorized_reviews_fp = 'work/vec_reviews.npy'

    with open(word_idx_dct_fp, 'wb+') as f: 
        pickle.dump(word_idx_dct_fp, f)

    np.save(embedding_weights_fp, embedding_weights)
    np.save(vectorized_reviews_fp, vectorized_reviews)
