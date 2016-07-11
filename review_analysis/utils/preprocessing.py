"""A script for preparing reviews and helpfulness ratios to input into models."""

import sys
import numpy as np
import pickle
from review_analysis.utils.data_io import return_data
from review_analysis.utils.mappings import create_mapping_dicts, \
        gen_embedding_weights

def filter_by_length(reviews, ratios, maxlen):
    """Filter out review/ratio pairs where the review is shorter than the max length.

    Args:
    ----
        reviews: list of list of strings
        ratios: 1d np.ndarray of floats
        maxlen: int

    Output: 
    ------
        filtered_reviews: list of list of strings
        filtered_ratios: 1d np.ndarray of floats 
    """

    filtered_reviews = []
    filtered_ratios = []
    for review, ratio in zip(reviews, ratios): 
        if len(review) < maxlen:
            filtered_reviews.append(review)
            filtered_ratios.append(ratio)

    return filtered_reviews, filtered_ratios

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
        vec_words = [wrd_idx_dct.get(wrd, 0) for wrd in review]
        vec_txts.append(vec_words)

    vec_txts = np.array(vec_txts)
    return vec_txts

def format_reviews(vectorized_reviews, ratios, maxlen=50): 
    """Format the reviews to be `maxlen` long, including the EOS character.
    
    Take `maxlen` minus 1 words from each review, and then tack on the integer 
    1, representing the end-of-sequence (EOS) symbol. 

    Args: 
    ----
        vectorized_reviews: 1d np.ndarray of list of ints
        ratios: 1d np.ndarray of floats 
        maxlen (optional): int

    Return: 
    ------
        formatted_reviews: 2d np.ndarray
            Has shape equal to (`len(vectorized_reviews)`, `maxlen`)
    """
    
    formatted_reviews = []
    filtered_ratios = []
    maxlen -= 1
    for review, ratio in zip(vectorized_reviews, ratios):
        if len(review) >= maxlen: 
            review_subset = review[:maxlen]
            review_subset.append(1)
            formatted_reviews.append(review_subset)
            filtered_ratios.append(ratio)

    formatted_reviews = np.array(formatted_reviews)
    filtered_ratios = np.array(filtered_ratios)
    return formatted_reviews, filtered_ratios

def filter_ratios(ratios, min=0.0, max=1.00):
    """Return a mask to filter the inputted ratios by the given min/max. 

    Args: 
    ----
        ratios: 1d np.ndarray of floats
        min (optional): float
        max (optional): float

    Return: 
    -------
        ratio_mask: 1d np.ndarray of bools
    """

    lower_bound_mask = ratios >= min
    upper_bound_mask = ratios <= max
    ratio_mask = np.logical_and(lower_bound_mask, upper_bound_mask)

    return ratio_mask

if __name__ == '__main__': 
    try: 
        embed_dim = sys.argv[1]
    except: 
        raise Exception("Usage: {} embed_dim".format(sys.argv[0]))
        
    wrd_embedding = return_data("word_embedding", embed_dim=embed_dim)
    reviews, ratios = return_data("reviews")
    # Filter the original reviews/ratios to those that are less than 200 words.
    # Do this here so that the same exact review/ratio pairs can be passed into 
    # the net versus non-net models later in the pipeline.
    reviews, ratios = filter_by_length(reviews, ratios, 200)
    wrd_idx_dct, idx_wrd_dct, wrd_vec_dct = \
            create_mapping_dicts(wrd_embedding, reviews)
    embedding_weights = gen_embedding_weights(wrd_idx_dct, wrd_vec_dct,
                                              int(embed_dim))
    vectorized_reviews = vectorize_txts(reviews, wrd_idx_dct)

    wrd_idx_dct_fp = 'work/wrd_idx_dct.pkl'
    embedding_weights_fp = 'work/embedding_weights.npy'
    vectorized_reviews_fp = 'work/vec_reviews.npy' # For use with net models.
    raw_reviews_fp = 'work/raw_reviews.pkl' # For use with non-net models.
    ratios_fp = 'work/filtered_ratios.npy'

    with open(wrd_idx_dct_fp, 'wb+') as f: 
        pickle.dump(wrd_idx_dct_fp, f)
    with open(raw_reviews_fp, 'wb+') as f: 
        pickle.dump(reviews, f)

    np.save(embedding_weights_fp, embedding_weights)
    np.save(vectorized_reviews_fp, vectorized_reviews)
    np.save(ratios_fp, ratios)

