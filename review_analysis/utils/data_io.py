"""A module holding functions to handle data I/O."""

import pickle
import numpy as np
from gensim.models.word2vec import Word2Vec

def return_data(data_type, embed_dim=50): 
    """Return the data specified by the inputted `data_type`.

    This function is built to allow for easier calls for the data from scripts
    external to this one. 

    Args: 
    ----
        data_type: str
        embed_dim (optional): int

    Return: varied
    """

    if data_type == "word_embedding": 
        embedding_fp = 'data/word_embeddings/glove.6B.{}d.txt'.format(embed_dim)
        wrd_embedding = Word2Vec.load_word2vec_format(embedding_fp, binary=False)
        return wrd_embedding
    elif data_type == "reviews": 
        reviews_fp = 'work/reviews/amazon/filtered_tokenized_reviews.pkl'
        ratios_fp = 'work/reviews/amazon/filtered_ratios.txt'

        with open(reviews_fp, 'rb') as f: 
            reviews = pickle.load(f)
        ratios = np.loadtxt(ratios_fp)
        return reviews, ratios 
    else: 
        raise Exception('Invalid data type requested!')
