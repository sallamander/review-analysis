"""A module for creating visuals surrounding word counts, word vectors, etc."""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from review_analysis.utils.preprocessing import filter_ratios
from review_analysis.utils.data_io import return_data

def filter_corpora(corpora, num_top_wrds):
    """Filter each inputted corpus into only `num_top_words` words.

    Args: 
    ----
        corpora: dict
            name (str): corpus (1d np.ndarray of strings) pairs
        num_top_words :int

    Return:
    ------
        filtered_corpora: dict
    """

    vectorizer = CountVectorizer(max_features=num_top_wrds, stop_words='english')

    filtered_corpora = {}
    for name, corpus in corpora.items():
        vectorizer.fit_transform(corpus)
        most_common_wrds = vectorizer.get_feature_names()
        filtered_corpora[name] = most_common_wrds

    return filtered_corpora

def gen_wrd_vec_matrix(filtered_corpora, wrd_embedding):
    """Generate a wrd:embedding matrix for each inputted corpus.

    Args:
    ----
        filtered_corpora: dict
            name (str): filtered corpus (list of words) pairs
        wrd_embedding: gensim.models.word2vec.Word2Vec fitted model

    Return:
    ------
        wrd_embedding_corpora: dict
            name (str): word embedding (2d np.ndarray) pairs
    """

    wrd_embedding_corpora = {}
    embed_dim = wrd_embedding.vector_size
    for name, corpus in filtered_corpora.items():
        embedding_matrix = np.zeros((0, embed_dim))
        for idx, wrd in enumerate(corpus):
            if wrd in wrd_embedding:
                # The `wrd_embedding` vectors are 1d by default 
                wrd_vector = wrd_embedding[wrd][np.newaxis]
                embedding_matrix = np.concatenate([embedding_matrix, 
                                                   wrd_vector])
        wrd_embedding_corpora[name] = embedding_matrix

    return wrd_embedding_corpora

if __name__ == '__main__':
    # Loading the entire df will allow for throwing the raw texts into 
    # sklearns `CountVectorizer`.
    filtered_reviews_df_fp = 'work/reviews/amazon/filtered_food_reviews.csv' 
    filtered_ratios_fp = 'work/reviews/amazon/filtered_ratios.npy' 

    reviews_df = pd.read_csv(filtered_reviews_df_fp)
    ratios = np.load(filtered_ratios_fp)
    wrd_embedding = return_data("word_embedding")

    unhelpful_mask = filter_ratios(ratios, max=0.10)
    middle_mask = filter_ratios(ratios, min=0.10, max=0.90)
    helpful_mask = filter_ratios(ratios, min=0.90)

    unhelpful_reviews = reviews_df.loc[unhelpful_mask, 'text'].values
    middle_reviews = reviews_df.loc[middle_mask, 'text'].values
    helpful_reviews = reviews_df.loc[helpful_mask, 'text'].values
    
    corpora = {'unhelpful': unhelpful_reviews, 'middle': middle_reviews, 
               'helpful': helpful_reviews}
    num_top_wrds = 100 
    filtered_corpora = filter_corpora(corpora, num_top_wrds)
    wrd_embedding_corpora = gen_wrd_vec_matrix(filtered_corpora, wrd_embedding)
