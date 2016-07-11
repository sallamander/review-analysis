"""A module for creating visuals surrounding word counts, word vectors, etc."""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from review_analysis.utils.preprocessing import filter_ratios

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

if __name__ == '__main__':
    # Loading the entire df will allow for throwing the raw texts into 
    # sklearns `CountVectorizer`.
    filtered_reviews_df_fp = 'work/reviews/amazon/filtered_food_reviews.csv' 
    filtered_ratios_fp = 'work/reviews/amazon/filtered_ratios.npy' 

    reviews_df = pd.read_csv(filtered_reviews_df_fp)
    ratios = np.load(filtered_ratios_fp)

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
