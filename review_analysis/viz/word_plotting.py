"""A module for creating visuals surrounding word counts, word vectors, etc."""

import numpy as np
import pandas as pd
from review_analysis.utils.preprocessing import filter_ratios

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

    
