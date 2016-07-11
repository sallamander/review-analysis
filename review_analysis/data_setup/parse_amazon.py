"""A script for parsing Amazon Fine Foods data reviews. 

Data can be found at http://snap.stanford.edu/data/finefoods.txt.gz. 
"""

import re
import sys
import pickle
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize

def parse_food_reviews(input_fp):
    """Parse Amazon food reviews from `input_fp`. 

    This code is pulled from the following link, with minor modifications made: 
    https://github.com/benhamner/snap-amazon-fine-foods/blob/master/src/process.py

    Args:
    ----
        input_fp: str

    Return: 
    ------
        reviews_df: pandas.DataFrame
    """

    reviews = []
    
    with open(input_fp, encoding='latin-1') as f: 
        line = f.readline()
        while line != "":
            product_id = re.findall("^product/productId: (.*)",  line)[0].strip()
            user_id = re.findall("^review/userId: (.*)", f.readline())[0].strip()
            username = re.findall("^review/profileName: (.*)", 
                    f.readline())[0].strip()
            # Handle weird edge cases.  
            line = f.readline()
            if line[:6] != "review":
                line = f.readline()
            helpfulness = re.findall("^review/helpfulness: (.*)", line)[0].strip()
            score = re.findall("^review/score: (.*)", f.readline())[0].strip()
            time = re.findall("^review/time: (.*)", f.readline())[0].strip()
            summary = re.findall("^review/summary: (.*)", f.readline())[0].strip()
            text = re.findall("^review/text: (.*)", f.readline())[0].strip()

            helpfulness_numerator, helpfulness_denominator = helpfulness.split("/")
            helpfulness_numerator = int(helpfulness_numerator)
            helpfulness_denominator = int(helpfulness_denominator)
            score = int(float(score))
            time = int(time)
            # Skip over the newline that separates each review.  
            line = f.readline()
            line = f.readline()

            review = [product_id, user_id, username, helpfulness_numerator, \
                      helpfulness_denominator, score, time, summary, text]
            reviews.append(review)

    columns = ["product_id", "user_id", "username", "helpfulness_numerator", \
               "helpfulness_denominator", "score", "time", "summary", "text"]
    reviews_df = pd.DataFrame(reviews, columns=columns)

    return reviews_df

if __name__ == '__main__':
    if len(sys.argv) < 3: 
        raise RuntimeError('Usage: python parse_amazon.py input_fp output_dir')
    else: 
        input_fp = sys.argv[1]
        output_dir = sys.argv[2]

    food_reviews_fp1 = output_dir + 'raw_food_reviews.csv'
    tokenized_reviews_fp1 = output_dir + 'raw_tokenized_reviews.pkl'
    ratios_fp1 = output_dir + 'raw_ratios.npy'
    food_reviews_fp2 = output_dir + 'filtered_food_reviews.csv'
    tokenized_reviews_fp2 = output_dir + 'filtered_tokenized_reviews.pkl'
    ratios_fp2 = output_dir + 'filtered_ratios.npy'

    reviews_df = parse_food_reviews(input_fp) # Total 568454 obs. 
    reviews_df['helpfulness_ratio'] = (reviews_df['helpfulness_numerator'] /      
                                       reviews_df['helpfulness_denominator'])
    
    reviews = reviews_df['text'].values
    ratios = reviews_df['helpfulness_ratio'].values
    tokenized_reviews = [word_tokenize(review.lower()) for review in reviews]
    tokenized_reviews = np.array(tokenized_reviews)

    nans_mask = np.isnan(reviews_df['helpfulness_ratio']) # Captures 270052 obs.
    bad_data_mask = reviews_df['helpfulness_ratio'] > 1.0 # Captures 2 obs. 
    num_votes_mask = reviews_df['helpfulness_denominator'] > 5 # Captures 52643 obs.
    
    temp_mask = np.logical_and(~nans_mask, ~bad_data_mask)
    final_mask = np.logical_and(temp_mask, num_votes_mask).values
    
    filtered_reviews_df = reviews_df[final_mask]
    filtered_tokenized_reviews = np.array(tokenized_reviews)[final_mask]
    filtered_ratios = ratios[final_mask]

    reviews_df.to_csv(food_reviews_fp1)
    filtered_reviews_df.to_csv(food_reviews_fp2)

    np.save(ratios_fp1, ratios)
    np.save(ratios_fp2, filtered_ratios)

    with open(tokenized_reviews_fp1, 'wb+') as f: 
        pickle.dump(tokenized_reviews, f)
    with open(tokenized_reviews_fp2, 'wb+') as f: 
        pickle.dump(filtered_tokenized_reviews, f)
