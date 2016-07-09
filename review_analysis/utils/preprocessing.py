"""A script for preparing reviews and helpfulness ratios to input into models."""

import sys
from review_analysis.utils.data_io import return_data

if __name__ == '__main__': 
    try: 
        embed_dim = sys.argv[1]
    except: 
        raise Exception("Usage: {} embed_dim".format(sys.argv[0]))
        
    wrd_embedding = return_data("word_embedding", embed_dim=embed_dim)
    reviews, ratios = return_data("reviews")
