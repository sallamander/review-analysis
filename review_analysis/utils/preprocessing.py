"""A script for preparing reviews and helpfulness ratios to input into models."""

import sys
from review_analysis.utils.data_io import return_data
from review_analysis.utils.mappings import create_mapping_dicts, \
        gen_embedding_weights

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
