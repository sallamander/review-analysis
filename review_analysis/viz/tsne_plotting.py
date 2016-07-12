"""A module for creating visuals surrounding word counts, word vectors, etc."""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tsne import tsne
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

def gen_tsne_embedding(wrd_embedding_corpora):
    """Run TSNE for each inputted corpus. 

    Args:
    ----
        wrd_matrix_corpora: dict
            name (str): word matrix (2d np.ndarray) pairs

    Return:
    ------
        tsne_embedding_corpora:dict
            name (str): tsne embedding (2d np.ndarray) pairs
    """

    tsne_embedding_corpora = {}
    for name, wrd_embedding in wrd_embedding_corpora.items():
        tsne_embedding = tsne(wrd_embedding)
        tsne_embedding_corpora[name] = tsne_embedding

    return tsne_embedding_corpora

def plot_tsne(tsne_embedding_corpora, filtered_corpora, save_fp=None, title=None):
    """Plot the tsne embeddings of the vocab in `filtered_corpora`.

    Args:
    ----
        tsne_embedding_corpora:dict
            name (str): tsne embedding (2d np.ndarray) pairs
        filtered_corpora: dict
            name (str): filtered corpus (list of words) pairs
        save_fp (optional): str
        title (optional): str
    """

    dim_mins, dim_maxes = calc_corpora_embedding_bounds(tsne_embedding_corpora)

    plt.figure(figsize=(10, 6))
    ax = plt.subplot(111)
    ax.axis('off')

    num = 0
    corpora_size = len(filtered_corpora)
    legend_handles = []
    for name, corpus in filtered_corpora.items():
        num += 1
        tsne_embeddings = tsne_embedding_corpora[name]
        tsne_embeddings = (tsne_embeddings - dim_mins) / (dim_maxes - dim_mins)
        color = plt.cm.Set3(num / corpora_size)
        patch = mpatches.Patch(color=color, label=name)
        legend_handles.append(patch)

        for tsne_embedding, wrd in zip(tsne_embeddings, corpus):
            plt.text(tsne_embedding[0], tsne_embedding[1], wrd, 
                     color=color, fontdict={'weight': 'bold', 'size': 12})
    
    plt.legend(loc='best', handles=legend_handles)
    if title: 
        plt.title(title)
    if save_fp:
        plt.savefig(save_fp)
    else: 
        plt.show()

def calc_corpora_embedding_bounds(tsne_embedding_corpora):
    """Calculate the min/max values for each embedding dimension. 

    These will be used to scale the tsne embeddings so that the words all appear
    tightly in the plot.

    Args:
    ----
        tsne_embedding_corpora:dict
            name (str): tsne embedding (2d np.ndarray) pairs

    Return:
    ------
        dim_mins: 1d np.ndarray
        dim_maxes: 1d np.ndarray
    """

    mins, maxes = [], [] 
    for name, embedding in tsne_embedding_corpora.items():
        mins.append(embedding.min(0).tolist())
        maxes.append(embedding.max(0).tolist())

    mins = np.array(mins)
    maxes = np.array(maxes)
    dim_mins = mins.min(0)
    dim_maxes = maxes.max(0)

    return dim_mins, dim_maxes

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
    
    corpora = {'Unhelpful Reviews (0.0 - 0.10)': unhelpful_reviews, 
               'Middle Reviews (0.10 - 0.90)': middle_reviews, 
               'Helpful Reviews (0.90 - 1.00)': helpful_reviews}
    num_top_wrds = 100

    filtered_corpora = filter_corpora(corpora, num_top_wrds)
    wrd_embedding_corpora = gen_wrd_vec_matrix(filtered_corpora, wrd_embedding)
    tsne_embedding_corpora = gen_tsne_embedding(wrd_embedding_corpora)
    
    title = 'Word Embeddings for the Top {} Words In Reviews'.format(num_top_wrds)
    save_fp = 'work/tsne.png'
    plot_tsne(tsne_embedding_corpora, filtered_corpora, save_fp, title)
