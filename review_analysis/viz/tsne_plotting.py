"""A module for creating visuals surrounding word counts, word vectors, etc.

The `tsne` function below was taken from: 
https://lvdmaaten.github.io/tsne/
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
from nltk import pos_tag
from sklearn.feature_extraction.text import CountVectorizer
from tsne import tsne
from review_analysis.utils.preprocessing import filter_ratios
from review_analysis.utils.data_io import return_data

def filter_corpora(corpora, num_top_wrds, skip=0):
    """Filter each inputted corpus into only `num_top_words` after `skip` words.

    Args: 
    ----
        corpora: list of tuples
            (name (str), corpus (1d np.ndarray of strings) pairs)
        num_top_words :int
        skip (optional): int
            allows for looking at the second, third, fourth `num_top_wrds`

    Return:
    ------
        filtered_corpora: list of tuples
    """

    num_top_wrds += skip 
    vectorizer = CountVectorizer(max_features=num_top_wrds, stop_words='english')

    filtered_corpora = []
    for name, corpus in corpora:
        vectorizer.fit_transform(corpus)
        most_common_wrds = vectorizer.get_feature_names()[skip:]
        filtered_corpora.append((name, most_common_wrds))

    return filtered_corpora

def gen_wrd_vec_matrix(filtered_corpora, wrd_embedding):
    """Generate a wrd:embedding matrix for each inputted corpus.

    Args:
    ----
        filtered_corpora: list of tuples 
            (name (str), filtered corpus (list of words) pairs)
        wrd_embedding: gensim.models.word2vec.Word2Vec fitted model

    Return:
    ------
        wrd_embedding_corpora: list of tuples
    """

    wrd_embedding_corpora = [] 
    embed_dim = wrd_embedding.vector_size
    for name, corpus in filtered_corpora:
        embedding_matrix = np.zeros((0, embed_dim))
        for idx, wrd in enumerate(corpus):
            if wrd in wrd_embedding:
                # The `wrd_embedding` vectors are 1d by default 
                wrd_vector = wrd_embedding[wrd][np.newaxis]
                embedding_matrix = np.concatenate([embedding_matrix, 
                                                   wrd_vector])
        wrd_embedding_corpora.append((name, embedding_matrix))

    return wrd_embedding_corpora

class TSNEEmbedder(object): 
    """A class for running TSNE on inputted corpora. 

    This class runs TSNE simultaneously over all of the inputted corpora. If
    multiple, this ensures that each corpora is embedded in the same vector 
    space.

    Args: 
    ----
        wrd_embedding_corpora: list of tuples
            (name (str), word embeddings (2d np.ndarray) pairs)
    """

    def __init__(self, wrd_embedding_corpora):
        self.wrd_embedding_corpora = wrd_embedding_corpora
        self.embedding_dims = wrd_embedding_corpora[0][1].shape[1]
        self.num_corpus = len(wrd_embedding_corpora)

        self.master_wrd_embedding = self._concat_wrd_embeddings()
        self.master_tsne_embedding = tsne(self.master_wrd_embedding)
        self.tsne_embedding_corpora = self._flatten_tsne_embeddings()

    def _concat_wrd_embeddings(self): 
        """Concat all of the word embeddings in the corpus."""

        master_wrd_embedding = np.zeros((0, self.embedding_dims))
        for _, embedding in self.wrd_embedding_corpora:
            master_wrd_embedding = np.concatenate([master_wrd_embedding, embedding])

        return master_wrd_embedding
    
    def _flatten_tsne_embeddings(self):
        """Flatten `master_tsne_embedding` to map back to the original corpora."""

        tsne_embedding_corpora = {} 
        start_idx = 0
        for name, embedding in self.wrd_embedding_corpora:
            end_idx = start_idx + embedding.shape[0]
            tsne_embedding = self.master_tsne_embedding[start_idx:end_idx]
            tsne_embedding_corpora[name] = tsne_embedding
            start_idx = end_idx

        return tsne_embedding_corpora

class TSNEPlotter(object):
    """A class for building visualizations from TSNE embeddings.

    Args:
    ----
        tsne_embedding_corpora: dict
            name (str): tsne embedding (2d np.ndarray) pairs
        filtered_corpora: list of tuples
            (name (str), filtered corpus (list of words) pairs)
        pos_filters (optional): list 
            parts of speech to plot
        save_dir (optional): str
    """

    def __init__(self, tsne_embedding_corpora, filtered_corpora, title=None,
                 pos_filter=None, save_fp=None):
        self.tsne_embedding_corpora = tsne_embedding_corpora
        self.filtered_corpora = filtered_corpora
        self.title = title
        self.corpora_size = len(filtered_corpora)
        self.pos_filter = {} if not pos_filter else pos_filter
        self.save_fp = save_fp
        self._build_plots()

    def _build_plots(self):
        """Build the plot with the inputted embeddings and words."""

        self._calc_corpora_embedding_bounds()
        self._set_legend()
        self._scale_tsne_embeddings()

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.axis('off')
        for idx, corpus in enumerate(self.filtered_corpora):
            name, wrds = corpus
            tsne_embeddings = self.tsne_embedding_corpora[name]
            color = self.legend_handles[idx].get_facecolor()
            wrds_n_pos = pos_tag(wrds)
            for tsne_embedding, wrd_n_pos in zip(tsne_embeddings, wrds_n_pos):
                if wrd_n_pos[1] in self.pos_filter or not self.pos_filter:
                    plt.text(tsne_embedding[0], tsne_embedding[1], wrd_n_pos[0], 
                             fontdict={'weight': 'bold', 'size': 12}, 
                             color=color)
        plt.legend(loc='best', handles=self.legend_handles)
        if self.title: 
            plt.title(self.title)
        if self.save_fp:
            plt.savefig(self.save_fp)
        
    def _calc_corpora_embedding_bounds(self):
        """Calculate the min/max values for each embedding dimension. 

        These will be used to scale the tsne embeddings so that the words all appear
        tightly in the plot.
        """

        mins, maxes = [], [] 
        for embedding in self.tsne_embedding_corpora.values():
            mins.append(embedding.min(0).tolist())
            maxes.append(embedding.max(0).tolist())

        mins, maxes = np.array(mins), np.array(maxes)
        self.dim_mins, self.dim_maxes = mins.min(0), maxes.max(0)

    def _set_legend(self):
        """Set up `self.legend_handles`."""
        
        self.legend_handles = []
        for idx, corpus in enumerate(self.filtered_corpora, 1):
            name = corpus[0]
            color = plt.cm.Set3(idx / self.corpora_size)
            patch = mpatches.Patch(color=color, label=name)
            self.legend_handles.append(patch)

    def _scale_tsne_embeddings(self): 
        """Scale the tsne embeddings using `self.dim_mins/self.dim_maxes`"""

        for name, embedding in self.tsne_embedding_corpora.items():
            embedding = ((embedding - self.dim_mins) / 
                         (self.dim_maxes - self.dim_mins))
            self.tsne_embedding_corpora[name] = embedding

if __name__ == '__main__':
    if len(sys.argv) < 4: 
        msg = "Usage: python tsne_plotting.py min_ratio max_ratio skip_wrds"
        raise RuntimeError(msg)
    else: 
        min_ratio = float(sys.argv[1])
        max_ratio = float(sys.argv[2])
        skip_wrds = int(sys.argv[3])
    # Loading the entire df will allow for throwing the raw texts into 
    # sklearns `CountVectorizer`.
    filtered_reviews_df_fp = 'work/reviews/amazon/filtered_food_reviews.csv' 
    filtered_ratios_fp = 'work/reviews/amazon/filtered_ratios.npy' 

    reviews_df = pd.read_csv(filtered_reviews_df_fp)
    ratios = np.load(filtered_ratios_fp)
    wrd_embedding = return_data("word_embedding")

    unhelpful_mask = filter_ratios(ratios, max=min_ratio)
    middle_mask = filter_ratios(ratios, min=min_ratio, max=max_ratio)
    helpful_mask = filter_ratios(ratios, min=min_ratio)

    unhelpful_reviews = reviews_df.loc[unhelpful_mask, 'text'].values
    middle_reviews = reviews_df.loc[middle_mask, 'text'].values
    helpful_reviews = reviews_df.loc[helpful_mask, 'text'].values
    
    # Use a list of tuples for the corpora to ensure order, so each corpus
    # gets the same color label each time through the `tsne_plot` below.
    unhelpful_title = 'Unhelpful Reviews (0.0 - {})'.format(min_ratio)
    middle_title = 'Middle Reviews ({} - {})'.format(min_ratio, max_ratio) 
    helpful_title = 'Helpful Reviews ({} - 1.00)'.format(max_ratio)
    corpora = [(unhelpful_title, unhelpful_reviews), 
               (middle_title, middle_reviews), 
               (helpful_title, helpful_reviews)]
    num_top_wrds = 100 
    filtered_corpora = filter_corpora(corpora, num_top_wrds, skip_wrds)
    wrd_embedding_corpora = gen_wrd_vec_matrix(filtered_corpora, wrd_embedding)
    tsne_embedder = TSNEEmbedder(wrd_embedding_corpora)
    tsne_embedding_corpora = tsne_embedder.tsne_embedding_corpora
    
    # generate an appropriate title based on parameters passed in 
    if skip_wrds: 
        min_wrd_idx, max_wrd_idx = skip_wrds, skip_wrds + num_top_wrds
        top_wrds_title = '{} - {}'.format(min_wrd_idx, max_wrd_idx)
    else: 
        top_wrds_title = num_top_wrds
    title = 'Word Embeddings for the Top {} Words In Reviews'.format(top_wrds_title)
    save_fp = 'work/viz/tsne_{}_{}_{}_{}/tsne.png'.format(num_top_wrds, skip_wrds, 
                                                          min_ratio, max_ratio)
    
    # ensure the directory to store the tsne pngs is created
    if not os.path.exists(os.path.dirname(save_fp)):
        os.makedirs(os.path.dirname(save_fp), exist_ok=True)
    tsne_plotter = TSNEPlotter(tsne_embedding_corpora, filtered_corpora,           
                               save_fp=save_fp, title=title)

    title = 'Word Embeddings for the {} in Top {} Words in Reviews'
    pos_tags = {'Adjectives': {'JJ', 'JJR', 'JJS'},
               'Nouns': {'NN', 'NNP', 'NNPS', 'NNS'}, 
               'Adverbs': {'RB', 'RBR', 'RBS'}, 
               'Verbs': {'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'}}
    save_fp = 'work/viz/tsne_{}_{}_{}_{}/tsne_{}.png'
    for descriptor, tags in pos_tags.items():
            tag_title = title.format(descriptor, top_wrds_title)
            tag_save_fp = save_fp.format(num_top_wrds, skip_wrds, min_ratio,
                                         max_ratio, descriptor)
            TSNEPlotter(tsne_embedding_corpora, filtered_corpora, 
                        pos_filter=tags, save_fp=tag_save_fp, title=tag_title)
