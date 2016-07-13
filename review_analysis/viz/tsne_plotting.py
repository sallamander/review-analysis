"""A module for creating visuals surrounding word counts, word vectors, etc."""

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

def filter_corpora(corpora, num_top_wrds):
    """Filter each inputted corpus into only `num_top_words` words.

    Args: 
    ----
        corpora: list of tuples
            (name (str), corpus (1d np.ndarray of strings) pairs)
        num_top_words :int

    Return:
    ------
        filtered_corpora: list of tuples
    """

    vectorizer = CountVectorizer(max_features=num_top_wrds, stop_words='english')

    filtered_corpora = []
    for name, corpus in corpora:
        vectorizer.fit_transform(corpus)
        most_common_wrds = vectorizer.get_feature_names()
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
    for name, wrd_embedding in wrd_embedding_corpora:
        tsne_embedding = tsne(wrd_embedding)
        tsne_embedding_corpora[name] = tsne_embedding

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
            parts of speech to also plot
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

        mins = np.array(mins)
        maxes = np.array(maxes)
        self.dim_mins = mins.min(0)
        self.dim_maxes = maxes.max(0)

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
    
    # Use a list of tuples instead of a dict. to ensure order, so each corpus
    # gets the same color label each time through the `tsne_plot` below.
    corpora = [('Unhelpful Reviews (0.0 - 0.10)', unhelpful_reviews), 
               ('Middle Reviews (0.10 - 0.90)', middle_reviews), 
               ('Helpful Reviews (0.90 - 1.00)', helpful_reviews)]
    num_top_wrds = 100
    filtered_corpora = filter_corpora(corpora, num_top_wrds)
    wrd_embedding_corpora = gen_wrd_vec_matrix(filtered_corpora, wrd_embedding)
    tsne_embedding_corpora = gen_tsne_embedding(wrd_embedding_corpora)
    
    title = 'Word Embeddings for the Top {} Words In Reviews'.format(num_top_wrds)
    save_fp = 'work/viz/tsne_{}.png'.format(num_top_wrds)
    tsne_plotter = TSNEPlotter(tsne_embedding_corpora, filtered_corpora,           
                               save_fp=save_fp, title=title)

    title = 'Word Embeddings for the {} in Top {} Words in Reviews'
    pos_tags = {'Adjectives': {'JJ', 'JJR', 'JJS'},
               'Nouns': {'NN', 'NNP', 'NNPS', 'NNS'}, 
               'Adverbs': {'RB', 'RBR', 'RBS'}, 
               'Verbs': {'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'}}
    save_fp = 'work/viz/tsne_{}_{}.png'
    for descriptor, tags in pos_tags.items():
            tag_title = title.format(descriptor, num_top_wrds)
            tag_save_fp = save_fp.format(descriptor, num_top_wrds)
            TSNEPlotter(tsne_embedding_corpora, filtered_corpora, pos_filter=tags, 
                      save_fp=tag_save_fp, title=tag_title)
