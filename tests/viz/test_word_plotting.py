import pytest
import numpy as np
from gensim.models.word2vec import Word2Vec
from review_analysis.viz.word_plotting import filter_corpora

class TestWordPlotting: 

    def test_filter_corpora(self):
        corpus1 = np.array(['This is the first corpus...', 
                             'There are only two documents.', 
                             'Pysche!'])
        corpus2 = np.array(['Now there is a second corpus!'])
        corpora = {'corpus1': corpus1, 'corpus2': corpus2}
        
        num_top_wrds = 2
        filtered_corpora = filter_corpora(corpora, num_top_wrds)

        # <= because stopwords are removed, which may not leave enough words
        # in the text examples to hit `num_top_wrds`
        assert (len(filtered_corpora) == len(corpora))
        assert (len(filtered_corpora['corpus1']) <= num_top_wrds)
        assert (len(filtered_corpora['corpus2']) <= num_top_wrds)

        num_top_wrds = 5
        filtered_corpora = filter_corpora(corpora, num_top_wrds)

        assert (len(filtered_corpora) == len(corpora))
        assert (len(filtered_corpora['corpus1']) <= num_top_wrds)
        assert (len(filtered_corpora['corpus2']) <= num_top_wrds)
