import pytest
import numpy as np
from gensim.models.word2vec import Word2Vec
from review_analysis.viz.tsne_plotting import filter_corpora, \
        gen_wrd_vec_matrix, gen_tsne_embedding, TSNEPlotter 

class TestWordPlotting: 

    def setup_class(cls):

        cls.embedding_corpora = {'Test1': np.zeros((2, 10)), 
                                 'Test2': np.ones((5, 10)), 
                                 'Test3': np.random.random((10, 10))}
        cls.filtered_corpora = [('Test1', ['This' ,'is', 'a', 'list']), 
                                ('Test2', ['Hi', 'Matthew', '??']), 
                                ('Test3', ['Cant', 'forget', 'Felix', '!'])]

    def teardown_class(cls):
       del cls.embedding_corpora, cls.filtered_corpora

    def test_calc_corpora_embedding_bounds(self):

        tsne_plotter = TSNEPlotter(self.embedding_corpora, self.filtered_corpora)

        assert np.all(tsne_plotter.dim_mins == 0)
        assert np.all(tsne_plotter.dim_maxes == 1)

    def test_scale_tsne_embeddings(self): 

        tsne_plotter = TSNEPlotter(self.embedding_corpora, self.filtered_corpora)
        for embedding in tsne_plotter.tsne_embedding_corpora.values():
            assert (embedding.min() >= 0)
            assert (embedding.max() <= 1)

    def test_filter_corpora(self):
        corpus1 = np.array(['This is the first corpus...', 
                             'There are only two documents.', 
                             'Pysche!'])
        corpus2 = np.array(['Now there is a second corpus!'])
        corpora = [('corpus1', corpus1), ('corpus2', corpus2)]
        
        num_top_wrds = 2
        filtered_corpora = filter_corpora(corpora, num_top_wrds)

        # <= because stopwords are removed, which may not leave enough words
        # in the text examples to hit `num_top_wrds`
        assert (len(filtered_corpora) == len(corpora))
        assert (len(filtered_corpora[0][1]) <= num_top_wrds)
        assert (len(filtered_corpora[1][1]) <= num_top_wrds)

        num_top_wrds = 5
        filtered_corpora = filter_corpora(corpora, num_top_wrds)

        assert (len(filtered_corpora) == len(corpora))
        assert (len(filtered_corpora[0][1]) <= num_top_wrds)
        assert (len(filtered_corpora[1][1])<= num_top_wrds)

    def test_gen_wrd_vec_matrix(self):
        
        filtered_corpus1 = ['beer', 'for', 'Jesse', 'if', 'he', 'finds', 'this']
        filtered_corpus2 = ['two', 'for', 'Dan']
        sentences = [filtered_corpus1, filtered_corpus2]
        vocab = set(word for sentence in sentences for word in sentence)
 
        filtered_corpora = [('corpus1' , filtered_corpus1), 
                            ('corpus2' , filtered_corpus2)]
        wrd_embedding = Word2Vec(sentences, min_count=1, size=len(vocab))
        embed_dim = wrd_embedding.vector_size
        wrd_matrix_corpora = gen_wrd_vec_matrix(filtered_corpora, wrd_embedding)

        assert (len(wrd_matrix_corpora) == len(filtered_corpora))
        assert (wrd_matrix_corpora[0][1].shape == 
                (len(filtered_corpus1), embed_dim))

    def test_gen_tsne_embedding(self):

        wrd_embedding_corpus1 = np.zeros((10, 50))
        wrd_embedding_corpus2 = np.zeros((20, 50))
        wrd_embedding_corpora = {'corpus1': wrd_embedding_corpus1, 
                                 'corpus2': wrd_embedding_corpus2}
        tsne_embedding_corpora = gen_tsne_embedding(wrd_embedding_corpora)

        assert (len(tsne_embedding_corpora) == len(wrd_embedding_corpora))
        assert (tsne_embedding_corpora['corpus1'].shape == ((10, 2)))
        assert (tsne_embedding_corpora['corpus2'].shape == ((20, 2)))
