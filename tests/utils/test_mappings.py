import numpy as np
from gensim.models.word2vec import Word2Vec
from review_analysis.utils.mappings import create_mapping_dicts, \
        _filter_corpus, gen_embedding_weights

class TestMappings: 

    def setup_class(cls): 
        reviews = [['This', 'is', 'a', 'helpful', 'review', '.'], 
                   ['This', 'is', 'an' , 'unhelpful', 'review', '.'],
                   ['Frogs', 'snakes', 'dogs', 'cats', 'horses', '?'], 
                   ['100', 'dollars', '...', 'day', 'off', '?', 'Jesse']]
        ratios = [1.0, 0.0, 0.25, 0.75]

        cls.reviews = np.array(reviews)
        cls.ratios = np.array(ratios)
        cls.vocab = set(word for review in reviews for word in review)
        cls.wrd_embedding = Word2Vec(cls.reviews, min_count=1, 
                                     size=len(cls.vocab))

    def teardown_class(cls): 
        del cls.reviews
        del cls.ratios
        del cls.vocab
        del cls.wrd_embedding

    def test_create_mapping_dicts(self): 
        
        wrd_idx_dct, idx_wrd_dct, wrd_vec_dct = \
                create_mapping_dicts(self.wrd_embedding)

        # +2 because of the additional 'EOS' and 'UNK' characters added
        assert (len(wrd_idx_dct) == (len(self.wrd_embedding.vocab) + 2))
        assert (len(idx_wrd_dct) == (len(self.wrd_embedding.vocab) + 2))

        wrd_vec = wrd_vec_dct['This']
        assert (len(wrd_vec_dct) == (len(self.wrd_embedding.vocab) + 2))
        assert (wrd_vec.shape[0] == (self.wrd_embedding.vector_size))

    def test_filter_corpus(self): 
        
        wrd_embedding = _filter_corpus(self.wrd_embedding, self.reviews,
                                       vocab_size=None)
        assert (len(wrd_embedding.vocab) == len(self.vocab))

        # This is descending because the alterations to the embedding are in place. 
        for filter_size in range(10, 5, -1):
            wrd_embedding = _filter_corpus(self.wrd_embedding, self.reviews, 
                                           vocab_size=filter_size)
            assert (len(wrd_embedding.vocab) == filter_size)
            assert (len(self.wrd_embedding.vocab) == filter_size)

    def test_gen_embedding_weights(self):
        
        embed_dim = 2
        wrd_idx_dct = {'UNK':0, 'EOS':1, 'LSTM':2, 'GRU':3}

        wrd_vec_dct = {}
        for wrd in wrd_idx_dct: 
            wrd_vec_dct[wrd] = np.zeros(embed_dim)

        embedding_weights = gen_embedding_weights(wrd_idx_dct, wrd_vec_dct, embed_dim)
        
        assert (type(embedding_weights) == np.ndarray)
        assert (embedding_weights.shape[0] == len(wrd_idx_dct))
        assert (embedding_weights.shape[1] == embed_dim)
