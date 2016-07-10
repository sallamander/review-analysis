import numpy as np
from gensim.models.word2vec import Word2Vec
from review_analysis.utils.mappings import create_mapping_dicts

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
        cls.wrd_embedding = Word2Vec(reviews, min_count=1, size=len(cls.vocab))

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
        
        
