import numpy as np
from review_analysis.utils.preprocessing import vectorize_txts

class TestPreprocessing: 

    def test_vectorize_texts(self): 

        wrd_idx_dct = {'UNK':0, 'EOS':1, 'breakfast': 2, 'tacos': 3}
        txts = np.array([['I', 'love', 'breakfast', 'tacos'], 
                         ['breakfast', 'tacos', 'breakfast', '!', '!', '!'], 
                         ['tacos', 'tacos', 'tacos', 'tacos', 'EOS']])

        vec_txts = vectorize_txts(txts, wrd_idx_dct)

        assert (vec_txts[0] == [0, 0, 2, 3])
        assert (vec_txts[1] == [2, 3, 2, 0, 0, 0])
        assert (vec_txts[2] == [3, 3, 3, 3, 1])
