import numpy as np
from review_analysis.utils.preprocessing import vectorize_txts, format_reviews

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

    def test_format_reviews(self): 

        vectorized_reviews = np.array([[0, 0, 2, 3], [2, 3, 2, 0, 0, 0], 
                                       [3, 3, 3, 3, 1]])
        ratios = np.array([[1.0], [0.75], [0.25]])

        Xs, ys = format_reviews(vectorized_reviews, ratios, 1)

        assert (Xs.shape[0] == len(vectorized_reviews))
        assert (Xs.shape[1] == 1)
        assert (ys.shape[0] == len(vectorized_reviews))
        # Should all be 1's, since `maxlen` is one, meaning the only thing 
        # outputted is the integer corresponding to end-of-sequence (EOS), 1.
        assert np.all(Xs == 1)

        Xs, ys = format_reviews(vectorized_reviews, ratios, 3)

        assert (Xs.shape[0] == len(vectorized_reviews))
        assert (Xs.shape[1] == 3)
        assert (ys.shape[0] == len(vectorized_reviews))

        Xs, ys = format_reviews(vectorized_reviews, ratios, 6)

        # One review is dropped because it is not minimum 5 characters long.
        assert (Xs.shape[0] == 2)
        assert (Xs.shape[1] == 6)
        assert (ys.shape[0] == 2)

