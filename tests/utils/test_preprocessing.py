import numpy as np
from review_analysis.utils.preprocessing import vectorize_txts, format_reviews, \
    filter_ratios, filter_by_length

class TestPreprocessing: 

    def test_vectorize_texts(self): 

        wrd_idx_dct = {'UNK':2, 'EOS':1, 'breakfast': 2, 'tacos': 3}
        txts = np.array([['I', 'love', 'breakfast', 'tacos'], 
                         ['breakfast', 'tacos', 'breakfast', '!', '!', '!'], 
                         ['tacos', 'tacos', 'tacos', 'tacos', 'EOS']])

        vec_txts = vectorize_txts(txts, wrd_idx_dct)

        assert (vec_txts[0] == [2, 2, 2, 3])
        assert (vec_txts[1] == [2, 3, 2, 2, 2, 2])
        assert (vec_txts[2] == [3, 3, 3, 3, 1])

    def test_format_reviews(self): 

        vectorized_reviews = np.array([[0, 0, 2, 3], [2, 3, 2, 0, 0, 0], 
                                       [3, 3, 3, 3, 1]])

        Xs = format_reviews(vectorized_reviews, 8)

        assert (Xs.shape[0] == len(vectorized_reviews))
        assert (Xs.shape[1] == 8)
        # Should all be 0's, since `maxlen` is longer than each of the sequences, 
        # meaning they should be padded with a 0 at the end.
        assert np.all(Xs[:, 0] == 0)

    def test_filter_ratios(self):
        
        ratios = np.arange(0.0, 1.1, 0.1)

        ratios_mask = filter_ratios(ratios)
        filtered_ratios = ratios[ratios_mask]
        assert (filtered_ratios.shape == ratios.shape)

        ratios_mask = filter_ratios(ratios, min=0.5)
        filtered_ratios = ratios[ratios_mask]
        assert (filtered_ratios.shape[0] == 6)

        ratios_mask = filter_ratios(ratios, max=0.5)
        filtered_ratios = ratios[ratios_mask]
        assert (filtered_ratios.shape[0] == 6)

        ratios_mask = filter_ratios(ratios, min=0.2, max=0.5)
        filtered_ratios = ratios[ratios_mask]
        assert (filtered_ratios.shape[0] == 4)

        ratios_mask = filter_ratios(ratios, min=0.5, max=0.5)
        filtered_ratios = ratios[ratios_mask]
        assert (filtered_ratios.shape[0] == 1)

    def test_filter_by_length(self):

        reviews= [['This', 'is', 'a', 'review', '?'], ['Banana'], 
                  ['Frogs', 'are', 'cool'], ['I', 'like', 'turtle']]
        ratios = np.array([[0.25], [0.10], [0.99], [1.0]])

        filtered_reviews, filtered_ratios = filter_by_length(reviews, ratios, 6)
        assert (len(filtered_reviews) == len(reviews))
        assert (len(filtered_ratios) == len(ratios))

        filtered_reviews, filtered_ratios = filter_by_length(reviews, ratios, 4)
        assert (len(filtered_reviews) == 3)
        assert (len(filtered_ratios) == 3)

        filtered_reviews, filtered_ratios = filter_by_length(reviews, ratios, 3)
        assert (len(filtered_reviews) == 1)
        assert (len(filtered_ratios) == 1)

        filtered_reviews, filtered_ratios = filter_by_length(reviews, ratios, 1)
        assert (len(filtered_reviews) == 0)
        assert (len(filtered_ratios) == 0)
