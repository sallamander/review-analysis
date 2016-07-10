"""A module for generating word/idx/vector mappings and moving between them."""

import numpy as np
from collections import defaultdict, Counter
from gensim.corpora.dictionary import Dictionary

def create_mapping_dicts(wrd_embedding, reviews=None, vocab_size=None):
    """Generate word:index, word:vector, index:word dictionaries. 

    Args: 
    ----
        wrd_embedding: gensim.models.word2vec.Word2Vec fitted model
        reviews (optional): np.array (or array-like) of lists of strings
            Used to filter the vocabulary, either to only those words in `reviews`
            or the most common `vocab_size` words in `reviews` that are also in 
            the `wrd_embedding`.
        vocab_size (optional): int
            Keep only `vocab_size` most common words from the reviews. 

    Return: 
    ------
        word_idx_dct: dict
        idx_word_dct: dict
        word_vector_dct: dict
    """

    if reviews is not None: 
        wrd_embedding = _filter_corpus(wrd_embedding, reviews, vocab_size)

    gensim_dct = Dictionary()
    gensim_dct.doc2bow(wrd_embedding.vocab.keys(), allow_update=True)

    # Leave index 0 for unkown words, 1 for the end of sequence character (EOS) 
    wrd_idx_dct = {wrd: (idx + 2) for idx, wrd in gensim_dct.items()}
    idx_wrd_dct = {(idx + 2): wrd for idx, wrd in gensim_dct.items()}
    wrd_idx_dct['EOS'] = 1
    idx_wrd_dct[1] = 'EOS'
    wrd_idx_dct['UNK'] = 0
    idx_wrd_dct[0] = 'UNK'

    wrd_vector_dct = {wrd: wrd_embedding[wrd] for idx, wrd in gensim_dct.items()}
    embedding_dim = wrd_embedding.vector_size
    wrd_vector_dct['EOS'] = np.zeros((embedding_dim))
    wrd_vector_dct['UNK'] = np.zeros((embedding_dim))

    return wrd_idx_dct, idx_wrd_dct, wrd_vector_dct 

def _filter_corpus(wrd_embedding, reviews, vocab_size): 
    """Set the `wrd_embeddding.vocab` to a subset of the vocab from `reviews`. 

    Args: 
    ----
        wrd_embedding: gensim.models.word2vec.Word2Vec fitted model
        reviews: list of lists of strings 
        vocab_size: int or None
            Determines the number of most common words to keep from `reviews`, 
            or all if None.

    Return: 
    ------
        wrd_embedding: gensim.models.word2vec.Word2Vec fitted model
            Original wrd_embedding with `vocab` attribute changed. 
    """

    embedding_vocab = set(wrd_embedding.vocab.keys())
    if vocab_size: 
        # Using a defaultdict is much faster than using Counters, and marginally 
        # faster than using raw dictionaries.
        review_wrd_counter = defaultdict(int)
        for review in reviews: 
            for wrd in review: 
                review_wrd_counter[wrd] += 1 

        master_counter = Counter({k:v for k, v in review_wrd_counter.items() \
                if k in embedding_vocab})
        most_common = master_counter.most_common(vocab_size)
        new_vocab = [wrd_count[0] for wrd_count in most_common]
    else: 
        review_vocab = set(wrd for review in reviews for wrd in review)
        new_vocab = embedding_vocab.intersection(review_vocab)

    new_vocab_dct = {wrd: wrd_embedding.vocab[wrd] for wrd in new_vocab}
    wrd_embedding.vocab = new_vocab_dct

    return wrd_embedding

def gen_embedding_weights(wrd_idx_dct, wrd_vec_dct, embed_dim):
    """Generate the initial embedding weights.

    Args: 
    ----
        wrd_idx_dct: dict
        wrd_vec_dct: dict
        embed_dim: int

    Return: 
    ------
        embedding_weights: 2d np.ndarry
    """

    n_wrds = len(wrd_idx_dct)
    embedding_weights = np.zeros((n_wrds, embed_dim))

    for wrd, idx in wrd_idx_dct.items():
        embedding_weights[idx, :] = wrd_vec_dct[wrd]

    return embedding_weights
