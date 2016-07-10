"""A module for generating word/idx/vector mappings and moving between them. """

import numpy as np
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
        wrd_embedding = _filter_corpus(reviews, wrd_embedding, vocab_size)

    gensim_dct = Dictionary()
    gensim_dct.doc2bow(wrd_embedding.vocab.keys(), allow_update=True)

    # Leave index 0 for unkown words, 1 for the end of sequence character (EOS) 
    word_idx_dct = {wrd: (idx + 2) for idx, wrd in gensim_dct.items()}
    idx_word_dct = {(idx + 2): wrd for idx, wrd in gensim_dct.items()}
    word_idx_dct['EOS'] = 1
    idx_word_dct[1] = 'EOS'
    word_idx_dct['UNK'] = 0
    idx_word_dct[0] = 'UNK'

    word_vector_dct = {wrd: wrd_embedding[wrd] for idx, wrd in gensim_dct.items()}
    embedding_dim = wrd_embedding.vector_size
    word_vector_dct['EOS'] = np.zeros((embedding_dim))
    word_vector_dct['UNK'] = np.zeros((embedding_dim))

    return word_idx_dct, idx_word_dct, word_vector_dct 

def _filter_corpus(reviews, wrd_embedding, vocab_size): 
    """Set the `wrd_embeddding.vocab` to a subset of the vocab from `reviews`. 

    Args: 
    ----
        reviews: list of lists of strings 
        wrd_embedding: gensim.models.word2vec.Word2Vec fitted model
        vocab_size: int or None
            Determines the number of most common words to keep from `reviews`, 
            or all if None.
            

    Return: 
    ------
        wrd_embedding: gensim.models.word2vec.Word2Vec fitted model
            Original wrd_embedding with `vocab` attribute changed. 
    """

    return wrd_embedding

