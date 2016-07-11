"""A script for fitting recurrent networks for predicting on amazon reviews."""

import pickle
import numpy as np
import sys
np.random.seed(609)
from sklearn.cross_validation import train_test_split
from keras.layers import Input
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import GRU, LSTM
from keras.layers.core import Dense
from keras.models import Model
from keras.callbacks import EarlyStopping
from review_analysis.utils.preprocessing import format_reviews
from review_analysis.utils.keras_callbacks import LossSaver
from review_analysis.utils.eval_utils import intellegently_guess

class KerasSeq2NoSeq(object):
    """Recurrent net for regression/classification based on an input sequence.

    Args: 
    ----
        input_length: int
            length of each input sequence
        cell_type: keras.layers.recurrent cell (LSTM or GRU)
        encoding_size: int
            number of units in the encoding layer
        loss: str
        output_activation: str
            expected to be either `sigmoid` or `linear`
        optimizer: str
        dropout (optional): float
        embedding_weights (optional): 2d np.ndarray
    """

    def __init__(self, input_length, cell_type, encoding_size, loss,
            output_activation, optimizer, dropout=0.0, embedding_weights=None):
        
        self.input_length = input_length
        self.cell_type = cell_type
        self.encoding_size = encoding_size
        self.loss = loss
        self.output_activation = output_activation
        self.optimizer = optimizer
        self.dropout = dropout
        self.embedding_weights = embedding_weights

        self.model = self._build_model()

    def _build_model(self): 
        """Build the model according to specifications passed into the __init__"""

        reviews = Input(shape=(input_length,), dtype='int32')

        # If embedding weights are passed in, run the input sequence through an
        # embedding layer, and otherwise straight into the recurrent encoder cell. 
        if embedding_weights is not None:
            vocab_size = embedding_weights.shape[0] 
            embed_dim = embedding_weights.shape[1] 
            embeddings = Embedding(input_dim=vocab_size, output_dim=embed_dim, 
                                   weights=[embedding_weights], 
                                   dropout=self.dropout, mask_zero=True)(reviews)
            inputs = embeddings
        else: 
            inputs = reviews
            
        layer = self.cell_type(self.encoding_size, dropout_W=self.dropout, 
                               dropout_U=self.dropout)(inputs)

        # If `self.output_activation` is `sigmoid`, treat the problem as 
        # classification, and otherwise a regression problem. 
        if self.output_activation=='sigmoid': 
            layer = Dense(2, activation=self.output_activation)(layer)
        elif self.output_activation == 'linear': 
            layer = Dense(1, activation=self.output_activation)(layer)
        else: 
            message = 'KerasSeq2NoSeq being used outside of its intended usage'
            raise RuntimeError(message)

        model = Model(input=reviews, output=layer)
        model.compile(loss=self.loss, optimizer=self.optimizer)

        return model
    
    def fit(self, X_train, y_train, batch_size=32, nb_epoch=10, 
            early_stopping_tol=0, logging=False, logging_fp='work/',       
            validation_data=None):
        """Fit the model. 

        Args:
        ----
            X_train: 2d np.ndarray 
                must have `shape[1]` equal to `self.input_length`
            y_train: 1d np.ndarray
            batch_size (optional): int
            nb_epoch (optional): int
            early_stoppint_tol (optional): int
            logging (optional): bool
                save batch loss throughout training
            logging_fp (optional): str
            validation_data (optional): tuple  
        """
        
        callbacks=[]
        if early_stopping_tol: 
            monitor = 'loss' if not validation_data else 'val_loss'
            early_stopping = EarlyStopping(monitor=monitor, 
                                           patience=early_stopping_tol)
            callbacks.append(early_stopping)
        if logging: 
            logger = LossSaver(logging_fp)
            callbacks.append(logger)

        self.model.fit(X_train, y_train, nb_epoch=nb_epoch, batch_size=batch_size,
                       callbacks=callbacks, validation_data=validation_data)

if __name__ == '__main__':
    if len(sys.argv) < 4: 
        error_message = "Usage: python model.py cell_type optimizer encoding_size" 
        raise RuntimeError(error_message)
    else: 
        cell_type = sys.argv[1]
        optimizer = sys.argv[2]
        encoding_size = int(sys.argv[3])

        if cell_type == 'GRU':
            cell = GRU
        else: 
            cell = LSTM

    embedding_weights_fp = 'work/embedding_weights.npy'
    vectorized_reviews_fp = 'work/vec_reviews.npy'
    ratios_fp = 'work/filtered_ratios.npy'

    embedding_weights = np.load(embedding_weights_fp)
    vectorized_reviews = np.load(vectorized_reviews_fp)
    ratios = np.load(ratios_fp)

    input_length = 201 
    Xs = format_reviews(vectorized_reviews, maxlen=input_length)
    ys = ratios
    
    X_train, X_test, y_train, y_test = train_test_split(Xs, ys, test_size=0.2, 
                                                        random_state=609)

    print('Need to beat... {}'.format(intellegently_guess(y_train)))
    logging_fp = 'work/mean_squared_error/{}/{}_{}_'.format(cell_type, optimizer, 
                                                            encoding_size)
    keras_model = KerasSeq2NoSeq(input_length, cell_type=cell, 
                                 encoding_size=encoding_size, 
                                 loss='mean_squared_error', 
                                 output_activation='linear', optimizer=optimizer, 
                                 embedding_weights=embedding_weights)
    keras_model.fit(X_train, y_train, batch_size=32, nb_epoch=15, logging=True, 
                    logging_fp=logging_fp, validation_data=(X_test, y_test))