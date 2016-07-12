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
        optimizer: str
        dropout (optional): float
        embedding_weights (optional): 2d np.ndarray
    """

    def __init__(self, input_length, cell_type, encoding_size, loss, optimizer,   
                 dropout=0.0, embedding_weights=None):
        
        self.input_length = input_length
        self.cell_type = cell_type
        self.encoding_size = encoding_size
        self.loss = loss
        self.metrics = []
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

        if self.loss == 'binary_crossentropy': 
            layer = Dense(2, activation='softmax')(layer)
            self.metrics.append('accuracy')
        elif self.loss == 'mean_squared_error': 
            layer = Dense(1, activation='linear')(layer)
        else: 
            msg = 'Expects loss to be `binary_crossentroy` or `mean_squared_error`!'
            raise RuntimeError(msg)

        model = Model(input=reviews, output=layer)
        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)

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
            logger = LossSaver(logging_fp, self.metrics)
            callbacks.append(logger)

        self.model.fit(X_train, y_train, nb_epoch=nb_epoch, batch_size=batch_size,
                       callbacks=callbacks, validation_data=validation_data, 
                       shuffle=False)

if __name__ == '__main__':
    if len(sys.argv) < 4: 
        msg = "Usage: python model.py metric cell_type optimizer encoding_size dropout_pct"
        raise RuntimeError(msg)
    else: 
        metric = sys.argv[1]
        cell_type = sys.argv[2]
        optimizer = sys.argv[3]
        encoding_size = int(sys.argv[4])
        if len(sys.argv) >= 6: 
            dropout = float(sys.argv[5])
        else: 
            dropout = 0

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

    if metric == "binary_crossentropy": 
        new_ys = np.zeros((len(ys), 2))
        new_ys[:, 1] = ys > 0.50
        ys = new_ys

    X_train, X_test, y_train, y_test = train_test_split(Xs, ys, test_size=0.2, 
                                                        random_state=609)
    
    train_mean = y_train.mean()
    train_guessing_error = intellegently_guess(y_train, train_mean)
    test_guessing_error = intellegently_guess(y_test, train_mean)
    print('Train needs to beat... {}'.format(train_guessing_error))
    print('Test needs to beat... {}'.format(test_guessing_error))
    logging_fp = 'work/{}/{}/{}_{}_{}_'.format(metric, cell_type, optimizer, 
                                               encoding_size, dropout)
    keras_model = KerasSeq2NoSeq(input_length, cell_type=cell, 
                                 encoding_size=encoding_size, 
                                 loss=metric, optimizer=optimizer,
                                 dropout=dropout, embedding_weights=embedding_weights)
    keras_model.fit(X_train, y_train, batch_size=32, nb_epoch=20, logging=True, 
                    logging_fp=logging_fp, validation_data=(X_test, y_test))
