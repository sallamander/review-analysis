"""A script for evaluating a Keras model results.

I've stuck to evaluating the model that gave the best results, which was the 
network whose weights are being used below.
"""

import sys
sys.setrecursionlimit(10000) # avoid errors from dropout in Keras model
import numpy as np
import pickle
from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib
from keras.layers.recurrent import GRU
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from review_analysis.models.keras_model import KerasSeq2NoSeq
from review_analysis.utils.eval_utils import ConfusionMatrix
from review_analysis.utils.preprocessing import format_reviews


def load_model_data(model_type, transform=False, input_length=201, personal=False):
    """Load the relavent data for predicting with the inputted `model_type`.

    Args: 
    ----
        model_type : str
            expected to be either "net" or "non-net"
        transform (optional) : bool
            perform transformation on `y` for use in a classification problem 
        input_length (optional): int
            determines the length of the input sequences going into an RNN
        personal (optional): bool 
            determines whether to take a manual train/test split, which allows
            for comparison of the confusion matrix obs. back to the raw data frame

    Returns:
    -------
        X_train: 2d np.ndarray
        X_test: 2d np.ndarray
        y_train: 1d np.ndarray
        y_test: 1d np.ndarray
    """

    raw_reviews_fp = 'work/raw_reviews.pkl'
    vectorized_reviews_fp = 'work/vec_reviews.npy'
    ratios_fp = 'work/filtered_ratios.npy'
    ys = np.load(ratios_fp)

    if model_type == "net":
        vectorized_reviews = np.load(vectorized_reviews_fp)
        Xs = format_reviews(vectorized_reviews, maxlen=input_length)
    elif model_type == "non-net":
        with open(raw_reviews_fp, 'rb') as f: 
            raw_reviews = pickle.load(f) # `raw_reviews` are tokenized
        Xs = [' '.join(review) for review in raw_reviews]
    else: 
        raise RuntimeError("Invalid `model_type` inputted!")

    if transform: 
        ys = ys >= 0.50 

    X_train, X_test, y_train, y_test = train_test_split(Xs, ys, test_size=0.2, 
                                                        random_state=609)
    
    if personal: 
        X_train, X_test = Xs[:33470], Xs[33470:]
        y_train, y_test = ys[:33470], ys[33470:]

    if model_type == "non-net":
        max_features = 5000
        vectorizer = TfidfVectorizer(max_features=max_features)
        X_train = vectorizer.fit_transform(X_train)
        X_test = vectorizer.transform(X_test)
    
    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    if len(sys.argv) < 2: 
        msg = "Usage: python eval_models.py model_type"
        raise RuntimeError(msg)
    else: 
        model_type = sys.argv[1]

    if model_type == "net":
        # instantiate a copy of the best fit Keras model and load weights
        weights_fp = 'work/weights/mean_squared_error_GRU_adagrad_512_0.35.h5'
        embedding_weights_fp = 'work/embedding_weights.npy'
        embedding_weights = np.load(embedding_weights_fp)
        model = KerasSeq2NoSeq(input_length=201, cell_type=GRU, encoding_size=512,
                               loss='mean_squared_error', optimizer='adagrad', 
                               dropout=0.35, embedding_weights=embedding_weights) 
        model = model.model # the actual model is stored as an attribute 
        model.load_weights(weights_fp)
    else: 
        model = joblib.load('work/models/random_forest/best_rf.pkl')
    
    X_train, X_test, y_train, y_test = load_model_data(model_type, transform=True,
                                                       personal=True)

    y_pred = model.predict(X_test)
    # transform y pred to line up with y_test (e.g. 0 * 2), which is binary
    y_pred = (y_pred > 0.50)
    
    confusion_mat = ConfusionMatrix()

    if model_type == "net":
        confusion_mat.fit(y_test, y_pred[:, 0])
    else: 
        confusion_mat.fit(y_test, y_pred)

    confusion_cell_counts = confusion_mat.get_cell_counts()
    confusion_cell_obs = confusion_mat.get_cell_obs()

    # specify them here to ensure the order printed
    labels = ('true_positives', 'true_negatives', 'false_positives',
              'false_negatives')
    
    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy: {}'.format(accuracy))
    for label in labels:
        obs = confusion_cell_counts[label]
        label_to_print = ' '.join(label.split('_'))
        print("Number of {}: {}".format(label_to_print, obs))
