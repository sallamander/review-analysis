"""A script for fitting non neural network models on text.

These models will simply fit on a term-frequency, inverse-document frequency 
matrix generated from the corpus.
"""

import sys
import os
import pickle
import pandas as pd
import numpy as np
import multiprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import ElasticNet, SGDClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from review_analysis.utils.eval_utils import intellegently_guess

def get_grid_params(model_name):
    """Return the appropriate model parameters to search over. 

    Args:
    ----
        model_name: str

    Return: 
    ------
        param_dct: dict
    """

    if model_name == 'logistic': 
        param_dct = {'penalty': ['elasticnet'], 'alpha': [0.00001, 0.0001, 0.001],
                     'l1_ratio': [0.10, 0.15, 0.20]}
    elif model_name == 'linear':
        param_dct = {'alpha': [0.1, 1.0, 10.0], 'l1_ratio' : [0.25, 0.50, 0.75]}
    elif model_name == 'random_forest':
        param_dct = {'n_estimators': [32, 64, 128], 
                     'min_samples_leaf': [1, 5, 10], 
                     'max_depth': [2, 4, 8, 16], 
                     'max_features': ['sqrt', 'log2']}
    else: 
        raise RuntimeError('Unsupported `model_name` inputted!')

    return param_dct

def get_model(model_name, problem_type): 
    """Return the appropriate model to run through a GridSearchCV

    Args:
    ----
        model_name: str
        problem_type: str

    Return: 
    ------
        varied: insantiated model object
    """
    # if user isn't "sallamander", it's on a dedicated instance - use all cores.
    num_usable_cores = multiprocessing.cpu_count() \
        if os.environ['USER'] != 'sallamander' else 1
    rand_state=609

    if model_name == 'linear':
        model = ElasticNet(random_state=rand_state)
    elif model_name == 'logistic': 
        model = SGDClassifier(loss='log', random_state=rand_state)
    elif model_name == 'random_forest':
        if problem_type == 'regression':
            model = RandomForestRegressor(n_jobs = num_usable_cores, 
                                          random_state=rand_state)
        elif problem_type == 'classification': 
            model = RandomForestClassifier(n_jobs = num_usable_cores, 
                                           random_state=rand_state)
    else: 
        raise RuntimeError('Unsupported `model_name` inputted!')

    return model 

if __name__ == '__main__':
    if len(sys.argv) < 4:
        msg = "Usage: python non_net_model.py problem_type model_name max_features"
        raise RuntimeError(msg)
    else: 
        problem_type = sys.argv[1]
        model_name = sys.argv[2]
        max_features = int(sys.argv[3])
 
    raw_reviews_fp = 'work/raw_reviews.pkl'
    ratios_fp = 'work/filtered_ratios.npy'

    with open(raw_reviews_fp, 'rb') as f: 
        raw_reviews = pickle.load(f) # `raw_reviews` are tokenized
    ratios = np.load(ratios_fp)
    review_txts = [' '.join(review) for review in raw_reviews]

    X_train, X_test, y_train, y_test = train_test_split(review_txts, ratios, 
                                                        test_size=0.2, 
                                                        random_state=609)
    
    vectorizer = TfidfVectorizer(max_features=max_features)
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    train_mean = y_train.mean()
    train_guessing_error = intellegently_guess(y_train, train_mean)
    test_guessing_error = intellegently_guess(y_test, train_mean)
    print('Train needs to beat... {}'.format(train_guessing_error))
    print('Test needs to beat... {}'.format(test_guessing_error))
    
    
    grid_params = get_grid_params(model_name)
    model = get_model(model_name, problem_type)
