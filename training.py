# training.py

import pickle
import models
import os

def train_model(model_name, best_params, X_train, y_train):
    clf = models.get_reg(model_name, best_params, X_train, y_train)
    print(f'Training {model_name}...')
    if model_name != 'XGBoost' and model_name != 'MLP':
        clf.fit(X_train, y_train)
    return clf

def save_model(model, path):
    filename = os.path.join(path, 'model.pkl')
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
