# training.py

import pickle
import models

def train_model(model_name, best_params, X_train, y_train):
    clf = models.get_classifier(model_name, best_params, X_train, y_train)
    clf.fit(X_train, y_train)
    return clf

def save_model(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
