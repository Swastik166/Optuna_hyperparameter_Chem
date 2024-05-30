# objectives.py

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic
import xgboost as xgb
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_squared_error
import optuna

class RandomForestObjective(object):
    def __init__(self, X, y, X_val, y_val):
        self.X = X
        self.y = y
        self.X_val = X_val
        self.y_val = y_val

    def __call__(self, trial):
        n_estimators = trial.suggest_int('n_estimators', 1, 100)
        max_depth = trial.suggest_int('max_depth', 1, 100)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)
        clf = RandomForestRegressor(n_estimators=n_estimators, 
                                    max_depth=max_depth, 
                                    min_samples_split=min_samples_split, 
                                    min_samples_leaf=min_samples_leaf)
        clf.fit(self.X, self.y)
        val_preds = clf.predict(self.X_val)
        score = mean_squared_error(self.y_val, val_preds)
        return score

class GradientBoostingObjective(object):
    def __init__(self, X, y, X_val, y_val):
        self.X = X
        self.y = y
        self.X_val = X_val
        self.y_val = y_val

    def __call__(self, trial):
        n_estimators = trial.suggest_int('n_estimators', 1, 100)
        max_depth = trial.suggest_int('max_depth', 1, 100)
        clf = GradientBoostingRegressor(n_estimators=n_estimators, 
                                        max_depth=max_depth)
        clf.fit(self.X, self.y)
        val_preds = clf.predict(self.X_val)
        score = mean_squared_error(self.y_val, val_preds)
        return score

class SVRObjective(object):
    def __init__(self, X, y, X_val, y_val):
        self.X = X
        self.y = y
        self.X_val = X_val
        self.y_val = y_val

    def __call__(self, trial):
        C = trial.suggest_float('C', 1e-3, 1e3, log=True)
        epsilon = trial.suggest_float('epsilon', 1e-5, 1e1, log=True)
        kernel = trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly'])
        clf = SVR(C=C, epsilon=epsilon, kernel=kernel)
        clf.fit(self.X, self.y)
        val_preds = clf.predict(self.X_val)
        score = mean_squared_error(self.y_val, val_preds)
        return score

class GaussianProcessObjective(object):
    def __init__(self, X, y, X_val, y_val):
        self.X = X
        self.y = y
        self.X_val = X_val
        self.y_val = y_val

    def __call__(self, trial):
        kernel_type = trial.suggest_categorical('kernel_type', ['RBF', 'Matern', 'RationalQuadratic'])
        if kernel_type == 'RBF':
            length_scale = trial.suggest_float('length_scale', 1e-3, 1e3, log=True)
            kernel = RBF(length_scale=length_scale)
        elif kernel_type == 'Matern':
            length_scale = trial.suggest_float('length_scale', 1e-3, 1e3, log=True)
            nu = trial.suggest_float('nu', 0.5, 2.5)
            kernel = Matern(length_scale=length_scale, nu=nu)
        elif kernel_type == 'RationalQuadratic':
            length_scale = trial.suggest_float('length_scale', 1e-3, 1e3, log=True)
            alpha = trial.suggest_float('alpha', 0.1, 10.0)
            kernel = RationalQuadratic(length_scale=length_scale, alpha=alpha)
        clf = GaussianProcessRegressor(kernel=kernel)
        warnings.filterwarnings(action='ignore', category=ConvergenceWarning)
        clf.fit(self.X, self.y)
        val_preds = clf.predict(self.X_val)
        score = mean_squared_error(self.y_val, val_preds)
        return score

class XGBoostObjective(object):
    def __init__(self, X, y, X_val, y_val):
        self.X = X
        self.y = y
        self.X_val = X_val
        self.y_val = y_val

    def __call__(self, trial):
        param = {'verbosity': 0, 
                 'objective': 'reg:squarederror',
                 'eval_metric': 'rmse',
                 'booster': trial.suggest_categorical('booster', ['gbtree', 'gblinear', 'dart']),
                 'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
                 'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True)}
        if param['booster'] in ['gbtree', 'dart']:
            param['max_depth'] = trial.suggest_int('max_depth', 1, 9)
            param['eta'] = trial.suggest_float('eta', 1e-8, 1.0, log=True)
            param['gamma'] = trial.suggest_float('gamma', 1e-8, 1.0, log=True)
            param['grow_policy'] = trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide'])
        if param['booster'] == 'dart':
            param['sample_type'] = trial.suggest_categorical('sample_type', ['uniform', 'weighted'])
            param['normalize_type'] = trial.suggest_categorical('normalize_type', ['tree', 'forest'])
            param['rate_drop'] = trial.suggest_float('rate_drop', 1e-8, 1.0, log=True)
            param['skip_drop'] = trial.suggest_float('skip_drop', 1e-8, 1.0, log=True)
        dtrain = xgb.DMatrix(self.X, label=self.y)
        dval = xgb.DMatrix(self.X_val, label=self.y_val)
        bst = xgb.train(param, dtrain, evals=[(dval, 'eval')], num_boost_round=100, early_stopping_rounds=10)
        preds = bst.predict(dval)
        score = mean_squared_error(self.y_val, preds)
        return score
