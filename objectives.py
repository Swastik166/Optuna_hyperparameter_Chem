# objectives.py

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, ConstantKernel, WhiteKernel
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_squared_error
import optuna
import numpy as np

class RandomForestObjective(object):
    def __init__(self, X, y, X_val, y_val):
        self.X = X
        self.y = y
        self.X_val = X_val
        self.y_val = y_val

    def __call__(self, trial):
        n_estimators = trial.suggest_int('n_estimators', 10, 1010, step = 50)
        max_depth = trial.suggest_int('max_depth', 1, 100)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)
        reg = RandomForestRegressor(n_estimators=n_estimators, 
                                    max_depth=max_depth, 
                                    min_samples_split=min_samples_split, 
                                    min_samples_leaf=min_samples_leaf, random_state=42)
        reg.fit(self.X, self.y)
        train_preds = reg.predict(self.X)
        train_rmse = np.sqrt(mean_squared_error(self.y, train_preds))
        val_preds = reg.predict(self.X_val)
        val_rmse = np.sqrt(mean_squared_error(self.y_val, val_preds))
        score = np.abs(train_rmse - val_rmse)
        return score

class GradientBoostingObjective(object):
    def __init__(self, X, y, X_val, y_val):
        self.X = X
        self.y = y
        self.X_val = X_val
        self.y_val = y_val

    def __call__(self, trial):
        n_estimators = trial.suggest_int('n_estimators', 10, 3010, step=100)
        max_depth = trial.suggest_int('max_depth', 1, 15)
        learning_rate = trial.suggest_categorical('learning_rate', [0.01, 0.1])
        subsample = trial.suggest_categorical('subsample', [0.001, 0.01, 0.1, 1]) #C_val
        min_samples_split = trial.suggest_categorical('min_samples_split', [2, 5, 10])
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
        alpha = trial.suggest_float('alpha', 0.1, 0.9) 
        reg = GradientBoostingRegressor(loss = 'huber', n_estimators=n_estimators, 
                                        max_depth=max_depth, learning_rate=learning_rate,
                                        subsample=subsample, min_samples_split=min_samples_split,
                                        min_samples_leaf=min_samples_leaf, alpha=alpha,
                                        random_state=42)
        reg.fit(self.X, self.y)
        train_preds = reg.predict(self.X)
        train_rmse = np.sqrt(mean_squared_error(self.y, train_preds))
        val_preds = reg.predict(self.X_val)
        val_rmse = np.sqrt(mean_squared_error(self.y_val, val_preds))
        score = np.abs(train_rmse - val_rmse)
        return score

class SVRObjective(object):
    def __init__(self, X, y, X_val, y_val):
        self.X = X
        self.y = y
        self.X_val = X_val
        self.y_val = y_val

    def __call__(self, trial):
        C = trial.suggest_categorical('C', [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100])
        epsilon = trial.suggest_float('epsilon', 1e-6, 1e-1, log=True)
        kernel = trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly'])
        
        reg = SVR(C=C, epsilon=epsilon, kernel=kernel)
        reg.fit(self.X, self.y)
        train_preds = reg.predict(self.X)
        train_rmse = np.sqrt(mean_squared_error(self.y, train_preds))
        val_preds = reg.predict(self.X_val)
        val_rmse = np.sqrt(mean_squared_error(self.y_val, val_preds))
        score = np.abs(train_rmse - val_rmse)
        return score

class GaussianProcessObjective(object):
    def __init__(self, X, y, X_val, y_val):
        self.X = X
        self.y = y
        self.X_val = X_val
        self.y_val = y_val

    def __call__(self, trial):
        kernel_type = trial.suggest_categorical('kernel_type', ['RBF', 'Matern', 'RationalQuadratic', 'ConstantKernel'])
        
        if kernel_type == 'RBF':
            length_scale = trial.suggest_float('length_scale', 1e-2, 1e2, log=True)
            kernel = RBF(length_scale=length_scale)
            
        elif kernel_type == 'Matern':
            length_scale = trial.suggest_float('length_scale', 1e-2, 1e2, log=True)
            nu = trial.suggest_float('nu', 0.5, 2.5)
            kernel = Matern(length_scale=length_scale, nu=nu)
            
        elif kernel_type == 'RationalQuadratic':
            length_scale = trial.suggest_float('length_scale', 1e-3, 1e3, log=True)
            alpha = trial.suggest_float('alpha', 0.1, 10.0)
            kernel = RationalQuadratic(length_scale=length_scale, alpha=alpha)
            
        elif kernel_type == 'ConstantKernel':
            length_scale = trial.suggest_float('length_scale', 1e-2, 1e2, log=True)
            nu = trial.suggest_float('nu', 0.5, 2.5)
            kernel = ConstantKernel() * Matern(length_scale=length_scale, nu=nu) + WhiteKernel()
            
            
        est = make_pipeline(StandardScaler(), GaussianProcessRegressor(kernel=kernel, random_state=42))
        reg = TransformedTargetRegressor(regressor=est, transformer=StandardScaler())
        warnings.filterwarnings(action='ignore', category=ConvergenceWarning)
        
        reg.fit(self.X, self.y)
        train_preds = reg.predict(self.X)
        train_rmse = np.sqrt(mean_squared_error(self.y, train_preds))
        val_preds = reg.predict(self.X_val)
        val_rmse = np.sqrt(mean_squared_error(self.y_val, val_preds))
        score = np.abs(train_rmse - val_rmse)

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
                 'booster': trial.suggest_categorical('booster', ['gbtree', 'dart']),
                 'lambda': trial.suggest_float('lambda', 1e-3, 1.0, log=True),
                 'alpha': trial.suggest_float('alpha', 1e-3, 1.0, log=True),
                 'max_depth' : trial.suggest_int('max_depth', 1, 9),
                 'eta': trial.suggest_float('eta', 1e-2, 1.0, log=True),
                 'gamma': trial.suggest_float('gamma', 1e-2, 1.0, log=True),
                 'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide'])
                 }
        
            
        if param['booster'] == 'dart':
            param['sample_type'] = trial.suggest_categorical('sample_type', ['uniform', 'weighted'])
            param['normalize_type'] = trial.suggest_categorical('normalize_type', ['tree', 'forest'])
            param['rate_drop'] = trial.suggest_float('rate_drop', 1e-2, 1.0, log=True)
            param['skip_drop'] = trial.suggest_float('skip_drop', 1e-2, 1.0, log=True)
            
        pruning_callback = optuna.integration.XGBoostPruningCallback(trial, 'validation-rmse')
        dtrain = xgb.DMatrix(self.X, label=self.y)
        dval = xgb.DMatrix(self.X_val, label=self.y_val)
        
        param['tree_method'] = 'gpu_hist'
        param['gpu_id'] = 3   
        
        bst = xgb.train(param, dtrain, evals=[(dval, 'validation')], callbacks=[pruning_callback], verbose_eval = False)
        train_preds = bst.predict(dtrain)
        train_rmse = np.sqrt(mean_squared_error(self.y, train_preds))
        preds = bst.predict(dval)
        val_rmse = np.sqrt(mean_squared_error(self.y_val, preds))
        score = np.abs(train_rmse - val_rmse)
        return score
    

class MLP(object):
    def __init__(self, X, y, X_val, y_val):
        self.X = X
        self.y = y
        self.X_val = X_val
        self.y_val = y_val

    def __call__(self, trial):
        hidden_layer_sizes = trial.suggest_int('hidden_layer_sizes', 50, 500, step=50)
        activation = trial.suggest_categorical('activation', ['identity', 'logistic', 'tanh', 'relu'])
        alpha = trial.suggest_float('alpha', 1e-6, 1e-3
                                    , log=True)
        reg = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver='adam', alpha=alpha, random_state=42)
        reg.fit(self.X, self.y)
        train_preds = reg.predict(self.X)
        train_rmse = np.sqrt(mean_squared_error(self.y, train_preds))
        val_preds = reg.predict(self.X_val)
        val_rmse = np.sqrt(mean_squared_error(self.y_val, val_preds))
        score = np.abs(train_rmse - val_rmse)
        return score
        