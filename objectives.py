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
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error

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
        sc = np.abs(train_rmse - val_rmse)
        score = sc + val_rmse
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
        sc = np.abs(train_rmse - val_rmse)
        
        score = sc + val_rmse
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
        sc = np.abs(train_rmse - val_rmse)
        score = sc + val_rmse
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
        sc = np.abs(train_rmse - val_rmse)
        score = sc + val_rmse

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
        param['gpu_id'] = 6   
        
        bst = xgb.train(param, dtrain, evals=[(dval, 'validation')], callbacks=[pruning_callback], verbose_eval = False)
        train_preds = bst.predict(dtrain)
        train_rmse = np.sqrt(mean_squared_error(self.y, train_preds))
        preds = bst.predict(dval)
        val_rmse = np.sqrt(mean_squared_error(self.y_val, preds))
        sc = np.abs(train_rmse - val_rmse)
        score = sc + val_rmse
        return score
    

class MLP(object):
    def __init__(self, X, y, X_val, y_val, device='cuda'):
        self.device = device
        #choose gpu number
        if device == 'cuda':
            torch.cuda.set_device(6)
        self.X = torch.tensor(X, dtype=torch.float32).to(device)
        self.y = torch.tensor(y, dtype=torch.float32).to(device).view(-1, 1)
        self.X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
        self.y_val = torch.tensor(y_val, dtype=torch.float32).to(device).view(-1, 1)


    def __call__(self, trial):
        input_size = self.X.shape[1]
        output_size = 1

        hidden_layer_sizes = trial.suggest_int('hidden_layer_sizes', 50, 500, step=50)
        activation_name = trial.suggest_categorical('activation', ['relu', 'tanh', 'logistic'])
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        alpha = trial.suggest_float('alpha', 1e-6, 1e-3, log=True)

        if activation_name == 'relu':
            activation = nn.ReLU()
        elif activation_name == 'tanh':
            activation = nn.Tanh()
        elif activation_name == 'logistic':
            activation = nn.Sigmoid()
        else:
            activation = nn.Identity()

        class MLP_m(nn.Module):
            def __init__(self, input_size, hidden_layer_sizes, output_size, activation):
                super(MLP_m, self).__init__()
                self.model = nn.Sequential(
                    nn.Linear(input_size, hidden_layer_sizes),
                    activation,
                    nn.Linear(hidden_layer_sizes, output_size)
                )

            def forward(self, x):
                return self.model(x)

        model = MLP_m(input_size, hidden_layer_sizes, output_size, activation).to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=alpha)

        num_epochs = 1000
        early_stopping_patience = 10
        best_val_loss = float('inf')
        epochs_no_improve = 0

        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(self.X)
            loss = criterion(outputs, self.y)
            loss.backward()
            optimizer.step()

            model.eval()
            val_outputs = model(self.X_val)
            val_loss = criterion(val_outputs, self.y_val).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= early_stopping_patience:
                break

        train_preds = model(self.X).detach().cpu().numpy()
        train_rmse = np.sqrt(mean_squared_error(self.y.cpu().numpy(), train_preds))
        val_preds = model(self.X_val).detach().cpu().numpy()
        val_rmse = np.sqrt(mean_squared_error(self.y_val.cpu().numpy(), val_preds))

        sc = np.abs(train_rmse - val_rmse)
        score = sc + val_rmse 
        return score
        