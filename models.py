# models.py

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, ConstantKernel, WhiteKernel
import xgboost as xgb
from sklearn.neural_network import MLPRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim

class PyTorchMLP(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes, activation, output_size=1):
        super(PyTorchMLP, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_layer_sizes)
        self.output = nn.Linear(hidden_layer_sizes, output_size)

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'logistic':
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Identity()

    def forward(self, x):
        x = self.activation(self.hidden(x))
        x = self.output(x)
        return x

def train_pytorch_mlp(X_train, y_train, hidden_layer_sizes, activation, alpha, learning_rate, device='cpu'):
    input_size = X_train.shape[1]
    model = PyTorchMLP(input_size, hidden_layer_sizes, activation).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=alpha)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)

    num_epochs = 1000
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

    return model


def get_reg(regressor_name, best_params, X_train, y_train):
    if regressor_name == 'RandomForest':
        return RandomForestRegressor(
            n_estimators=best_params['n_estimators'], 
            max_depth=best_params['max_depth'], 
            min_samples_split=best_params['min_samples_split'], 
            min_samples_leaf=best_params['min_samples_leaf'],
            random_state=42
        )
        
    elif regressor_name == 'GradientBoosting':
        return GradientBoostingRegressor(loss='huber',
            n_estimators=best_params['n_estimators'], 
            max_depth=best_params['max_depth'],
            learning_rate=best_params['learning_rate'],
            subsample=best_params['subsample'],
            min_samples_split=best_params['min_samples_split'],
            min_samples_leaf=best_params['min_samples_leaf'],
            alpha=best_params['alpha'],
            random_state=42
        )
        
    elif regressor_name == 'SVR':
        return SVR(
            C=best_params['C'], 
            epsilon=best_params['epsilon'], 
            kernel=best_params['kernel']
        )
        
    elif regressor_name == 'GaussianProcess':
        kernel_type = best_params['kernel_type']
        if kernel_type == 'RBF':
            kernel = RBF(length_scale=best_params['length_scale'])
        elif kernel_type == 'Matern':
            kernel = Matern(length_scale=best_params['length_scale'], nu=best_params['nu'])
        elif kernel_type == 'RationalQuadratic':
            kernel = RationalQuadratic(length_scale=best_params['length_scale'], alpha=best_params['alpha'])
        elif kernel_type == 'ConstantKernel':
            kernel = ConstantKernel() * Matern(length_scale=best_params['length_scale'], nu=best_params['nu']) + WhiteKernel()
        
        est = make_pipeline(StandardScaler(), GaussianProcessRegressor(kernel=kernel, random_state=42))
        reg = TransformedTargetRegressor(regressor=est, transformer=StandardScaler())
        return reg
    
    elif regressor_name == 'XGBoost':
        param = {'verbosity': 0, 
                 'objective': 'reg:squarederror',
                 'eval_metric': 'rmse',
                 'booster': best_params['booster'],
                 'lambda': best_params['lambda'],
                 'alpha': best_params['alpha'],
                 'max_depth': best_params['max_depth'],
                 'eta': best_params['eta'],
                 'gamma': best_params['gamma'],
                 'grow_policy': best_params['grow_policy']}

        if best_params['booster'] == 'dart':
            param.update({
                'sample_type': best_params['sample_type'],
                'normalize_type': best_params['normalize_type'],
                'rate_drop': best_params['rate_drop'],
                'skip_drop': best_params['skip_drop']
            })
        dtrain = xgb.DMatrix(X_train, label=y_train)
        param['tree_method'] = 'gpu_hist'
        param['gpu_id'] = 6
        return xgb.train(param, dtrain, num_boost_round=100)
    
    elif regressor_name == 'MLP':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        #choose gpu number
        if device == 'cuda':
            torch.cuda.set_device(6)
        return train_pytorch_mlp(
            X_train, y_train, 
            hidden_layer_sizes=best_params['hidden_layer_sizes'], 
            activation=best_params['activation'], 
            alpha=best_params['alpha'], 
            learning_rate=best_params['learning_rate'], 
            device=device
        )
