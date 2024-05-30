# tuning.py

import optuna
import models
import os
from objectives import RandomForestObjective, GradientBoostingObjective, SVRObjective, GaussianProcessObjective, XGBoostObjective

def tune_model(model_name, X, y, X_val, y_val):
    
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    print('-'*50)
    print(f'Tuning {model_name}...')
    print('-'*50)
    
    if model_name == 'RandomForest':
        objective = RandomForestObjective(X, y, X_val, y_val)
    elif model_name == 'GradientBoosting':
        objective = GradientBoostingObjective(X, y, X_val, y_val)
    elif model_name == 'SVR':
        objective = SVRObjective(X, y, X_val, y_val)
    elif model_name == 'GaussianProcess':
        objective = GaussianProcessObjective(X, y, X_val, y_val)
    elif model_name == 'XGBoost':
        objective = XGBoostObjective(X, y, X_val, y_val)
        
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100)
    
    return study, study.best_params

def opt_plots(study, filename):
    fig1 = optuna.visualization.plot_optimization_history(study)
    fig1.write_image(f'{filename}_opt_history.png')
    fig2 = optuna.visualization.plot_slice(study)
    fig3 = optuna.visualization.plot_contour(study)
    fig4 = optuna.visualization.plot_parallel_coordinate(study)
    fig5 = optuna.visualization.plot_param_importances(study)
    
    

def save_best_params(params, filename):
    with open(filename, 'w') as f:
        for key, value in params.items():
            f.write(f'{key}: {value}\n')
