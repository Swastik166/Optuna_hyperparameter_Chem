# evaluation.py

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import xgboost as xgb
import torch
import os


def ensure_dir_exists(path):
    os.makedirs(path, exist_ok=True)


def evaluate_model(df, model, X_train, y_train, X_test, y_test, val, sheet_name):
    if model.__class__.__name__ == 'Booster':
        dtrain = xgb.DMatrix(X_train)
        dtest = xgb.DMatrix(X_test)
        train_preds = model.predict(dtrain)
        test_preds = model.predict(dtest)
        
    elif isinstance(model, torch.nn.Module):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cuda':
            model = model.to(device)
            torch.cuda.set_device(6)
        model.eval()  # Set the model to evaluation mode
        
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            train_preds = model(X_train_tensor).cpu().numpy().flatten()
            test_preds = model(X_test_tensor).cpu().numpy().flatten()
        
    else:
        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)
    
    metrics = {
        'Model': model.__class__.__name__,
        'MAE_train': mean_absolute_error(y_train, train_preds),
        'MAE_test': mean_absolute_error(y_test, test_preds),
        'MSE_train': mean_squared_error(y_train, train_preds),
        'MSE_test': mean_squared_error(y_test, test_preds),
        'R2_train': r2_score(y_train, train_preds),
        'R2_test': r2_score(y_test, test_preds),
        'RMSE_train': mean_squared_error(y_train, train_preds, squared=False),
        'RMSE_test': mean_squared_error(y_test, test_preds, squared=False),
        'sheet_name': sheet_name,
    }
    
    pred_df = df.iloc[int(val):].copy()
    pred_df = pred_df.iloc[:, :3]
    pred_df['Predictions'] = test_preds
    pred_df['Model'] = model.__class__.__name__
    pred_df['sheet_name'] = sheet_name
    
    print('Evaluation Done..........')
    return metrics, pred_df

def save_results(metrics, predictions, metrics_path, predictions_path):
    metrics_df = pd.DataFrame([metrics])
    metrics_file = os.path.join(metrics_path, 'metrics.csv')
    preds_file = os.path.join(predictions_path, 'predictions.csv')
    
    ensure_dir_exists(metrics_path)
    ensure_dir_exists(predictions_path)
    
    #Saving metrics
    if not os.path.isfile(metrics_file):
        metrics_df.to_csv(metrics_file, mode='w', header=True, index=False)
    else:
        metrics_df.to_csv(metrics_file, mode='a', header=False, index=False)
        
    #Saving predictions
    if not os.path.isfile(preds_file):
        predictions.to_csv(preds_file, mode='w', header=True, index=False)
    else:
        predictions.to_csv(preds_file, mode='a', header=False, index=False)

    
    print('Files Saved.....')
    print('-'*50)
    print('\n')
    print('\n')
    
    


    