# evaluation.py

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd

def evaluate_model(model, X_train, y_train, X_test, y_test):
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    
    metrics = {
        'MAE_train': mean_absolute_error(y_train, train_preds),
        'MSE_train': mean_squared_error(y_train, train_preds),
        'R2_train': r2_score(y_train, train_preds),
        'RMSE_train': mean_squared_error(y_train, train_preds, squared=False),
        'MAE_test': mean_absolute_error(y_test, test_preds),
        'MSE_test': mean_squared_error(y_test, test_preds),
        'R2_test': r2_score(y_test, test_preds),
        'RMSE_test': mean_squared_error(y_test, test_preds, squared=False),
    }
    
    predictions = pd.DataFrame({'y_true': y_test, 'y_pred': test_preds})
    return metrics, predictions

def save_results(metrics, predictions, metrics_filename, predictions_filename):
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(metrics_filename, index=False)
    predictions.to_csv(predictions_filename, index=False)
