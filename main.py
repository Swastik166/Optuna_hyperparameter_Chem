# main.py

import argparse
import os
import glob
import models
import tuning
import training
import evaluation
import utils
import pandas as pd
from openpyxl import Workbook
import time
import analyze_metrics


def create_directories(base_path, subdirs):
    for subdir in subdirs:
        path = os.path.join(base_path, subdir)
        os.makedirs(path, exist_ok=True)
    return {subdir: os.path.join(base_path, subdir) for subdir in subdirs}


def main(args, pred_path, metrics_path):
    
    files = glob.glob(os.path.join(args.input_folder, "*.xlsx"))
    

    
    for file in files:
        basename = os.path.basename(file).split('.')[0]
        
        xls = pd.ExcelFile(file)
        
        base_metrics_path = os.path.join(metrics_path, basename)
        base_pred_path = os.path.join(pred_path, basename)
        #dirs = create_directories(args.save_path, [f"{basename}_metrics", f"{basename}_predictions"])

               
        for sheet_name in xls.sheet_names:

            print('\n', '='*50)
            print('\n', '='*50, '\n')
            print(f'Starting Hyperparameter Tuning, Training, and Evaluation for, {basename},\n {sheet_name}, ...')
                        
            df = utils.load_data(file, sheet_name)
            X_train, X_val, X_test, y_train, y_val, y_test = utils.split_data(df, args.train_idx, args.val_idx)

            for model_name in args.models:
                start = time.time()
               
                study, best_params = tuning.tune_model(model_name, X_train, y_train, X_val, y_val)
                print(f'Best parameters for {model_name}: {best_params}')
                
                print(f'Time taken for {model_name}: {time.time()-start:.2f} seconds')
                print('-'*50)
                print('\n')
                
                clf = training.train_model(model_name, best_params, X_train, y_train)
                metrics, predictions = evaluation.evaluate_model(df, clf, X_train, y_train, X_test, y_test, args.val_idx,sheet_name)
                
                
                #Directory handling
                model_dir = os.path.join(args.save_path, model_name, f"{basename}_results", sheet_name)
                pred_dir = os.path.join(base_pred_path, model_name)
                
                os.makedirs(model_dir, exist_ok=True)
                os.makedirs(pred_dir, exist_ok=True)
                

                
                
              
                if args.visualize:
                    tuning.opt_plots(study, model_dir)
    

                tuning.save_best_params(best_params, model_dir)
                training.save_model(clf, model_dir)
                            

                evaluation.save_results(metrics, predictions, base_metrics_path, pred_dir)
            print(f'Finished Hyperparameter Tuning, Training, and Evaluation for, {basename},\n {sheet_name}, ...')
            print('\n', '='*50)
            print('Analysis of Metrics Started...')
            analyze_metrics.generate_plots(metrics_path, basename)
            
        analyze_metrics.generate_mean_rmse_plot(metrics_path)
        print('Analysis of Metrics Completed...')
        print('\n', '='*50)
        print('\n', '='*50, '\n')
        



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter tuning, training, and evaluation of regression models.")
    parser.add_argument("--input_folder", type=str, required=True, help="Input folder containing the Excel files.")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the results.")
    parser.add_argument("--models", nargs="+", default=["RandomForest", "GradientBoosting", "SVR", "GaussianProcess", "XGBoost", "MLP"], help="List of models to use.")
    parser.add_argument("--train_idx", type=int, default=184, help="Train size for splitting the data.")
    parser.add_argument("--val_idx", type=int, default=210, help="Validation size for splitting the data.")
    parser.add_argument("--visualize", action="store_true", help="Visualize the results.")
    args = parser.parse_args()

    #Create a new folder named 'optuna_results' in the save path if it does not exist
    
    args.save_path = os.path.join(args.save_path, "optuna_results")
    os.makedirs(args.save_path, exist_ok=True)
    
    pred_path = os.path.join(args.save_path, 'predictions')
    os.makedirs(pred_path, exist_ok=True)
    
    metrics_path = os.path.join(args.save_path, 'metrics')
    os.makedirs(metrics_path, exist_ok=True)

    main(args, pred_path, metrics_path)
