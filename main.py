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


def main(args):
    
    files = glob.glob(os.path.join(args.input_folder, "*.xlsx"))
    
    for file in files:
        basename = os.path.basename(file).split('.')[0]
        
        xls = pd.ExcelFile(file)
        
        for sheet_name in xls.sheet_names:
            print('-'*50)
            print(f'Starting Hyperparameter Tuning, Training, and Evaluation for, {basename},\n {sheet_name}, ...')
                        
            df = utils.load_data(file, sheet_name)
            X_train, X_val, X_test, y_train, y_val, y_test = utils.split_data(df, args.train_idx, args.val_idx)

            for model_name in args.models:
                
               
                
                study, best_params = tuning.tune_model(model_name, X_train, y_train, X_val, y_val)
                clf = training.train_model(model_name, best_params, X_train, y_train)
                metrics, predictions = evaluation.evaluate_model(clf, X_train, y_train, X_test, y_test)
                
                #Directory handling
                model_path = os.path.join(args.save_path, model_name)
                file_path = os.path.join(model_path, basename+'_results')
                sheet_path = os.path.join(file_path, sheet_name)
                
                os.makedirs(sheet_path, exist_ok=True)
                
              
                if args.visualize:
                    tuning.opt_plots(study, sheet_path)

                metrics_filename = (f"{args.save_path}/{basename}_{sheet_name}_{model_name}_metrics.csv")
                predictions_filename = f"{args.save_path}/{basename}_{sheet_name}_{model_name}_predictions.csv"
                model_filename = (f"{args.save_path}/{basename}_{sheet_name}_{model_name}_model.pkl")
                params_filename = (f"{args.save_path}/{basename}_{sheet_name}_{model_name}_params.txt")

                tuning.save_best_params(best_params, params_filename)
                training.save_model(clf, model_filename)
                evaluation.save_results(metrics, predictions, metrics_filename, predictions_filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter tuning, training, and evaluation of regression models.")
    parser.add_argument("--input_folder", type=str, required=True, help="Input folder containing the Excel files.")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the results.")
    parser.add_argument("--models", nargs="+", default=["RandomForest", "GradientBoosting", "SVR", "GaussianProcess", "XGBoost"], help="List of models to use.")
    parser.add_argument("--train_idx", type=int, default=184, help="Train size for splitting the data.")
    parser.add_argument("--val_idx", type=int, default=210, help="Validation size for splitting the data.")
    parser.add_argument("--visualize", action="store_true", help="Visualize the results.")
    args = parser.parse_args()

    #Create a new folder named 'optuna_results' in the save path if it does not exist
    
    args.save_path = os.path.join(args.save_path, "optuna_results")
    os.makedirs(args.save_path, exist_ok=True)

    

    main(args)
