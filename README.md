# Optuna_hyperparameter_Chem
An end-to-end pipeline for hyperparameter tuning for chemical data for various models.

## Usage

Install the required libraries with the following command:

```
pip install -r requirements.txt
```

General Usage
``` 
python main.py --input_folder '/path/to/input/folder' --save_path 'path/to/save/folder' --visualize
```

For hyperparameter tuning of specific models only mention them as string for the --model argument. 
Example
```
python main.py --input_folder '/path/to/input/folder'
               --save_path 'path/to/save/folder' --visualize
               --models 'RandomForest' 'SVR' 
```

## Input Files

The pipeline expects input files in Excel format (`.xlsx`). Each input file should contain multiple sheets, with each sheet representing a different dataset. The first three columns of each sheet should contain the following information:

- `Sl. No.`: The index of the data point.
- `SMILES`: SMILES
- `Target`: The target could be anything ex, yield, reaction barrier

The remaining columns should contain the fingerprint or any other feature




## Saved Files and Structure

The pipeline saves the following files and directories:

- `optuna_results`: The main directory where all the results are saved.
  - `<model_name>`: A directory for each model used in the pipeline.
    - `<dataset_name>_results`: A directory for each dataset used in the pipeline.
      - `<sheet_name>`: A directory for each sheet in the dataset.
        - `best_params.txt`: A text file containing the best hyperparameters found by Optuna for the model on the sheet.
        - `model.pkl`: A pickle file containing the trained model for the sheet.
        - `plot_optimization_history.png`: A plot showing the optimization history of the hyperparameters for the model on the sheet.
        - `plot_slice.png`: A plot showing the relationship between each hyperparameter and the objective function for the model on the sheet.
        - `plot_parallel_coordinate.png`: A plot showing the relationship between all hyperparameters and the objective function for the model on the sheet.
        - `plot_param_importances.png`: A plot showing the importance of each hyperparameter for the model on the sheet.
  - `metrics`: A directory containing the performance metrics for each model and each dataset.
    - `<dataset_name>`: A directory for each dataset used in the pipeline.
      - `metrics.csv`: A CSV file containing the performance metrics for each model and each sheet in the dataset.
      - `<model_name>_metrics_plot.png`: A plot showing the performance metrics for the model on the dataset.
      - `mean_rmse_plot.png`: A plot showing the mean RMSE for each model on the dataset.
  - `predictions`: A directory containing the predicted values for each model and each dataset.
    - `<dataset_name>`: A directory for each dataset used in the pipeline.
      - `<model_name>`: A directory for each model used in the pipeline.
        - `predictions.csv`: A CSV file containing the predicted values for the model on the dataset.
  - `plots`: A directory containing the regression plots for the actual vs predicted values for each model and each sheet in the input data.
    - `<dataset_name>_<sheet_name>.png`: A regression plot for the model and the sheet.

## Repository Structure

- `main.py`: The main script that orchestrates the entire pipeline.
- `models.py`: Defines the regression models that can be used in the pipeline.
- `objectives.py`: Defines the objective functions for hyperparameter tuning.
- `evaluation.py`: Contains functions for evaluating the models.
- `training.py`: Contains functions for training the models.
- `tuning.py`: Contains functions for hyperparameter tuning.
- `utils.py`: Contains utility functions for loading data and splitting it into training, validation, and test sets.
- `analyze_metrics.py`: Contains functions for analyzing the performance metrics of the models.
- `prediction_analysis.py`: Generates regression plots for the actual vs predicted values for each model and each sheet in the input data.
- `requirements.txt`: Lists the required Python libraries for running the pipeline.




Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.