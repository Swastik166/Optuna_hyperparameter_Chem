# Optuna_hyperparameter_Chem
An end-to-end pipeline for hyperparameter tuning for chemical data for various models.

#Usage

Install the required libraries with the following command:
'''pip install -r requirements.txt'''

General Usage
''' python main.py --input_folder '/path/to/input/folder' --save_path 'path/to/save/folder' '''

For hyperparameter tuning of specific models only mention them as string for the --model argument. 
Example
'''python main.py --input_folder '/raid/aiccg/rbsunoj/swastik/C-H-activation/repository-esque implementation/fingerprints' --save_path '/raid/aiccg/rbsunoj/swastik/C-H-activation/repository-esque implementation/fingerprints' --models 'RandomForest' 'SVR' 
'''