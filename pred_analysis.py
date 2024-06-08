import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score

def generate_plots(pred_path):
    files = os.listdir(pred_path)
    for file in files:
        sub_files = os.listdir(os.path.join(pred_path, file))
        for sub_file in sub_files:
            csv_files = os.listdir(os.path.join(pred_path, file, sub_file))
            csv_files = [f for f in csv_files if f.endswith('.csv')]
            for csv in csv_files:
                df = pd.read_csv(os.path.join(pred_path, file, sub_file, csv))
                df = df.groupby('Sl. No.').mean()
                r2 = r2_score(df['Yield'], df['Predictions'])
                plt.figure(figsize=(10,6))
                sns.regplot(x='Yield', y='Predictions', data=df, line_kws={'color': 'red'})
                plt.xlabel('Actual', fontsize=12)
                plt.title(f'Predicted vs Actual for {file}_{sub_file} with R2: {r2:.2f}', fontsize=14)
                plt.ylabel('Predicted', fontsize=12)
                plt.tight_layout()
                save_path = os.path.join(pred_path, 'plots')
                os.makedirs(save_path, exist_ok=True)
                save_path = os.path.join(save_path, f'{file}_{sub_file}.png')
                plt.savefig(save_path, bbox_inches='tight')
                plt.close()
    print('All prediction plots saved successfully')
