import os
import pandas as pd
import matplotlib.pyplot as plt

def load_data(metrics_path, dataset_name, file):
    df = pd.read_csv(os.path.join(metrics_path, dataset_name, file))
    return df

def calculate_mean_data(model_data):
    mean_data = pd.DataFrame({
        'sheet_name': ['mean'],
        'MAE_train': [model_data['MAE_train'].mean()],
        'MAE_test': [model_data['MAE_test'].mean()],
        'MSE_train': [model_data['MSE_train'].mean()],
        'MSE_test': [model_data['MSE_test'].mean()],
        'R2_train': [model_data['R2_train'].mean()],
        'R2_test': [model_data['R2_test'].mean()],
        'RMSE_train': [model_data['RMSE_train'].mean()],
        'RMSE_test': [model_data['RMSE_test'].mean()]
    })
    return mean_data

def plot_metrics(combined_data, model, dataset_name, metrics_path):
    metrics = ['MAE', 'MSE', 'R2', 'RMSE']
    colors = ['#1f77b4', '#ff7f0e']

    fig, axes = plt.subplots(2, 2, figsize=(20, 10))
    fig.suptitle(f'Performance Metrics for {model} on {dataset_name} Dataset')

    for i, metric in enumerate(metrics):
        ax = axes[i // 2, i % 2]
        train_metric = f'{metric}_train'
        test_metric = f'{metric}_test'

        combined_data.set_index('sheet_name')[[train_metric, test_metric]].plot(kind='bar', ax=ax, color=colors)
        ax.set_title(f'{metric} by Sheet Name', fontsize=14)
        ax.set_ylabel(f'{metric}', fontsize=12)
        ax.set_xlabel('Sheet Name', fontsize=12)
        ax.legend(['Train', 'Test'], loc='best', fontsize=10)
        ax.tick_params(axis='x', rotation=45)
        ax.tick_params(axis='y', labelsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = os.path.join(metrics_path, dataset_name, f'{model}_metrics_plot.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

def plot_mean_rmse(mean_rmse_df, dataset_name, metrics_path):
    colors = ['#1f77b4', '#ff7f0e']

    fig, ax = plt.subplots(figsize=(15, 8))
    mean_rmse_df.set_index('Model')[['Mean_RMSE_train', 'Mean_RMSE_test']].plot(kind='bar', ax=ax, color=colors)
    ax.set_title(f'Mean RMSE for Models on {dataset_name} Dataset', fontsize=20)
    ax.set_ylabel('Mean RMSE', fontsize=15)
    ax.set_xlabel('Model', fontsize=15)

    model_dict = {
        'RandomForestRegressor': 'RF',
        'GradientBoostingRegressor': 'GB',
        'SVR': 'SVR',
        'TransformedTargetRegressor': 'GP',
        'Booster': 'XGB',
        'PyTorchMLP': 'DNN'
    }
    ax.set_xticklabels([model_dict.get(model, model) for model in mean_rmse_df['Model']], rotation=45, ha='right')

    rects = ax.patches
    labels = mean_rmse_df['Mean_RMSE_train'].tolist() + mean_rmse_df['Mean_RMSE_test'].tolist()

    for rect, label in zip(rects, labels):
        height = rect.get_height()
        ax.text(
            rect.get_x() + rect.get_width() / 2, height, f'{label:.2f}',
            ha='center', va='bottom', fontsize=12, color='black'
        )

    ax.legend(['Train', 'Test'], loc='best', fontsize=10)
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15)
    plt.tight_layout()
    save_path = os.path.join(metrics_path, dataset_name, 'mean_rmse_plot.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

def generate_plots(metrics_path, dataset_name):
    files = os.listdir(os.path.join(metrics_path, dataset_name))
    csv_files = [file for file in files if file.endswith('.csv')]

    mean_rmse_data = []
    for file in csv_files:
        df_test = load_data(metrics_path, dataset_name, file)
        models = df_test['Model'].unique()
        for model in models:
            model_data = df_test[df_test['Model'] == model]
            metrics_data = model_data.drop(columns=['Model', 'sheet_name'])
            mean_data = calculate_mean_data(model_data)
            combined_data = pd.concat([metrics_data, mean_data], ignore_index=True)
            plot_metrics(combined_data, model, dataset_name, metrics_path)

            mean_rmse_data.append({
                'Model': model,
                'Mean_RMSE_train': model_data['RMSE_train'].mean(),
                'Mean_RMSE_test': model_data['RMSE_test'].mean()
            })

    mean_rmse_df = pd.DataFrame(mean_rmse_data)
    plot_mean_rmse(mean_rmse_df, dataset_name, metrics_path)

    print(f'Plots saved for {dataset_name} dataset')

def generate_mean_rmse_plot(metrics_path):
    folders = os.listdir(metrics_path)
    for dataset_name in folders:
        generate_plots(metrics_path, dataset_name)

    print('All plots saved successfully')
