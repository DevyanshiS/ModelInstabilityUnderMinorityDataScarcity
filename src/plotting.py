import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob

def get_latest_results():
    list_of_files = glob.glob('outputs/results/*.csv')
    if not list_of_files:
        raise FileNotFoundError("No results CSV found in outputs/results/")
    latest_file = max(list_of_files, key=os.path.getctime)
    print(f"Plotting results from: {latest_file}")
    return pd.read_csv(latest_file)

def plot_performance_heatmap(df):
    """
    Plots a heatmap of mean F1 score across Models and Techniques 
    averaged over all datasets.
    """
    plt.figure(figsize=(12, 8))
    pivot = df.pivot_table(index='Technique', columns='Model', values='F1_mean', aggfunc='mean')
    
    # Sort techniques such that Base is first and AMBA is last
    order = ['Base', 'ROS', 'RUS', 'SMOTE', 'ADASYN', 'TomekLinks', 
             'ClassWeight', 'BalancedBagging', 'EasyEnsemble', 'ThresholdMoving', 'AMBA']
    actual_order = [t for t in order if t in pivot.index] + [t for t in pivot.index if t not in order]
    pivot = pivot.loc[actual_order]
    
    sns.heatmap(pivot, annot=True, cmap='viridis', fmt='.3f')
    plt.title('Mean F1 Score across all Datasets')
    plt.tight_layout()
    plt.savefig('outputs/f1_heatmap.png', dpi=300)
    plt.close()
    
def plot_stability_profile(df):
    """
    Plots the Coefficient of Variation of F1 (Stability) for different baselines.
    Lower CV -> Higher Stability.
    """
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='Technique', y='F1_CV', palette='Set2')
    plt.axhline(0, color='r', linestyle='--')
    plt.xticks(rotation=45)
    plt.title('Stability Profile: Coefficient of Variation of F1 (Lower is better)')
    plt.ylabel('CV of F1')
    plt.tight_layout()
    plt.savefig('outputs/stability_profile_cv_f1.png', dpi=300)
    plt.close()
    
def plot_dataset_breakdown(df):
    """
    Plots F1 score improvements for AMBA vs Base grouped by dataset.
    """
    base_df = df[df['Technique'] == 'Base'].groupby('Dataset')['F1_mean'].mean()
    amba_df = df[df['Technique'] == 'AMBA'].groupby('Dataset')['F1_mean'].mean()
    
    comp_df = pd.DataFrame({'Base': base_df, 'AMBA': amba_df})
    comp_df.plot(kind='bar', figsize=(10, 6), color=['gray', '#2ca02c'])
    
    plt.title('Base vs AMBA Performance by Dataset (Mean F1)')
    plt.ylabel('Mean F1 Score')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig('outputs/dataset_breakdown.png', dpi=300)
    plt.close()

def generate_visualizations():
    try:
        df = get_latest_results()
        
        # Clean data (remove error rows if any)
        if 'Error' in df.columns:
            df = df[df['Error'].isna()]
            
        plot_performance_heatmap(df)
        plot_stability_profile(df)
        plot_dataset_breakdown(df)
        
        print("Visualizations saved to outputs/")
    except Exception as e:
        print(f"Failed to generate visualizations: {e}")

if __name__ == "__main__":
    generate_visualizations()
