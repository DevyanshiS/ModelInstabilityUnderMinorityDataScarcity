import pandas as pd
import numpy as np
import glob
import os

def generate_report():
    list_of_files = glob.glob('outputs/results/*.csv')
    if not list_of_files:
        raise FileNotFoundError("No results CSV found in outputs/results/")
    latest_file = max(list_of_files, key=os.path.getctime)
    
    df = pd.read_csv(latest_file)
    
    # Keep only fully successful runs for aggregate reporting.
    if 'Status' in df.columns:
        df = df[df['Status'] == 'OK']
    elif 'Error' in df.columns:
        df = df[df['Error'].isna()]
    
    report = []
    report.append("# Experimental Results Report: Model Instability Under Minority Data Scarcity\n")
    
    report.append("## Overview\n")
    report.append("This report summarizes the experimental evaluation of the AMBA framework against 9 baselines across 4 datasets and 7 base classifiers.\n")
    
    report.append("## Overall Performance (Averaged Across all Datasets & Models)\n")
    
    # Aggregate F1, AUC, G-Mean across all datasets and models for each technique
    agg_df = df.groupby('Technique')[['F1_mean', 'ROC_AUC_mean', 'G_Mean_mean', 'F1_CV', 'MCC_std']].mean().sort_values(by='F1_mean', ascending=False)
    
    report.append(agg_df.round(4).to_markdown())
    report.append("\n\n")
    
    report.append("## Dataset Breakdown (Mean F1 Score per Technique)\n")
    pivot_ds = df.pivot_table(index='Technique', columns='Dataset', values='F1_mean', aggfunc='mean').round(4)
    # Order so AMBA is at the bottom for easy comparison
    if 'AMBA' in pivot_ds.index:
        order = [i for i in pivot_ds.index if i != 'AMBA'] + ['AMBA']
        pivot_ds = pivot_ds.loc[order]
    report.append(pivot_ds.to_markdown())
    report.append("\n\n")
    
    report.append("## Stability Analysis\n")
    report.append("A key contribution of AMBA is improving the stability of models under heavy class imbalance. The following table showcases the Coefficient of Variation (CV) for F1 Score (Lower is better).\n")
    cv_df = df.groupby('Technique')[['F1_CV', 'Recall_CV']].mean().sort_values(by='F1_CV', ascending=True).round(4)
    report.append(cv_df.to_markdown())
    report.append("\n\n")
    
    report.append("## Visualizations\n")
    report.append("The following visualizations have been generated and saved to the `outputs/` directory:\n")
    report.append("- `f1_heatmap.png`: Mean F1 Score across all Datasets (Models vs Techniques)\n")
    report.append("- `stability_profile_cv_f1.png`: Boxplot showing the distribution of F1 Coefficient of Variation for all techniques.\n")
    report.append("- `dataset_breakdown.png`: Bar chart comparing Base vs AMBA performance per dataset.\n")
    
    report.append("\n## Conclusion\n")
    report.append("Based on the aggregated metrics above, review the performance of `AMBA` in relation to standard techniques like `SMOTE`, `BalancedBagging`, and `ThresholdMoving`. If AMBA consistently ranks in the top tier for F1 and G-Mean while maintaining a low CV, the research hypothesis stated in the proposal is strongly supported.")
    
    with open('outputs/final_report.md', 'w') as f:
        f.write('\n'.join(report))
        
    print("Final report generated at outputs/final_report.md")

if __name__ == "__main__":
    generate_report()
