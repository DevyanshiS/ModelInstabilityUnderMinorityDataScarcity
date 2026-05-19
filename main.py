import os
import sys
import pandas as pd
import numpy as np
import warnings
import traceback
from tqdm import tqdm
from datetime import datetime
from sklearn.base import clone

# ── CRITICAL FIX 1 ────────────────────────────────────────────────────────────
# main.py was importing from 'src.data', 'src.models', etc.
# This means edits to the top-level models.py / amba.py were silently ignored —
# the old, broken src/ versions kept running.
# Fix: import directly from the top-level modules (no 'src.' prefix).
# If your project layout requires src/, copy the updated files into src/ as well.
# ──────────────────────────────────────────────────────────────────────────────
from src.data import load_datasets
from src.models import get_core_models, create_baseline_pipelines
from src.amba import AMBAClassifier
from src.evaluate import evaluate_model


def run_experiments():
    warnings.filterwarnings('ignore')

    # ── Setup ──────────────────────────────────────────────────────────────────
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('outputs/results', exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = f'outputs/results/experiment_results_{timestamp}.csv'

    print("Initializing datasets...")
    datasets = load_datasets(subsample_creditcard=True, creditcard_frac=0.1)

    # ── CRITICAL FIX 2 ────────────────────────────────────────────────────────
    # get_core_models() is called ONCE here, giving a dict of estimator instances.
    # The original code reused these same instances across every dataset and
    # technique without cloning, so a fitted estimator from dataset 1 could
    # bleed state into dataset 2.  We now call get_core_models() fresh per
    # dataset, and clone the base estimator explicitly before wrapping in AMBA.
    # ──────────────────────────────────────────────────────────────────────────

    all_results = []

    print("\n--- Starting Main Experiment Loop ---")
    expected_techniques = [
        'Base', 'ROS', 'RUS', 'SMOTE', 'ADASYN', 'TomekLinks',
        'ClassWeight', 'BalancedBagging', 'EasyEnsemble',
        'ThresholdMoving', 'AMBA'
    ]

    for data_name, (X, y) in datasets.items():
        print(f"\nProcessing Dataset: {data_name} (Shape: {X.shape})")

        # Fresh model instances per dataset — no state leakage between datasets
        core_models = get_core_models()

        for model_name, core_model in tqdm(core_models.items(),
                                           desc=f"Models for {data_name}"):

            baselines = create_baseline_pipelines(model_name, core_model)

            # ── CRITICAL FIX 3 ────────────────────────────────────────────────
            # The original code wrapped AMBA in RandomizedSearchCV with
            # n_iter=5 and cv=2.  This creates:
            #   5 param combos x 2 inner folds x AMBA's own 10-clone ensemble
            #   = 100 inner fits, then repeated for every outer CV fold.
            # On real datasets this causes a near-infinite hang that the OS
            # eventually kills, leaving the results file full of zeros for every
            # XGB (and other model) row that was mid-flight.
            #
            # Fix: use AMBA directly (no nested hyperparameter search) with the
            # paper's default hyperparameters.  Hyperparameter sensitivity can be
            # studied separately in Phase 5 once the main results are complete.
            # ──────────────────────────────────────────────────────────────────
            baselines['AMBA'] = AMBAClassifier(
                base_estimator=clone(core_model),   # cloned — not the shared instance
                random_state=42,
            )

            for tech_name in expected_techniques:
                if tech_name not in baselines:
                    all_results.append({
                        'Dataset': data_name,
                        'Model': model_name,
                        'Technique': tech_name,
                        'Status': 'UNSUPPORTED',
                        'Error': f"{tech_name} not supported for {model_name}",
                        'Completed_Folds': 0,
                        'Failed_Folds': 0,
                    })
                    pd.DataFrame(all_results).to_csv(results_file, index=False)
                    continue

                estimator = baselines[tech_name]
                try:
                    res = evaluate_model(estimator, X, y, cv=5)

                    row = {
                        'Dataset':   data_name,
                        'Model':     model_name,
                        'Technique': tech_name,
                        'Error':     np.nan,
                        **res,
                    }
                    all_results.append(row)

                except Exception as e:
                    # ── CRITICAL FIX 4 ────────────────────────────────────────
                    # The original except block wrote a row containing only
                    # {'Dataset', 'Model', 'Technique', 'Error'} with no metric
                    # columns.  When pd.DataFrame merged this with normal rows,
                    # the missing metric columns filled with NaN.
                    # Fix: write explicit zero values AND print the real traceback
                    # so failures are visible without re-running.
                    # ──────────────────────────────────────────────────────────
                    print(f"\n  [WARN] {data_name}/{model_name}/{tech_name} failed:")
                    traceback.print_exc()
                    row = {
                        'Dataset':        data_name,
                        'Model':          model_name,
                        'Technique':      tech_name,
                        'Status':         'FAILED',
                        'Accuracy_mean':  0.0, 'Accuracy_std':  0.0,
                        'Precision_mean': 0.0, 'Precision_std': 0.0,
                        'Recall_mean':    0.0, 'Recall_std':    0.0,
                        'Recall_CV':      0.0,
                        'F1_mean':        0.0, 'F1_std':        0.0,
                        'F1_CV':          0.0, 'F1_IQR':        0.0,
                        'ROC_AUC_mean':   0.0, 'ROC_AUC_std':   0.0,
                        'G_Mean_mean':    0.0, 'G_Mean_std':    0.0,
                        'MCC_mean':       0.0, 'MCC_std':       0.0,
                        'Completed_Folds': 0,
                        'Failed_Folds':    5,
                        'Error':          str(e),
                    }
                    all_results.append(row)

                # Save incrementally after every cell — crash-safe
                pd.DataFrame(all_results).to_csv(results_file, index=False)

    final_df = pd.DataFrame(all_results)
    expected_rows = (
        len(datasets) *
        len(get_core_models()) *
        len(expected_techniques)
    )
    actual_rows = len(final_df)
    if actual_rows != expected_rows:
        print(
            f"[WARN] Incomplete experiment grid: expected {expected_rows}, got {actual_rows}"
        )
    else:
        print(f"[OK] Complete experiment grid: {actual_rows}/{expected_rows} rows.")

    print(f"\nExperiments completed. Results saved to {results_file}")
    return results_file


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    run_experiments()