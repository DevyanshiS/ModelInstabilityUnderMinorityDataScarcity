import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from imblearn.metrics import geometric_mean_score
from scipy.stats import iqr
import warnings

def evaluate_model(estimator, X, y, cv=5, random_state=42):
    """
    Evaluates an estimator using Stratified K-Fold CV.
    Returns a dictionary of performance and stability metrics.
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    
    metrics = {
        'Accuracy': [], 'Precision': [], 'Recall': [], 'F1': [], 
        'ROC_AUC': [], 'G_Mean': [], 'MCC': []
    }
    
    failed_folds = 0
    fold_errors = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for train_idx, test_idx in skf.split(X, y):
            # Check if testing set has both classes
            if len(np.unique(y[test_idx])) < 2:
                continue
                
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]
            
            # Fit clone to avoid state leakage
            try:
                from sklearn.base import clone
                model = Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler()),
                    ('clf', clone(estimator)),
                ])
                model.fit(X_train, y_train)
                
                y_pred = model.predict(X_test)
                if hasattr(model, "predict_proba"):
                    y_prob = model.predict_proba(X_test)[:, 1]
                elif hasattr(model, "decision_function"):
                    scores = model.decision_function(X_test)
                    y_prob = (scores - np.min(scores)) / (np.ptp(scores) + 1e-9)
                else:
                    y_prob = y_pred
                
                metrics['Accuracy'].append(accuracy_score(y_test, y_pred))
                metrics['Precision'].append(precision_score(y_test, y_pred, zero_division=0))
                metrics['Recall'].append(recall_score(y_test, y_pred, zero_division=0))
                metrics['F1'].append(f1_score(y_test, y_pred, zero_division=0))
                metrics['ROC_AUC'].append(roc_auc_score(y_test, y_prob))
                metrics['G_Mean'].append(geometric_mean_score(y_test, y_pred))
                metrics['MCC'].append(matthews_corrcoef(y_test, y_pred))
            except Exception as e:
                failed_folds += 1
                fold_errors.append(f"fold({len(train_idx)}/{len(test_idx)}): {type(e).__name__}: {e}")

    # Compute aggregates and stability (Coefficient of Variation = std / mean)
    if all(len(v) == 0 for v in metrics.values()):
        raise RuntimeError(
            "All CV folds failed during evaluation. "
            + (" | ".join(fold_errors[:3]) if fold_errors else "No fold error captured.")
        )

    results = {}
    for k, v in metrics.items():
        if len(v) == 0:
            results[f"{k}_mean"] = 0
            results[f"{k}_std"] = 0
            if k in ['F1', 'Recall']:
                results[f"{k}_CV"] = 0
        else:
            mean_val = np.mean(v)
            std_val = np.std(v)
            results[f"{k}_mean"] = mean_val
            results[f"{k}_std"] = std_val
            
            # Compute specific stability metrics requested in thesis
            if k in ['F1', 'Recall']:
                results[f"{k}_CV"] = std_val / mean_val if mean_val > 0 else 0
                
            if k == 'F1':
                results['F1_IQR'] = iqr(v)

    completed_folds = len(metrics['F1'])
    results['Completed_Folds'] = completed_folds
    results['Failed_Folds'] = failed_folds
    results['Status'] = 'OK' if failed_folds == 0 else 'PARTIAL_FAIL'
    if failed_folds > 0:
        results['Error'] = " | ".join(fold_errors[:3])

    return results
