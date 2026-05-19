import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from imblearn.metrics import geometric_mean_score
from xgboost import XGBClassifier

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.ensemble import BalancedBaggingClassifier, EasyEnsembleClassifier


# ==========================================
# Core Base Models
# ==========================================
def get_core_models(random_state=42):
    """
    Returns a dictionary of the 7 core base classifiers.
    Dynamically loads cuML (Linux+NVIDIA) if available, otherwise falls back
    to Scikit-Learn.

    FIX: Removed deprecated 'use_label_encoder=False' from XGBClassifier.
         This parameter was removed in XGBoost 2.0 and causes a UserWarning
         (which some environments promote to an error) while doing nothing
         useful. The equivalent behaviour is the default in XGB >= 1.6.
    """
    # 1. Try cuML for GPU Scikit-Learn Equivalents
    try:
        import cuml
        from cuml.linear_model import LogisticRegression as cuLR
        from cuml.svm import SVC as cuSVC
        from cuml.ensemble import RandomForestClassifier as cuRF
        from cuml.neighbors import KNeighborsClassifier as cuKNN
        USE_CUML = True
        print("\n✅ cuML loaded. Using NVIDIA GPU for core Scikit-Learn wrappers.")
    except ImportError:
        USE_CUML = False

    # 2. Configure XGBoost device
    # FIX: removed 'use_label_encoder': False — deprecated no-op in XGB >= 2.0.
    xgb_params = {
        'random_state': random_state,
        'eval_metric': 'logloss',
        'tree_method': 'hist',
    }

    import platform
    if USE_CUML:
        xgb_params['device'] = 'cuda'
    elif platform.system() == 'Darwin' and platform.machine() == 'arm64':
        try:
            _test = XGBClassifier(device='mps', n_estimators=1)
            _test.fit(np.zeros((2, 1)), np.array([0, 1]))
            xgb_params['device'] = 'mps'
            print("\n✅ Apple Silicon MPS detected for XGBoost.")
        except Exception:
            pass

    if USE_CUML:
        models = {
            'LR':  cuLR(max_iter=1000),
            'SVM': cuSVC(probability=True),
            'DT':  DecisionTreeClassifier(random_state=random_state),
            'RF':  cuRF(random_state=random_state),
            'XGB': XGBClassifier(**xgb_params),
            'MLP': MLPClassifier(random_state=random_state, max_iter=1000,
                                 early_stopping=True),
            'KNN': cuKNN(),
        }
    else:
        models = {
            'LR':  LogisticRegression(random_state=random_state, max_iter=1000),
            'SVM': SVC(probability=True, random_state=random_state, max_iter=2000),
            'DT':  DecisionTreeClassifier(random_state=random_state),
            'RF':  RandomForestClassifier(random_state=random_state),
            'XGB': XGBClassifier(**xgb_params),
            'MLP': MLPClassifier(random_state=random_state, max_iter=1000,
                                 early_stopping=True),
            'KNN': KNeighborsClassifier(),
        }

    return models


# ==========================================
# Threshold Moving Wrapper
# ==========================================
class ThresholdMovingClassifier(BaseEstimator, ClassifierMixin):
    """
    Trains a base classifier and tunes the decision threshold using OOF
    cross-validation to maximise G-Mean.
    """
    def __init__(self, estimator, cv=3, random_state=None):
        self.estimator = estimator
        self.cv = cv
        self.random_state = random_state
        self.optimal_threshold_ = 0.5
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        skf = StratifiedKFold(n_splits=self.cv, shuffle=True,
                              random_state=self.random_state)

        oof_probs = np.zeros(len(y))
        for train_idx, val_idx in skf.split(X, y):
            clone_est = clone(self.estimator)
            clone_est.fit(X[train_idx], y[train_idx])
            oof_probs[val_idx] = clone_est.predict_proba(X[val_idx])[:, 1]

        best_gmean, best_thresh = -1.0, 0.5
        for t in np.linspace(0.1, 0.9, 50):
            y_pred_t = (oof_probs >= t).astype(int)
            g = geometric_mean_score(y, y_pred_t)
            if g > best_gmean:
                best_gmean, best_thresh = g, t

        self.optimal_threshold_ = best_thresh
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y)
        return self

    def predict(self, X):
        probs = self.estimator_.predict_proba(X)[:, 1]
        return (probs >= self.optimal_threshold_).astype(int)

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)


# ==========================================
# XGBoost ClassWeight Wrapper
# ==========================================
class XGBClassWeightWrapper(BaseEstimator, ClassifierMixin):
    """
    Wraps XGBClassifier to support balanced class weighting via
    scale_pos_weight, which is computed from the training data at fit time.

    FIX: XGBClassifier does not have a 'class_weight' parameter — it uses
    'scale_pos_weight' instead. The original code detected this but then
    did nothing, leaving ClassWeight missing from the XGB baselines dict.
    This wrapper computes scale_pos_weight = N_majority / N_minority at fit
    time so it works identically to class_weight='balanced' on other models.
    """
    def __init__(self, base_xgb_estimator):
        self.base_xgb_estimator = base_xgb_estimator

    def fit(self, X, y):
        counts = np.bincount(y.astype(int))
        # Assumes binary: minority=1, majority=0
        min_cls = int(np.argmin(counts))
        maj_cls = int(1 - min_cls)
        scale = counts[maj_cls] / (counts[min_cls] + 1e-9)

        self.estimator_ = clone(self.base_xgb_estimator)
        self.estimator_.set_params(scale_pos_weight=scale)
        self.estimator_.fit(X, y)
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return self.estimator_.predict(X)

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)


# ==========================================
# Baseline Techniques Setup
# ==========================================
def create_baseline_pipelines(model_name, base_estimator, random_state=42):
    """
    Given a base estimator, creates a dictionary of the 9 baseline techniques
    plus the unmodified 'Base' baseline.

    FIX 1 — ClassWeight for XGB: The original code hit the
        `elif hasattr(cw_clf, 'scale_pos_weight')` branch and then silently
        did nothing, so 'ClassWeight' was absent from XGB's baselines dict.
        Now uses XGBClassWeightWrapper which computes scale_pos_weight at
        fit time from the actual training labels.

    FIX 2 — EasyEnsemble for all models: The original code only created
        EasyEnsembleClassifier for 'LR' and 'DT'. All other models (XGB, RF,
        SVM, MLP, KNN) were missing 'EasyEnsemble' from their baselines dict.
        The restriction has been removed — EasyEnsemble is now built for every
        model. EasyEnsembleClassifier wraps any base estimator; limiting it to
        LR/DT was unnecessary.
    """
    baselines = {}

    # 0. Base — no modification
    baselines['Base'] = clone(base_estimator)

    # 1. Random Oversampling
    baselines['ROS'] = ImbPipeline([
        ('sampler', RandomOverSampler(random_state=random_state)),
        ('clf', clone(base_estimator)),
    ])

    # 2. Random Undersampling
    baselines['RUS'] = ImbPipeline([
        ('sampler', RandomUnderSampler(random_state=random_state)),
        ('clf', clone(base_estimator)),
    ])

    # 3. SMOTE
    baselines['SMOTE'] = ImbPipeline([
        ('sampler', SMOTE(random_state=random_state)),
        ('clf', clone(base_estimator)),
    ])

    # 4. ADASYN
    baselines['ADASYN'] = ImbPipeline([
        ('sampler', ADASYN(random_state=random_state)),
        ('clf', clone(base_estimator)),
    ])

    # 5. Tomek Links
    baselines['TomekLinks'] = ImbPipeline([
        ('sampler', TomekLinks()),
        ('clf', clone(base_estimator)),
    ])

    # 6. Class Weighting
    # FIX 1: XGBClassifier has no 'class_weight' param; use the wrapper instead.
    cw_clf = clone(base_estimator)
    if model_name == 'XGB' or hasattr(cw_clf, 'scale_pos_weight'):
        # XGBClassWeightWrapper computes scale_pos_weight at fit time
        baselines['ClassWeight'] = XGBClassWeightWrapper(clone(base_estimator))
    elif model_name == 'KNN':
        # KNN has no native class weighting mechanism; skip gracefully
        pass
    elif hasattr(cw_clf, 'class_weight'):
        cw_clf.set_params(class_weight='balanced')
        baselines['ClassWeight'] = cw_clf

    # 7. Balanced Bagging
    baselines['BalancedBagging'] = BalancedBaggingClassifier(
        estimator=clone(base_estimator),
        random_state=random_state,
    )

    # 8. EasyEnsemble
    # FIX 2: Removed the 'if model_name in ["LR", "DT"]' guard.
    # EasyEnsembleClassifier works with any sklearn-compatible base estimator.
    # The original guard caused 'EasyEnsemble' to be absent from the baselines
    # dict for XGB, RF, SVM, MLP, and KNN, making the runner write 0-filled rows.
    baselines['EasyEnsemble'] = EasyEnsembleClassifier(
        estimator=clone(base_estimator),
        random_state=random_state,
        n_jobs=-1,
    )

    # 9. Threshold Moving
    baselines['ThresholdMoving'] = ThresholdMovingClassifier(
        estimator=clone(base_estimator),
        random_state=random_state,
    )

    return baselines


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')

    print("Testing get_core_models()...")
    core = get_core_models()
    print(f"  Models: {list(core.keys())}")

    print("\nTesting create_baseline_pipelines() for each model...")
    for name, est in core.items():
        pipes = create_baseline_pipelines(name, est)
        print(f"  {name}: {sorted(pipes.keys())}")

    # Smoke-test every pipeline on a tiny imbalanced dataset
    from sklearn.datasets import make_classification
    from sklearn.preprocessing import StandardScaler
    import numpy as np

    X, y = make_classification(n_samples=300, weights=[0.85, 0.15],
                                n_features=10, random_state=0)
    X = StandardScaler().fit_transform(X)

    print("\nSmoke-testing all pipelines (fit + predict)...")
    all_pass = True
    for model_name, base_est in core.items():
        pipes = create_baseline_pipelines(model_name, base_est)
        for tech_name, pipeline in pipes.items():
            try:
                p = clone(pipeline)
                p.fit(X[:240], y[:240])
                preds = p.predict(X[240:])
                assert len(preds) == 60
            except Exception as e:
                print(f"  FAIL  {model_name}/{tech_name}: {type(e).__name__}: {e}")
                all_pass = False

    if all_pass:
        print("  All pipelines passed.")
    else:
        print("  Some pipelines failed — see above.")