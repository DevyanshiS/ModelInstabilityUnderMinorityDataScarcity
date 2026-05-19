import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import label_binarize
from imblearn.over_sampling import SMOTE


class AMBAClassifier(BaseEstimator, ClassifierMixin):
    """
    Adaptive Minority Boundary Augmentation (AMBA) Framework.

    A scikit-learn compatible classifier that addresses class imbalance through:
      - Component 0: Adaptive pre-flight routing based on IR and baseline AUC
      - Stage 1:     Adaptive boundary detection using an IR-conditioned f0
      - Stage 2:     Budget-capped, gradient-directed synthetic oversampling
                     with a diversity filter
      - Stage 3:     Stability-weighted ensemble aggregation

    Parameters
    ----------
    base_estimator : sklearn estimator
        The base classifier to clone for each ensemble member.
    tau_ir : float, default=5.0
        Imbalance-ratio threshold below which the pipeline bypasses synthesis
        and applies ThresholdMoving only.
    tau_f0 : float, default=20.0
        IR threshold for selecting f0: LR when IR < tau_f0, shallow RF otherwise.
    tau_b : float, default=0.6
        Boundary proximity threshold.  Minority samples with bi >= tau_b
        (where bi = 1 - 2|P(y=1|xi) - 0.5|) are treated as boundary samples.
    theta_auc : float, default=0.97
        Baseline AUC threshold above which Stage 1 & 2 are skipped (classes
        are already well-separated; only Stage 3 runs).
    alpha : float, default=0.3
        Gradient direction weight in boundary-directed synthesis.
    rho : float, default=0.7
        Fraction of the synthesis budget allocated to boundary-region samples.
    gamma : float, default=3.0
        Scarcity cap multiplier.  Total synthetic samples <= gamma * |D+|.
    n_clones : int, default=10
        Number of base classifiers in the stability-weighted ensemble (K).
    v_folds : int, default=5
        Number of folds used to estimate per-clone recall instability (V).
    k_nn : int, default=5
        Number of nearest neighbours used during SMOTE-style synthesis.
    delta_factor : float, default=0.01
        Diversity filter: a synthetic sample is rejected if its minimum
        distance to any real boundary sample is < delta_factor * mean_std(X+).
    random_state : int, default=42
        Random seed for reproducibility.
    """

    def __init__(
        self,
        base_estimator,
        tau_ir=5.0,
        tau_f0=20.0,
        tau_b=0.6,
        theta_auc=0.97,
        alpha=0.3,
        rho=0.7,
        gamma=3.0,
        n_clones=10,
        v_folds=5,
        k_nn=5,
        delta_factor=0.01,
        random_state=42,
    ):
        self.base_estimator = base_estimator
        self.tau_ir = tau_ir
        self.tau_f0 = tau_f0
        self.tau_b = tau_b
        self.theta_auc = theta_auc
        self.alpha = alpha
        self.rho = rho
        self.gamma = gamma
        self.n_clones = n_clones
        self.v_folds = v_folds
        self.k_nn = k_nn
        self.delta_factor = delta_factor
        self.random_state = random_state

        # set during fit
        self.ensemble_ = []
        self.weights_ = []
        self.tau_star_ = 0.5
        self.classes_ = None
        self.min_class_ = None
        self.maj_class_ = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X, y):
        """Fit AMBA on the imbalanced training data."""
        np.random.seed(self.random_state)

        self.classes_ = np.unique(y)
        class_counts = np.bincount(y.astype(int))
        self.min_class_ = int(np.argmin(class_counts))
        self.maj_class_ = int(1 - self.min_class_)

        N_min = class_counts[self.min_class_]
        N_maj = class_counts[self.maj_class_]
        IR = N_maj / N_min

        # ---- COMPONENT 0: ADAPTIVE PRE-FLIGHT ROUTING ----------------

        # Route A: mild imbalance — ThresholdMoving only
        if IR < self.tau_ir:
            self._fit_threshold_moving(X, y)
            return self

        # Route B: check pilot AUC — if classes already well-separated,
        #          skip synthesis and run Stage 3 on class-weighted data
        baseline_auc = self._pilot_auc(X, y)
        if baseline_auc >= self.theta_auc:
            X_aug, y_aug = X, y
            self._fit_ensemble(X_aug, y_aug)
            self._calibrate_threshold(X_aug, y_aug)
            return self

        # ---- STAGE 1: ADAPTIVE BOUNDARY DETECTION --------------------

        X_min = X[y == self.min_class_]
        X_maj = X[y == self.maj_class_]

        # Select f0 based on IR
        if IR >= self.tau_f0:
            f0 = RandomForestClassifier(
                n_estimators=50,
                max_depth=4,
                random_state=self.random_state,
            )
        else:
            f0 = LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                class_weight="balanced",
            )
        f0.fit(X, y)

        # Minority-class probabilities from f0
        base_probs = f0.predict_proba(X_min)[:, self.min_class_]

        # Boundary proximity score: bi in [0, 1]; 1 = on the boundary
        bi = 1.0 - 2.0 * np.abs(base_probs - 0.5)
        boundary_idx = np.where(bi >= self.tau_b)[0]
        safe_idx = np.where(bi < self.tau_b)[0]

        # Finite-difference gradient of P(min_class | x) w.r.t. x
        gradients = self._compute_gradients(f0, X_min)

        # ---- STAGE 2: BUDGET-CAPPED GUIDED SYNTHESIS -----------------

        X_aug, y_aug = self._synthesise(
            X, y, X_min, X_maj, boundary_idx, safe_idx, gradients, N_min, N_maj
        )

        # ---- STAGE 3: STABILITY-WEIGHTED ENSEMBLE --------------------

        self._fit_ensemble(X_aug, y_aug)
        self._calibrate_threshold(X_aug, y_aug)
        return self

    def predict_proba(self, X):
        """Return weighted-average class probabilities."""
        avg = np.zeros((X.shape[0], len(self.classes_)))
        for clf, w in zip(self.ensemble_, self.weights_):
            avg += w * clf.predict_proba(X)
        return avg

    def predict(self, X):
        """Predict class labels using the calibrated decision threshold."""
        probs = self.predict_proba(X)[:, self.min_class_]
        return np.where(probs >= self.tau_star_, self.min_class_, self.maj_class_)

    # ------------------------------------------------------------------
    # Component 0 helpers
    # ------------------------------------------------------------------

    def _pilot_auc(self, X, y):
        """3-fold cross-validated baseline AUC on the raw data."""
        skf = StratifiedKFold(
            n_splits=3, shuffle=True, random_state=self.random_state
        )
        scores = []
        for train_idx, val_idx in skf.split(X, y):
            clf = clone(self.base_estimator)
            try:
                clf.fit(X[train_idx], y[train_idx])
                preds = clf.predict_proba(X[val_idx])[:, self.min_class_]
                if len(np.unique(y[val_idx])) > 1:
                    scores.append(roc_auc_score(y[val_idx], preds))
            except Exception:
                scores.append(0.0)
        return float(np.mean(scores)) if scores else 0.0

    def _fit_threshold_moving(self, X, y):
        """
        Route A: mild imbalance — fit a single cross-validated classifier
        and calibrate tau* to maximise G-Mean on a validation fold.
        """
        skf = StratifiedKFold(
            n_splits=self.v_folds, shuffle=True, random_state=self.random_state
        )
        best_tau, best_gmean = 0.5, -1.0
        thresholds = np.arange(0.1, 0.91, 0.05)

        for tau in thresholds:
            fold_gmeans = []
            for train_idx, val_idx in skf.split(X, y):
                clf = clone(self.base_estimator)
                try:
                    clf.fit(X[train_idx], y[train_idx])
                    probs = clf.predict_proba(X[val_idx])[:, self.min_class_]
                    preds = (probs >= tau).astype(int)
                    if len(np.unique(y[val_idx])) < 2:
                        continue
                    tn, fp, fn, tp = confusion_matrix(
                        y[val_idx], preds, labels=[self.maj_class_, self.min_class_]
                    ).ravel()
                    sens = tp / (tp + fn + 1e-9)
                    spec = tn / (tn + fp + 1e-9)
                    fold_gmeans.append(np.sqrt(sens * spec))
                except Exception:
                    continue
            if fold_gmeans and np.mean(fold_gmeans) > best_gmean:
                best_gmean = np.mean(fold_gmeans)
                best_tau = tau

        self.tau_star_ = best_tau
        final_clf = clone(self.base_estimator)
        final_clf.fit(X, y)
        self.ensemble_ = [final_clf]
        self.weights_ = [1.0]

    # ------------------------------------------------------------------
    # Stage 1 helper
    # ------------------------------------------------------------------

    def _compute_gradients(self, f0, X_min):
        """
        Finite-difference estimate of d P(min_class|x)/dx for each x in X_min.
        Uses epsilon = 1e-3 * per-feature standard deviation (as per the paper).
        """
        n_samples, n_features = X_min.shape
        gradients = np.zeros((n_samples, n_features))
        feature_stds = np.std(X_min, axis=0) + 1e-8
        base_probs = f0.predict_proba(X_min)[:, self.min_class_]

        for j in range(n_features):
            eps = 1e-3 * feature_stds[j]
            X_pert = X_min.copy()
            X_pert[:, j] += eps
            pert_probs = f0.predict_proba(X_pert)[:, self.min_class_]
            gradients[:, j] = (pert_probs - base_probs) / eps

        return gradients

    # ------------------------------------------------------------------
    # Stage 2 helper
    # ------------------------------------------------------------------

    def _synthesise(self, X, y, X_min, X_maj, boundary_idx, safe_idx,
                    gradients, N_min, N_maj):
        """
        Budget-capped, gradient-directed synthetic oversampling.

        Returns augmented (X_aug, y_aug).
        """
        # Scarcity-proportional budget cap (equation from paper)
        n_needed = N_maj - N_min
        n_max_syn = min(n_needed, int(np.floor(self.gamma * N_min)))
        if n_max_syn <= 0:
            return X, y

        n_bnd = int(np.floor(self.rho * n_max_syn))
        n_safe = n_max_syn - n_bnd

        # Diversity filter threshold: delta = delta_factor * mean feature std
        delta = self.delta_factor * float(np.mean(np.std(X_min, axis=0) + 1e-8))

        S_bnd = self._boundary_synthesis(
            X_min, boundary_idx, gradients, n_bnd, delta
        )
        S_safe = self._safe_synthesis(X_min, safe_idx, n_safe)

        synthetic_X = []
        synthetic_y = []
        for sx in (S_bnd + S_safe):
            synthetic_X.append(sx)
            synthetic_y.append(self.min_class_)

        if not synthetic_X:
            return X, y

        X_syn = np.vstack(synthetic_X)
        y_syn = np.array(synthetic_y, dtype=y.dtype)
        X_aug = np.vstack([X, X_syn])
        y_aug = np.concatenate([y, y_syn])
        return X_aug, y_aug

    def _boundary_synthesis(self, X_min, boundary_idx, gradients, n_bnd, delta):
        """
        Generate n_bnd synthetic samples from D+_bnd using gradient-directed
        interpolation.  Rejects samples that violate the diversity filter.

        x_syn = x_i + lambda1*(x_nn - x_i) + alpha*lambda2*g_hat_i
        """
        if len(boundary_idx) == 0 or n_bnd == 0:
            return []

        X_bnd = X_min[boundary_idx]

        # Fit a k-NN on X_min so we can interpolate within the minority cloud
        k = min(self.k_nn, len(X_min) - 1)
        if k < 1:
            return []
        nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X_min)
        nn_distances, nn_indices = nbrs.kneighbors(X_bnd)
        # nn_indices[:,0] is the sample itself; neighbours start at index 1

        synthetic = []
        max_attempts = n_bnd * 15  # generous retry budget for diversity filter
        attempt = 0

        while len(synthetic) < n_bnd and attempt < max_attempts:
            attempt += 1
            # Pick a random boundary sample
            local_i = np.random.randint(len(boundary_idx))
            global_i = boundary_idx[local_i]

            # Pick a random k-NN (excluding self) from the full minority set
            neighbour_global = nn_indices[local_i, 1:]  # shape (k,)
            xnn = X_min[np.random.choice(neighbour_global)]

            # Gradient direction
            grad = gradients[global_i]
            norm = np.linalg.norm(grad)
            if norm > 1e-12:
                g_hat = grad / norm
            else:
                g_hat = np.random.randn(X_min.shape[1])
                g_hat /= np.linalg.norm(g_hat)

            lam1 = np.random.uniform(0.0, 1.0)
            lam2 = np.random.uniform(0.0, 1.0)

            # Boundary-directed interpolation (correct direction: toward boundary)
            x_i = X_min[global_i]
            x_syn = x_i + lam1 * (xnn - x_i) + self.alpha * lam2 * g_hat

            # Diversity filter: reject near-duplicates
            if X_bnd.shape[0] > 0:
                dists = np.linalg.norm(X_bnd - x_syn, axis=1)
                if np.min(dists) < delta:
                    continue

            synthetic.append(x_syn)

        return synthetic

    def _safe_synthesis(self, X_min, safe_idx, n_safe):
        """
        Generate n_safe synthetic samples from D+_safe via standard linear
        interpolation (SMOTE-style) between two random safe-region samples.
        """
        if len(safe_idx) < 2 or n_safe == 0:
            return []

        X_safe = X_min[safe_idx]
        synthetic = []
        for _ in range(n_safe):
            i1, i2 = np.random.choice(len(X_safe), size=2, replace=False)
            lam = np.random.uniform(0.0, 1.0)
            x_syn = X_safe[i1] + lam * (X_safe[i2] - X_safe[i1])
            synthetic.append(x_syn)
        return synthetic

    # ------------------------------------------------------------------
    # Stage 3 helpers
    # ------------------------------------------------------------------

    def _fit_ensemble(self, X, y):
        """
        Train n_clones bootstrap classifiers and assign inverse-instability
        weights based on the standard deviation of minority recall across
        v_folds cross-validation folds (as per Section 7.5 of the paper).

        Weight formula:  w_tilde_k = 1 / (1 + sigma_k)
                         w_k       = w_tilde_k / sum(w_tilde)
        """
        self.ensemble_ = []
        raw_weights = []

        skf = StratifiedKFold(
            n_splits=self.v_folds, shuffle=True, random_state=self.random_state
        )

        for k in range(self.n_clones):
            # Bootstrap sample
            boot_idx = np.random.choice(len(X), size=len(X), replace=True)
            X_boot, y_boot = X[boot_idx], y[boot_idx]

            clf = clone(self.base_estimator)
            clf.fit(X_boot, y_boot)
            self.ensemble_.append(clf)

            # V-fold minority recall on the full augmented set
            recall_scores = []
            for _, val_idx in skf.split(X, y):
                y_val = y[val_idx]
                minority_mask = y_val == self.min_class_
                if minority_mask.sum() == 0:
                    continue
                preds_val = clf.predict(X[val_idx])
                tp = np.sum(preds_val[minority_mask] == self.min_class_)
                recall_k = tp / minority_mask.sum()
                recall_scores.append(recall_k)

            if len(recall_scores) >= 2:
                sigma_k = float(np.std(recall_scores, ddof=1))
            else:
                sigma_k = 1.0  # pessimistic default when folds are too small

            raw_weights.append(1.0 / (1.0 + sigma_k))

        # Normalise
        total = sum(raw_weights)
        if total > 0:
            self.weights_ = [w / total for w in raw_weights]
        else:
            self.weights_ = [1.0 / self.n_clones] * self.n_clones

    def _calibrate_threshold(self, X, y):
        """
        Find tau* = argmax_tau G-Mean(tau) on a held-out validation fold.
        Uses the final ensemble's predict_proba for calibration.
        """
        skf = StratifiedKFold(
            n_splits=self.v_folds, shuffle=True, random_state=self.random_state
        )
        thresholds = np.arange(0.05, 0.96, 0.05)
        best_tau, best_gmean = 0.5, -1.0

        for tau in thresholds:
            fold_gmeans = []
            for _, val_idx in skf.split(X, y):
                probs = self.predict_proba(X[val_idx])[:, self.min_class_]
                preds = np.where(probs >= tau, self.min_class_, self.maj_class_)
                y_val = y[val_idx]
                if len(np.unique(y_val)) < 2:
                    continue
                try:
                    tn, fp, fn, tp = confusion_matrix(
                        y_val, preds,
                        labels=[self.maj_class_, self.min_class_]
                    ).ravel()
                    sens = tp / (tp + fn + 1e-9)
                    spec = tn / (tn + fp + 1e-9)
                    fold_gmeans.append(np.sqrt(sens * spec))
                except ValueError:
                    continue
            if fold_gmeans and np.mean(fold_gmeans) > best_gmean:
                best_gmean = np.mean(fold_gmeans)
                best_tau = tau

        self.tau_star_ = best_tau


# ======================================================================
# Quick smoke test
# ======================================================================

if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.metrics import f1_score, roc_auc_score
    from sklearn.preprocessing import StandardScaler

    print("=" * 60)
    print("AMBA smoke test")
    print("=" * 60)

    # --- Moderate imbalance (IR ≈ 19) ---
    X, y = make_classification(
        n_samples=2000,
        n_features=20,
        n_informative=10,
        weights=[0.95, 0.05],
        random_state=0,
    )
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    base_rf = RandomForestClassifier(n_estimators=50, random_state=42)
    amba = AMBAClassifier(base_estimator=base_rf, random_state=42)

    print("\nFitting AMBA (moderate IR ≈ 19)...")
    amba.fit(X, y)
    preds = amba.predict(X)
    probs = amba.predict_proba(X)[:, 1]

    print(f"  tau* calibrated to : {amba.tau_star_:.2f}")
    print(f"  Ensemble size      : {len(amba.ensemble_)}")
    print(f"  F1  (train)        : {f1_score(y, preds):.4f}")
    print(f"  AUC (train)        : {roc_auc_score(y, probs):.4f}")
    print(f"  Predict proba shape: {probs.shape}")

    # --- Mild imbalance (ThresholdMoving route) ---
    X2, y2 = make_classification(
        n_samples=500,
        n_features=10,
        weights=[0.6, 0.4],
        random_state=1,
    )
    X2 = scaler.fit_transform(X2)
    amba2 = AMBAClassifier(base_estimator=base_rf, random_state=42)
    print("\nFitting AMBA (mild IR, ThresholdMoving route)...")
    amba2.fit(X2, y2)
    print(f"  tau* calibrated to : {amba2.tau_star_:.2f}")
    print(f"  Ensemble size      : {len(amba2.ensemble_)}")

    # --- Extreme imbalance (IR ≈ 99) ---
    X3, y3 = make_classification(
        n_samples=3000,
        n_features=15,
        n_informative=8,
        weights=[0.99, 0.01],
        random_state=2,
    )
    X3 = scaler.fit_transform(X3)
    amba3 = AMBAClassifier(
        base_estimator=RandomForestClassifier(n_estimators=30, random_state=42),
        random_state=42,
    )
    print("\nFitting AMBA (extreme IR ≈ 99)...")
    amba3.fit(X3, y3)
    preds3 = amba3.predict(X3)
    print(f"  tau* calibrated to : {amba3.tau_star_:.2f}")
    print(f"  F1  (train)        : {f1_score(y3, preds3):.4f}")

    print("\nAll smoke tests passed.")