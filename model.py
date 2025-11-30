import pandas as pd
import numpy as np
import math
import random
import pickle
import os
import sys
from collections import Counter, defaultdict
from itertools import product
from copy import deepcopy

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

RNG = np.random.RandomState(42)
random.seed(42)


BEST_MODEL_PARAMS = {
    "model_type": None, 
    "parameters": None, 
    "performance": None, 
    "feature_names": None,
    "grid_search_completed": False
}

# Pre-defined parameter grids for quick reference
PARAM_GRIDS = {
    "RandomForest": {
        "n_estimators": [10, 20],
        "max_depth": [4, 6],
        "min_samples_split": [2, 5],
        "max_features": ['sqrt']
    },
    "GradientBoosting": {
        "n_estimators": [20, 50],
        "learning_rate": [0.05, 0.1],
        "max_depth": [2, 3],
        "min_samples_split": [2]
    }
}

def get_best_params():
    """Returns the best parameters found from grid search"""
    return BEST_MODEL_PARAMS

def create_model_with_best_params():
    """Creates a model instance with the stored best parameters"""
    if not BEST_MODEL_PARAMS["grid_search_completed"]:
        raise ValueError("Grid search not completed. Run main() first or set BEST_MODEL_PARAMS manually.")
    
    if BEST_MODEL_PARAMS["model_type"] == "GradientBoostingClassifierManual":
        return GradientBoostingClassifierManual(**BEST_MODEL_PARAMS["parameters"])
    elif BEST_MODEL_PARAMS["model_type"] == "RandomForestManual":
        return RandomForestManual(**BEST_MODEL_PARAMS["parameters"])
    else:
        raise ValueError("Unknown model type in BEST_MODEL_PARAMS")

def print_best_params():
    """Prints the current best parameters"""
    if not BEST_MODEL_PARAMS["grid_search_completed"]:
        print("Grid search not completed yet.")
        return
    print("=== BEST MODEL PARAMETERS ===")
    print(f"Model Type: {BEST_MODEL_PARAMS['model_type']}")
    print("Parameters:")
    for key, value in BEST_MODEL_PARAMS["parameters"].items():
        print(f"  {key}: {value}")
    if BEST_MODEL_PARAMS["performance"]:
        print("Test Performance:")
        for metric, score in BEST_MODEL_PARAMS["performance"].items():
            print(f"  {metric}: {score:.4f}")

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def accuracy_score(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return (y_true == y_pred).mean()

def precision_score_manual(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = np.sum((y_true==1) & (y_pred==1))
    fp = np.sum((y_true==0) & (y_pred==1))
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0

def recall_score_manual(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = np.sum((y_true==1) & (y_pred==1))
    fn = np.sum((y_true==1) & (y_pred==0))
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0

def f1_score_manual(y_true, y_pred):
    p = precision_score_manual(y_true, y_pred)
    r = recall_score_manual(y_true, y_pred)
    return 2*p*r / (p + r) if (p + r) > 0 else 0.0

def roc_auc_score_manual(y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    pos = y_true == 1
    neg = y_true == 0
    n_pos = pos.sum()
    n_neg = neg.sum()
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1)
    sum_pos_ranks = ranks[pos].sum()
    auc = (sum_pos_ranks - n_pos*(n_pos+1)/2) / (n_pos * n_neg)
    return auc

def stratified_train_test_split(X, y, test_size=0.2, random_state=42):
    np.random.seed(random_state)
    X = np.asarray(X)
    y = np.asarray(y)
    indices = np.arange(len(y))
    classes = np.unique(y)
    train_idx = []
    test_idx = []
    for c in classes:
        c_idx = indices[y == c]
        n_test = max(1, int(len(c_idx) * test_size))
        np.random.shuffle(c_idx)
        test_idx.extend(c_idx[:n_test].tolist())
        train_idx.extend(c_idx[n_test:].tolist())
    np.random.shuffle(train_idx)
    np.random.shuffle(test_idx)
    return np.array(train_idx), np.array(test_idx)

def stratified_kfold_indices(y, n_splits=5, random_state=42):
    np.random.seed(random_state)
    y = np.asarray(y)
    classes = np.unique(y)
    folds = [[] for _ in range(n_splits)]
    for c in classes:
        idx = np.where(y == c)[0].tolist()
        np.random.shuffle(idx)
        for i, ind in enumerate(idx):
            folds[i % n_splits].append(ind)
    folds = [np.array(f, dtype=int) for f in folds]
    return folds

class LogisticRegressionGD:
    def __init__(self, lr=0.1, n_iter=2000, l2=0.0, verbose=False):
        self.lr = lr
        self.n_iter = n_iter
        self.l2 = l2
        self.verbose = verbose
        self.w = None
        self.b = 0.0

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0.0
        for it in range(self.n_iter):
            z = X.dot(self.w) + self.b
            p = sigmoid(z)
            error = p - y
            grad_w = (X.T.dot(error)) / n_samples + self.l2 * self.w
            grad_b = error.mean()
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b
            if self.verbose and it % 500 == 0:
                loss = -np.mean(y*np.log(np.clip(p,1e-12,1)) + (1-y)*np.log(np.clip(1-p,1e-12,1)))
                print(f"[LR] iter {it}/{self.n_iter}, loss={loss:.5f}")
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        z = X.dot(self.w) + self.b
        return sigmoid(z)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

class KNNClassifier:
    def __init__(self, k=5):
        self.k = k
        self.X = None
        self.y = None

    def fit(self, X, y):
        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(y, dtype=int)

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        probs = []
        for x in X:
            dists = np.linalg.norm(self.X - x, axis=1)
            idx = np.argsort(dists)[:self.k]
            votes = self.y[idx]
            prob = votes.mean() 
            probs.append(prob)
        return np.array(probs)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

class DecisionTreeClassifierManual:
    def __init__(self, max_depth=5, min_samples_split=2, max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.tree = None

    def _gini(self, y):
        if len(y) == 0:
            return 0
        p1 = np.mean(y == 1)
        p0 = 1 - p1
        return 1.0 - p0**2 - p1**2

    def _best_split(self, X, y):
        n_samples, n_features = X.shape
        if self.max_features is None:
            features = range(n_features)
        else:
            if isinstance(self.max_features, float):
                k = max(1, int(self.max_features * n_features))
            else:
                k = min(n_features, int(self.max_features))
            features = np.random.choice(n_features, k, replace=False)
        best = {"impurity": self._gini(y), "feature": None, "threshold": None, "left_idx": None, "right_idx": None}
        base_impurity = best["impurity"]
        for feat in features:
            vals = np.unique(X[:, feat])
            if len(vals) <= 1:
                continue
            thresholds = (vals[:-1] + vals[1:]) / 2.0
            for thr in thresholds:
                left_mask = X[:, feat] <= thr
                right_mask = ~left_mask
                if left_mask.sum() < self.min_samples_split or right_mask.sum() < self.min_samples_split:
                    continue
                g_left = self._gini(y[left_mask])
                g_right = self._gini(y[right_mask])
                w = (left_mask.sum() / n_samples) * g_left + (right_mask.sum() / n_samples) * g_right
                if w < best["impurity"]:
                    best = {"impurity": w, "feature": feat, "threshold": thr,
                            "left_idx": np.where(left_mask)[0], "right_idx": np.where(right_mask)[0]}
        if best["feature"] is None:
            return None
        return best

    def _build_tree(self, X, y, depth=0):
        node = {}
        num_pos = int(np.sum(y == 1))
        num_neg = int(np.sum(y == 0))
        node["n_samples"] = len(y)
        node["num_pos"] = num_pos
        node["num_neg"] = num_neg
        node["prediction"] = 1 if num_pos >= num_neg else 0
        node["proba"] = num_pos / (num_pos + num_neg) if (num_pos + num_neg) > 0 else 0.0

        if depth >= self.max_depth or len(y) < self.min_samples_split or len(np.unique(y)) == 1:
            node["is_leaf"] = True
            return node

        split = self._best_split(X, y)
        if split is None:
            node["is_leaf"] = True
            return node

        node["is_leaf"] = False
        node["feature"] = split["feature"]
        node["threshold"] = split["threshold"]
        left_idx = split["left_idx"]
        right_idx = split["right_idx"]

        node["left"] = self._build_tree(X[left_idx], y[left_idx], depth+1)
        node["right"] = self._build_tree(X[right_idx], y[right_idx], depth+1)
        return node

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        self.n_features_in_ = X.shape[1]
        self.tree = self._build_tree(X, y, depth=0)
        return self

    def _predict_one_proba(self, x, node):
        if node["is_leaf"]:
            return node["proba"]
        feat = node["feature"]
        thr = node["threshold"]
        if x[feat] <= thr:
            return self._predict_one_proba(x, node["left"])
        else:
            return self._predict_one_proba(x, node["right"])

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_one_proba(x, self.tree) for x in X])

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

class DecisionTreeRegressorManual:
    def __init__(self, max_depth=3, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def _mse(self, y):
        if len(y) == 0:
            return 0
        return np.mean((y - y.mean())**2)

    def _best_split(self, X, y):
        n_samples, n_features = X.shape
        best = {"mse": self._mse(y), "feature": None, "threshold": None, "left_idx": None, "right_idx": None}
        for feat in range(n_features):
            vals = np.unique(X[:, feat])
            if len(vals) <= 1:
                continue
            thresholds = (vals[:-1] + vals[1:]) / 2.0
            for thr in thresholds:
                left_mask = X[:, feat] <= thr
                right_mask = ~left_mask
                if left_mask.sum() < self.min_samples_split or right_mask.sum() < self.min_samples_split:
                    continue
                mse_left = self._mse(y[left_mask])
                mse_right = self._mse(y[right_mask])
                weighted = (left_mask.sum()*mse_left + right_mask.sum()*mse_right) / n_samples
                if weighted < best["mse"]:
                    best = {"mse": weighted, "feature": feat, "threshold": thr,
                            "left_idx": np.where(left_mask)[0], "right_idx": np.where(right_mask)[0]}
        if best["feature"] is None:
            return None
        return best

    def _build_tree(self, X, y, depth=0):
        node = {}
        node["n_samples"] = len(y)
        node["value"] = float(np.mean(y)) if len(y) > 0 else 0.0
        if depth >= self.max_depth or len(y) < self.min_samples_split:
            node["is_leaf"] = True
            return node
        split = self._best_split(X, y)
        if split is None:
            node["is_leaf"] = True
            return node
        node["is_leaf"] = False
        node["feature"] = split["feature"]
        node["threshold"] = split["threshold"]
        left_idx = split["left_idx"]
        right_idx = split["right_idx"]
        node["left"] = self._build_tree(X[left_idx], y[left_idx], depth+1)
        node["right"] = self._build_tree(X[right_idx], y[right_idx], depth+1)
        return node

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.tree = self._build_tree(X, y, depth=0)
        return self

    def _predict_one(self, x, node):
        if node["is_leaf"]:
            return node["value"]
        feat = node["feature"]
        thr = node["threshold"]
        if x[feat] <= thr:
            return self._predict_one(x, node["left"])
        else:
            return self._predict_one(x, node["right"])

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_one(x, self.tree) for x in X])

class RandomForestManual:
    def __init__(self, n_estimators=10, max_depth=6, min_samples_split=2, max_features='sqrt', bootstrap=True):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.trees = []
        self.features_indices = []

    def _get_max_features(self, n_features):
        if self.max_features == 'sqrt':
            return max(1, int(math.sqrt(n_features)))
        elif self.max_features == 'log2':
            return max(1, int(math.log2(n_features)))
        elif isinstance(self.max_features, float):
            return max(1, int(self.max_features * n_features))
        elif isinstance(self.max_features, int):
            return min(n_features, self.max_features)
        else:
            return n_features

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        n_samples, n_features = X.shape
        self.trees = []
        self.features_indices = []
        for i in range(self.n_estimators):
            if self.bootstrap:
                idxs = np.random.choice(n_samples, n_samples, replace=True)
            else:
                idxs = np.arange(n_samples)
            max_feat = self._get_max_features(n_features)
            feat_idx = np.random.choice(n_features, max_feat, replace=False)
            xt = X[idxs][:, feat_idx]
            yt = y[idxs]
            tree = DecisionTreeClassifierManual(max_depth=self.max_depth,
                                                min_samples_split=self.min_samples_split,
                                                max_features=None)
            tree.fit(xt, yt)
            self.trees.append(tree)
            self.features_indices.append(feat_idx)
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        probs = np.zeros(X.shape[0], dtype=float)
        for tree, feat_idx in zip(self.trees, self.features_indices):
            xt = X[:, feat_idx]
            probs += tree.predict_proba(xt)
        probs = probs / len(self.trees)
        return probs

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

class GradientBoostingClassifierManual:
    def __init__(self, n_estimators=50, learning_rate=0.1, max_depth=3, min_samples_split=2):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []
        self.initial_prediction = 0.0

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n_samples = X.shape[0]
        p = np.clip(np.mean(y), 1e-6, 1-1e-6)
        self.F = np.log(p / (1 - p)) * np.ones(n_samples)
        self.trees = []
        for m in range(self.n_estimators):
            p = sigmoid(self.F)
            residual = y - p
            tree = DecisionTreeRegressorManual(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X, residual)
            update = tree.predict(X)
            self.F += self.learning_rate * update
            self.trees.append(tree)
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        if not hasattr(self, 'train_baseline'):
            return 0.5 * np.ones(X.shape[0])
        F = self.train_baseline * np.ones(X.shape[0])
        for tree in self.trees:
            F += self.learning_rate * tree.predict(X)
        return sigmoid(F)

    def predict(self, X, threshold=0.5):
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def fit_with_baseline(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n_samples = X.shape[0]
        p = np.clip(np.mean(y), 1e-6, 1-1e-6)
        self.train_baseline = np.log(p / (1 - p))
        self.F = self.train_baseline * np.ones(n_samples)
        self.trees = []
        for m in range(self.n_estimators):
            p = sigmoid(self.F)
            residual = y - p
            tree = DecisionTreeRegressorManual(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X, residual)
            update = tree.predict(X)
            self.F += self.learning_rate * update
            self.trees.append(tree)
        return self


def evaluate_model(model, X_train, y_train, X_val, y_val):
    model.fit(X_train, y_train)
    try:
        y_score = model.predict_proba(X_val)
    except Exception:
        y_pred = model.predict(X_val)
        y_score = y_pred.astype(float)
    y_pred = (y_score >= 0.5).astype(int)
    metrics = {
        "roc_auc": roc_auc_score_manual(y_val, y_score),
        "f1": f1_score_manual(y_val, y_pred),
        "precision": precision_score_manual(y_val, y_pred),
        "recall": recall_score_manual(y_val, y_pred),
        "accuracy": accuracy_score(y_val, y_pred)
    }
    return metrics

def cross_val_score_manual(ModelClass, params, X, y, n_splits=5, random_state=42):
    folds = stratified_kfold_indices(y, n_splits=n_splits, random_state=random_state)
    metrics_list = []
    for i in range(n_splits):
        val_idx = folds[i]
        train_idx = np.hstack([folds[j] for j in range(n_splits) if j != i])
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        model = ModelClass(**params) if isinstance(ModelClass, type) else ModelClass
        if isinstance(model, GradientBoostingClassifierManual):
            model.fit_with_baseline(X_tr, y_tr)
            y_score = model.predict_proba(X_val)
            y_pred = (y_score >= 0.5).astype(int)
            metrics = {
                "roc_auc": roc_auc_score_manual(y_val, y_score),
                "f1": f1_score_manual(y_val, y_pred)
            }
        else:
            metrics = evaluate_model(model, X_tr, y_tr, X_val, y_val)
        metrics_list.append(metrics)
    avg = {}
    for k in metrics_list[0].keys():
        avg[k] = np.mean([m[k] for m in metrics_list])
    return avg

def grid_search_manual(ModelClass, grid_params, X, y, n_splits=3, metric='roc_auc'):
    keys = list(grid_params.keys())
    all_results = []
    best = {"params": None, "score": -np.inf}
    for vals in product(*[grid_params[k] for k in keys]):
        params = dict(zip(keys, vals))
        print("Testing params:", params)
        avg_metrics = cross_val_score_manual(ModelClass, params, X, y, n_splits=n_splits)
        score = avg_metrics[metric]
        print(" -> cv avg", metric, "=", score)
        all_results.append((params, avg_metrics))
        if score > best["score"]:
            best["score"] = score
            best["params"] = params
    return best["params"], best["score"], all_results

def load_featured_data(path=None):
    """
    Load featured data - direct path since model.py is outside data folder
    """
    if path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(current_dir, "data", "cart_abandonment_featured.csv")
    
    print(f"Loading data from: {path}")
    
    if not os.path.exists(path):
        data_dir = os.path.dirname(path)
        if os.path.exists(data_dir):
            available_files = os.listdir(data_dir)
            print(f"Available files in data directory: {available_files}")
        raise FileNotFoundError(f"Data file not found at {path}")
    
    try:
        df = pd.read_csv(path)
        # Check if target column exists
        if "abandoned" not in df.columns:
            raise ValueError(f"Target column 'abandoned' not found in dataset. Available columns: {df.columns.tolist()}")
        
        # Drop non-feature columns
        columns_to_drop = [c for c in ["abandoned", "session_id", "user_id"] if c in df.columns]
        X = df.drop(columns=columns_to_drop, errors='ignore')
        y = df["abandoned"].values
        return X.values.astype(float), y, X.columns.tolist()
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        raise

def main():
    X_all, y_all, feature_names = load_featured_data()
    
    BEST_MODEL_PARAMS["feature_names"] = feature_names

    print("N samples:", X_all.shape[0], "N features:", X_all.shape[1])
    train_idx, test_idx = stratified_train_test_split(X_all, y_all, test_size=0.2, random_state=42)
    X_train, y_train = X_all[train_idx], y_all[train_idx]
    X_test, y_test = X_all[test_idx], y_all[test_idx]
    print("Train size:", X_train.shape[0], "Test size:", X_test.shape[0])

    print("\n=== Baseline cross-validation (5-fold) ===")
    models_to_try = {
        "LogisticRegression": (LogisticRegressionGD, {"lr": 0.1, "n_iter": 2000, "l2": 1e-4}),
        "KNN": (KNNClassifier, {"k": 5}),
        "DecisionTree": (DecisionTreeClassifierManual, {"max_depth": 6, "min_samples_split": 5}),
        "RandomForest": (RandomForestManual, {"n_estimators": 10, "max_depth": 6, "min_samples_split": 5, "max_features": 'sqrt'})
    }
    for name, (cls, params) in models_to_try.items():
        print(f"\n{name} CV:")
        avg_metrics = cross_val_score_manual(cls, params, X_train, y_train, n_splits=5)
        print(" Avg metrics:", avg_metrics)

    print("\n=== Manual grid search: RandomForest (small grid) ===")
    rf_grid = {
        "n_estimators": [10, 20],
        "max_depth": [4, 6],
        "min_samples_split": [2, 5],
        "max_features": ['sqrt']
    }
    best_rf_params, best_rf_score, rf_results = grid_search_manual(RandomForestManual, rf_grid, X_train, y_train, n_splits=3, metric='roc_auc')
    print("Best RF params:", best_rf_params, "Score:", best_rf_score)

    print("\n=== Manual grid search: GradientBoosting (small grid) ===")
    gb_grid = {
        "n_estimators": [20, 50],
        "learning_rate": [0.05, 0.1],
        "max_depth": [2, 3],
        "min_samples_split": [2]
    }
    best_gb_params, best_gb_score, gb_results = grid_search_manual(GradientBoostingClassifierManual, gb_grid, X_train, y_train, n_splits=3, metric='roc_auc')
    print("Best GB params:", best_gb_params, "Score:", best_gb_score)

    print("\n=== Final training on full training set ===")
    if best_gb_score >= best_rf_score:
        print("Selecting Gradient Boosting as final")
        final_model = GradientBoostingClassifierManual(**best_gb_params)
        final_model.fit_with_baseline(X_train, y_train)
        BEST_MODEL_PARAMS["model_type"] = "GradientBoostingClassifierManual"
        BEST_MODEL_PARAMS["parameters"] = best_gb_params
    else:
        print("Selecting Random Forest as final")
        final_model = RandomForestManual(**best_rf_params)
        final_model.fit(X_train, y_train)
        BEST_MODEL_PARAMS["model_type"] = "RandomForestManual"
        BEST_MODEL_PARAMS["parameters"] = best_rf_params

    try:
        y_score_test = final_model.predict_proba(X_test)
    except Exception:
        y_score_test = final_model.predict(X_test).astype(float)
    y_pred_test = (y_score_test >= 0.5).astype(int)

    test_metrics = {
        "roc_auc": roc_auc_score_manual(y_test, y_score_test),
        "f1": f1_score_manual(y_test, y_pred_test),
        "precision": precision_score_manual(y_test, y_pred_test),
        "recall": recall_score_manual(y_test, y_pred_test),
        "accuracy": accuracy_score(y_test, y_pred_test)
    }
    
    # STORE PERFORMANCE METRICS
    BEST_MODEL_PARAMS["performance"] = test_metrics
    BEST_MODEL_PARAMS["grid_search_completed"] = True
    
    print("\nTest set metrics:", test_metrics)
    
    # Print the best parameters for easy reference
    print("\n" + "="*50)
    print("BEST PARAMETERS STORED FOR FUTURE USE:")
    print_best_params()
    print("="*50)

    model_path = "final_manual_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({
            "model": final_model,
            "feature_names": feature_names,
            "best_params": BEST_MODEL_PARAMS 
        }, f)
    print(f"Saved final model to {model_path}")

def quick_train_with_best_params(data_path=None, save_model=True):
    """
    Quick training function that uses the stored best parameters
    Call this for future training instead of running the full grid search
    """
    if not BEST_MODEL_PARAMS["grid_search_completed"]:
        print("Best parameters not found. Running full grid search first...")
        main()
        return
    
    print("=== QUICK TRAINING WITH STORED BEST PARAMETERS ===")
    print_best_params()
    
    # Load data
    X_all, y_all, feature_names = load_featured_data(data_path)
    
    # Split data
    train_idx, test_idx = stratified_train_test_split(X_all, y_all, test_size=0.2, random_state=42)
    X_train, y_train = X_all[train_idx], y_all[train_idx]
    X_test, y_test = X_all[test_idx], y_all[test_idx]
    
    # Create and train model with best parameters
    final_model = create_model_with_best_params()
    
    # Train with appropriate method
    if BEST_MODEL_PARAMS["model_type"] == "GradientBoostingClassifierManual":
        final_model.fit_with_baseline(X_train, y_train)
    else:
        final_model.fit(X_train, y_train)
    
    try:
        y_score_test = final_model.predict_proba(X_test)
    except Exception:
        y_score_test = final_model.predict(X_test).astype(float)
    
    y_pred_test = (y_score_test >= 0.5).astype(int)
    
    test_metrics = {
        "roc_auc": roc_auc_score_manual(y_test, y_score_test),
        "f1": f1_score_manual(y_test, y_pred_test),
        "precision": precision_score_manual(y_test, y_pred_test),
        "recall": recall_score_manual(y_test, y_pred_test),
        "accuracy": accuracy_score(y_test, y_pred_test)
    }
    
    print("\nTest performance with stored parameters:")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    if save_model:
        model_path = "quick_trained_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump({
                "model": final_model,
                "feature_names": feature_names,
                "best_params": BEST_MODEL_PARAMS
            }, f)
        print(f"Saved quick-trained model to {model_path}")
    
    return final_model, test_metrics

if __name__ == "__main__":
    main()
    
    print("\n" + "="*60)
    print("ACCESSING STORED PARAMETERS:")
    stored_params = get_best_params()
    print(f"Best model type: {stored_params['model_type']}")
    print(f"Best parameters: {stored_params['parameters']}")
