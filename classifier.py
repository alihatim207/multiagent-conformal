import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
from environment import Environment, GRID_SIZE, observe


def pos_to_idx(pos):
    """Maps (x, y) to a flat integer index: x + y * GRID_SIZE."""
    x, y = pos
    return int(x + y * GRID_SIZE)

def idx_to_pos(idx):
    """Reverses pos_to_idx: modulo gives x, integer division gives y."""
    x = idx % GRID_SIZE
    y = idx // GRID_SIZE
    return (x, y)


def generate_dataset(n_samples=20000, seed=42):
    """
    Generates (observation, true_position) pairs by sampling random grid cells
    and adding noise via observe(). Uses the same noise model as the environment
    so train and test distributions match.

    Returns:
        X: shape (n_samples, 2) -- noisy observations [obs_x, obs_y]
        y: shape (n_samples,)   -- flat class indices (0 to GRID_SIZE^2 - 1)
    """
    rng = np.random.default_rng(seed)

    X = []
    y = []

    for _ in range(n_samples):
        true_x = rng.integers(0, GRID_SIZE)
        true_y = rng.integers(0, GRID_SIZE)
        true_pos = (true_x, true_y)

        obs = observe(true_pos, rng)

        X.append([obs[0], obs[1]])
        y.append(pos_to_idx(true_pos))

    X = np.array(X, dtype=float)
    y = np.array(y, dtype=int)

    return X, y


def split_dataset(X, y, train_frac=0.6, calib_frac=0.2, seed=42):
    """
    Splits into train / calibration / test.
    Calibration must be held out from training so conformal scores
    aren't inflated by the model having already seen those examples.

    Returns: X_train, X_calib, X_test, y_train, y_calib, y_test
    """
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=(1 - train_frac),
        random_state=seed
    )

    calib_ratio = calib_frac / (1 - train_frac)
    X_calib, X_test, y_calib, y_test = train_test_split(
        X_temp, y_temp,
        test_size=(1 - calib_ratio),
        random_state=seed
    )

    return X_train, X_calib, X_test, y_train, y_calib, y_test


def train_classifier(X_train, y_train):
    """
    Trains a softmax logistic regression on the training set.
    lbfgs handles multinomial logistic well; max_iter=1000 ensures convergence
    across 100 classes.
    """
    clf = LogisticRegression(
        solver='lbfgs',
        max_iter=1000,
        C=1.0
    )
    clf.fit(X_train, y_train)
    return clf


def compute_conformity_scores(clf, X_calib, y_calib):
    """
    For each calibration example, pulls the predicted probability for the true class.
    These scores feed into conformal calibration in Layer 3.

    Returns: shape (n_calib,) array of conformity scores.
    """
    probs = clf.predict_proba(X_calib)
    # Fancy indexing: probs[i, y_calib[i]] for every i simultaneously.
    scores = probs[np.arange(len(y_calib)), y_calib]
    return scores


if __name__ == '__main__':
    print("=== Classifier Training ===\n")

    print("Generating dataset...")
    X, y = generate_dataset(n_samples=20000, seed=42)
    print(f"  Samples: {len(X)}  |  Input shape: {X.shape}  |  Labels: {y.min()}-{y.max()}")
    print()

    print("Splitting dataset...")
    X_train, X_calib, X_test, y_train, y_calib, y_test = split_dataset(X, y)
    print(f"  Train: {len(X_train)}  |  Calib: {len(X_calib)}  |  Test: {len(X_test)}")
    print()

    print("Training logistic regression...")
    clf = train_classifier(X_train, y_train)
    print("  Done.")
    print()

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Top-1 accuracy: {acc:.3f}  (expected low -- 100 classes from 2 features)")
    print()

    print("Computing conformity scores...")
    scores = compute_conformity_scores(clf, X_calib, y_calib)
    print(f"  Mean: {scores.mean():.4f}  |  Median: {np.median(scores):.4f}  |  Range: [{scores.min():.4f}, {scores.max():.4f}]")
    print()

    print("Threshold q at different coverage levels:")
    for coverage in [0.70, 0.80, 0.90, 0.95]:
        alpha = 1 - coverage
        q = np.quantile(scores, alpha)
        print(f"  {coverage*100:.0f}%  (alpha={alpha:.2f})  ->  q = {q:.4f}")
    print()

    print("Saving to model.pkl...")
    with open('model.pkl', 'wb') as f:
        pickle.dump({
            'clf':     clf,
            'X_train': X_train,
            'X_calib': X_calib,
            'X_test':  X_test,
            'y_train': y_train,
            'y_calib': y_calib,
            'y_test':  y_test,
            'scores':  scores,
        }, f)
    print("  Saved.")
