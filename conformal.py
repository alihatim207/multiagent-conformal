import numpy as np
import pickle
from classifier import pos_to_idx, idx_to_pos, compute_conformity_scores, GRID_SIZE


class ConformalPredictor:
    """
    Wraps a trained classifier and produces prediction sets with coverage guarantees.
    Sits on top of the classifier: raw probabilities -> threshold -> prediction set.
    """

    def __init__(self, clf, alpha=0.10):
        """
        clf:   trained scikit-learn classifier with predict_proba()
        alpha: error rate -- alpha=0.10 targets 90% coverage.
        """
        self.clf = clf
        self.alpha = alpha
        self.q = None  # set during calibrate()

    def calibrate(self, X_calib, y_calib):
        """
        Computes threshold q from the calibration set.

        Uses the finite sample correction: q = Quantile(scores, ceil((n+1)*alpha) / n)
        which gives exact rather than asymptotic coverage guarantees.
        """
        scores = compute_conformity_scores(self.clf, X_calib, y_calib)

        n = len(scores)
        corrected_quantile = np.ceil((n + 1) * self.alpha) / n
        corrected_quantile = np.clip(corrected_quantile, 0, 1)
        self.q = np.quantile(scores, corrected_quantile)

        return scores

    def predict_set(self, obs):
        """
        Returns the prediction set for a single observation: all positions
        whose predicted probability >= q. The true position is in this set
        with probability at least (1 - alpha).

        Returns: (prediction_set, probs)
        """
        if self.q is None:
            raise ValueError("Must call calibrate() before predict_set()")

        obs_array = np.array([obs])
        probs = self.clf.predict_proba(obs_array)[0]

        included_indices = np.where(probs >= self.q)[0]
        prediction_set = [idx_to_pos(idx) for idx in included_indices]

        return prediction_set, probs

    def predict_set_batch(self, X):
        """
        Vectorized version of predict_set() for a batch of observations.
        Returns: (list of prediction sets, probs array of shape (n, 100))
        """
        probs_batch = self.clf.predict_proba(X)

        prediction_sets = []
        for probs in probs_batch:
            included_indices = np.where(probs >= self.q)[0]
            prediction_set = [idx_to_pos(idx) for idx in included_indices]
            prediction_sets.append(prediction_set)

        return prediction_sets, probs_batch


def evaluate_coverage(predictor, X_test, y_test):
    """
    Measures empirical coverage on the test set: fraction of examples where
    the true label falls inside the prediction set. Should be >= (1 - alpha).

    Returns: (coverage, avg_set_size, set_sizes array)
    """
    prediction_sets, _ = predictor.predict_set_batch(X_test)

    covered = 0
    set_sizes = []

    for i, pred_set in enumerate(prediction_sets):
        true_pos = idx_to_pos(y_test[i])
        if true_pos in pred_set:
            covered += 1
        set_sizes.append(len(pred_set))

    set_sizes = np.array(set_sizes)
    coverage = covered / len(y_test)
    avg_set_size = set_sizes.mean()

    return coverage, avg_set_size, set_sizes


if __name__ == '__main__':
    print("=== Conformal Prediction ===\n")

    print("Loading from model.pkl...")
    with open('model.pkl', 'rb') as f:
        saved = pickle.load(f)

    clf     = saved['clf']
    X_calib = saved['X_calib']
    X_test  = saved['X_test']
    y_calib = saved['y_calib']
    y_test  = saved['y_test']
    print("  Loaded.\n")

    alphas = [0.30, 0.20, 0.10, 0.05]

    print(f"{'Coverage Target':>20} {'Alpha':>8} {'q':>10} {'Actual Coverage':>18} {'Avg Set Size':>15}")
    print("-" * 75)

    for alpha in alphas:
        predictor = ConformalPredictor(clf, alpha=alpha)
        predictor.calibrate(X_calib, y_calib)
        coverage, avg_set_size, _ = evaluate_coverage(predictor, X_test, y_test)
        target = 1 - alpha
        print(f"{target*100:>19.0f}% {alpha:>8.2f} {predictor.q:>10.4f} "
              f"{coverage*100:>17.1f}% {avg_set_size:>15.1f}")

    print()
    print("Deep dive at 90% coverage (alpha=0.10):\n")

    predictor_90 = ConformalPredictor(clf, alpha=0.10)
    predictor_90.calibrate(X_calib, y_calib)
    coverage, avg_set_size, set_sizes = evaluate_coverage(predictor_90, X_test, y_test)

    print(f"  Threshold q:         {predictor_90.q:.4f}")
    print(f"  Empirical coverage:  {coverage*100:.1f}%  (target: 90%)")
    print(f"  Average set size:    {avg_set_size:.1f} cells out of 100")
    print(f"  Min / Max set size:  {set_sizes.min()} / {set_sizes.max()}")
    print()

    print("  Set size distribution:")
    for size in sorted(np.unique(set_sizes)):
        count = np.sum(set_sizes == size)
        pct = count / len(set_sizes) * 100
        bar = '#' * int(pct / 2)
        print(f"    Size {size:>3}: {count:>5} ({pct:>5.1f}%)  {bar}")
    print()

    print("Concrete example:\n")
    example_obs = X_test[0]
    example_true = idx_to_pos(y_test[0])
    pred_set, probs = predictor_90.predict_set(tuple(example_obs.astype(int)))

    print(f"  Observation:  {tuple(example_obs.astype(int))}")
    print(f"  True position: {example_true}")
    print(f"  Prediction set ({len(pred_set)} cells): {sorted(pred_set)}")
    print(f"  True position in set: {example_true in pred_set}")
    print()

    print("  Top 5 probabilities:")
    top5_idx = np.argsort(probs)[::-1][:5]
    for idx in top5_idx:
        pos = idx_to_pos(idx)
        marker = "<-- in set" if probs[idx] >= predictor_90.q else ""
        print(f"    {pos}: {probs[idx]:.4f}  {marker}")
    print()

    saved['predictor_90'] = predictor_90
    with open('model.pkl', 'wb') as f:
        pickle.dump(saved, f)
    print("  Saved predictor to model.pkl")
