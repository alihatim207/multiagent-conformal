import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
from environment import Environment, GRID_SIZE, observe

# -------------------------------------------------------
# POSITION <-> CLASS INDEX CONVERSION
# -------------------------------------------------------
# The classifier needs class labels as integers, not (x,y) tuples.
# We flatten the 2D grid into a 1D list of 100 class indices.
# Position (x, y) maps to index: x + y * GRID_SIZE
# Example: (0,0)->0, (1,0)->1, (9,0)->9, (0,1)->10, (9,9)->99

def pos_to_idx(pos):
    """
    Converts a (x, y) grid position to a flat integer class index.
    x + y * GRID_SIZE flattens row by row.
    """
    x, y = pos
    return int(x + y * GRID_SIZE)

def idx_to_pos(idx):
    """
    Converts a flat integer index back to (x, y) grid position.
    Reverse of pos_to_idx using modulo and integer division.
    % GRID_SIZE gives x (column), // GRID_SIZE gives y (row).
    """
    x = idx % GRID_SIZE
    y = idx // GRID_SIZE
    return (x, y)


# -------------------------------------------------------
# DATASET GENERATION
# -------------------------------------------------------
def generate_dataset(n_samples=20000, seed=42):
    """
    Generates a dataset of (observation, true_position) pairs
    by sampling random true positions across the grid and
    producing noisy observations of each.

    Why 20000 samples? With 100 classes, you want enough
    samples per class for the classifier to learn reliably.
    20000 / 100 = 200 samples per class on average.

    Args:
        n_samples: how many (observation, true_pos) pairs to generate
        seed:      random seed for reproducibility

    Returns:
        X: array of shape (n_samples, 2) -- the noisy observations
           Each row is [obs_x, obs_y]
        y: array of shape (n_samples,)  -- class indices (0-99)
           Each value is the flattened true position index
    """
    rng = np.random.default_rng(seed)

    X = []  # observations (input features)
    y = []  # true position class indices (labels)

    for _ in range(n_samples):
        # Sample a random true position anywhere on the grid.
        # rng.integers(low, high) samples a uniform integer in [low, high).
        # We sample x and y independently.
        true_x = rng.integers(0, GRID_SIZE)
        true_y = rng.integers(0, GRID_SIZE)
        true_pos = (true_x, true_y)

        # Generate a noisy observation of this true position.
        # This calls the same observe() function from environment.py,
        # so the noise model is identical at training time and test time.
        # This is important -- if the noise at test time were different,
        # the classifier's learned mapping would be wrong.
        obs = observe(true_pos, rng)

        # Store the observation as a feature vector [obs_x, obs_y]
        # and the true position as a class index.
        X.append([obs[0], obs[1]])
        y.append(pos_to_idx(true_pos))

    # Convert lists to numpy arrays.
    # np.array() turns a Python list into a NumPy array for fast math ops.
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=int)

    return X, y


# -------------------------------------------------------
# TRAIN / CALIBRATION / TEST SPLIT
# -------------------------------------------------------
def split_dataset(X, y, train_frac=0.6, calib_frac=0.2, seed=42):
    """
    Splits dataset into three parts: train, calibration, test.

    Why three splits (not the usual two)?
    - Train set:       used to fit the classifier weights
    - Calibration set: used by conformal prediction to find threshold q
    - Test set:        used to evaluate final coverage guarantee

    The calibration set MUST be separate from training data.
    If you calibrate on training data, the model has already seen
    those examples and its conformity scores will be overconfident,
    breaking the coverage guarantee.

    Args:
        train_frac: fraction of data for training (default 60%)
        calib_frac: fraction for calibration (default 20%)
        remaining:  goes to test set (default 20%)

    Returns:
        Six arrays: X_train, X_calib, X_test, y_train, y_calib, y_test
    """
    # First split: separate train from (calibration + test)
    # test_size = 1 - train_frac means we keep that fraction as remainder
    # random_state=seed ensures the split is reproducible
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=(1 - train_frac),
        random_state=seed
    )

    # Second split: split remainder into calibration and test
    # calib_frac / (1 - train_frac) gives the right proportion
    # e.g. 0.2 / 0.4 = 0.5 means half of remainder goes to calibration
    calib_ratio = calib_frac / (1 - train_frac)
    X_calib, X_test, y_calib, y_test = train_test_split(
        X_temp, y_temp,
        test_size=(1 - calib_ratio),
        random_state=seed
    )

    return X_train, X_calib, X_test, y_train, y_calib, y_test


# -------------------------------------------------------
# TRAIN CLASSIFIER
# -------------------------------------------------------
def train_classifier(X_train, y_train):
    """
    Trains a logistic regression classifier on the training set.

    Why logistic regression?
    - It directly outputs a probability distribution over all 100 classes
      via the softmax function, which is exactly what conformal prediction needs
    - It's fast to train, easy to understand, and debuggable
    - For a 2D input (obs_x, obs_y) -> 100 classes, it's well-suited

    LogisticRegression parameters:
    - multi_class='multinomial': use softmax over all classes simultaneously
      (as opposed to one-vs-rest which trains 100 separate binary classifiers)
    - solver='lbfgs': an optimization algorithm well-suited for multinomial
      logistic regression. L-BFGS is a quasi-Newton method that converges
      faster than plain gradient descent for this type of problem.
    - max_iter=1000: maximum optimization iterations. Default is 100 which
      is often not enough for 100 classes -- we increase it to ensure
      the optimizer converges fully.
    - C=1.0: inverse regularization strength. Higher C = less regularization
      = model fits training data more closely. We keep the default.
    """
    clf = LogisticRegression(
        solver='lbfgs',
        max_iter=1000,
        C=1.0
    )

    # fit() trains the model: finds weights W and bias b such that
    # softmax(W @ X + b) best predicts y across training examples.
    # This minimizes cross-entropy loss using L-BFGS optimization.
    clf.fit(X_train, y_train)

    return clf


# -------------------------------------------------------
# COMPUTE CONFORMITY SCORES
# -------------------------------------------------------
def compute_conformity_scores(clf, X_calib, y_calib):
    """
    Computes the conformity score for each calibration example.

    The conformity score for example i is:
        s_i = f(X_calib[i])[y_calib[i]]

    That is: run the observation through the classifier,
    get the full probability distribution over 100 classes,
    then pull out only the probability assigned to the TRUE label.

    This measures how "conforming" each example is -- how well
    the model's output matches what actually happened.

    Args:
        clf:     trained LogisticRegression classifier
        X_calib: calibration observations, shape (n_calib, 2)
        y_calib: true class indices for calibration set, shape (n_calib,)

    Returns:
        scores: array of shape (n_calib,) -- one conformity score per example
    """
    # predict_proba() returns a (n_samples, n_classes) array.
    # Each row is a probability distribution over all 100 classes.
    # Row i sums to 1.0.
    probs = clf.predict_proba(X_calib)

    # For each calibration example i, we want probs[i, y_calib[i]].
    # np.arange(len(y_calib)) creates [0, 1, 2, ..., n_calib-1]
    # Using it alongside y_calib does fancy indexing:
    # probs[0, y_calib[0]], probs[1, y_calib[1]], ... simultaneously.
    # This is much faster than a for loop.
    scores = probs[np.arange(len(y_calib)), y_calib]

    return scores


# -------------------------------------------------------
# MAIN: TRAIN AND SAVE EVERYTHING
# -------------------------------------------------------
if __name__ == '__main__':
    print("=== Layer 2: Classifier Training ===\n")

    # Step 1: Generate dataset
    print("Generating dataset...")
    X, y = generate_dataset(n_samples=20000, seed=42)
    print(f"  Total samples: {len(X)}")
    print(f"  Input shape:   {X.shape}  (obs_x, obs_y)")
    print(f"  Label range:   {y.min()} to {y.max()}  (0 to {GRID_SIZE**2 - 1})")
    print()

    # Step 2: Split into train / calibration / test
    print("Splitting dataset...")
    X_train, X_calib, X_test, y_train, y_calib, y_test = split_dataset(X, y)
    print(f"  Train:       {len(X_train)} samples")
    print(f"  Calibration: {len(X_calib)} samples")
    print(f"  Test:        {len(X_test)} samples")
    print()

    # Step 3: Train the classifier
    print("Training logistic regression classifier...")
    clf = train_classifier(X_train, y_train)
    print("  Done.")
    print()

    # Step 4: Evaluate on test set
    # predict() returns the single most likely class for each input.
    # This is top-1 accuracy -- how often the single prediction is correct.
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Top-1 accuracy on test set: {acc:.3f}")
    print("  (This will be low -- 100 classes from just 2 input features")
    print("   is a hard problem. Conformal prediction compensates for this.)")
    print()

    # Step 5: Compute and inspect conformity scores
    print("Computing conformity scores on calibration set...")
    scores = compute_conformity_scores(clf, X_calib, y_calib)
    print(f"  Mean conformity score:   {scores.mean():.4f}")
    print(f"  Median conformity score: {np.median(scores):.4f}")
    print(f"  Min:                     {scores.min():.4f}")
    print(f"  Max:                     {scores.max():.4f}")
    print()

    # Step 6: Show what q would be at different alpha levels
    # This previews Layer 3's threshold computation
    print("Preview: threshold q at different coverage levels:")
    for coverage in [0.70, 0.80, 0.90, 0.95]:
        alpha = 1 - coverage
        q = np.quantile(scores, alpha)
        print(f"  Coverage {coverage*100:.0f}%  (alpha={alpha:.2f})  ->  q = {q:.4f}")
    print()

    # Step 7: Save the classifier and all splits to disk
    # pickle.dump() serializes a Python object to a binary file
    # so we can reload it in later layers without retraining.
    print("Saving classifier and data splits...")
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
    print("  Saved to model.pkl")
    print()
    print("Layer 2 complete.")
