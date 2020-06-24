import numpy as np
# nayes.py - necessary features

# check array, should accept compressed and normal arrays

# get number of features from X.shape

# allow for multiclass classification by binarizing labels
# -> always binarize, this allows for a per-class instance count.
# -> If you take the dot product of the transposed binarization matrix
# -> and the feature matrix, you'll get discrete counts for each class
# -> in one quick process.

# count classes and features

# get class priors (log)

# get feature log probabilities

"""
In scikit-learn, the number of features is set by X.shape, while the number
of effective classes is set by a binarized y
"""


class MultiNayes:
    """
    Multinomial Naive Bayes algorithm.

    Paramaters
    ----------
    alpha : float, default=1.0
        Smoothing paramater, can be set to smaller values
        (0 for no smoothing)
    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def label_binarizer(self, y, classes=None, bin_labels=None):
        """convert labels into an array of shape
           (length of y, number of classes). This
           will assist in getting the log priors and probabilities"""
        if classes is None:
            classes = np.unique(y)
            bin_labels = np.zeros(
                                        (y.shape[0], classes.shape[0])
                                    )
            self.classes = classes
            self.bin_labels = bin_labels

        if bin_labels.shape[0] < 1:
            return None

        x = np.where(classes == y[0])
        bin_labels[0][x] = 1

        return self.label_binarizer(y[1:], classes, bin_labels[1:])

    def fit(self, X, y):
        # if X is not np.ndarray, convert from csr with `toarray()`
        if type(X) is not np.ndarray:
            X = X.toarray()

        self.label_binarizer(y)

        n_classes = self.classes.shape[0]
        n_features = X.shape[1]

        # initialize counter arrays
        self.class_count = np.zeros(n_classes)
        self.feature_count = np.zeros((n_classes, n_features))

        # count classes and features by getting
        # dot product of transposed binary labels
        # they are automatically separated into their
        # appropriate arrays
        self.feature_count += np.dot(self.bin_labels.T, X)
        self.class_count += self.bin_labels.sum(axis=0)

        # add smoothing
        if self.alpha > 0.0:
            self.feature_count += self.alpha
            smoothed_class_count = self.feature_count.sum(axis=1)

            # get conditional log probabilities
            self.feat_log_probs = (np.log(self.feature_count) -
                                   np.log(smoothed_class_count.reshape(-1, 1)))
        else:
            print(
                f"Alpha is {self.alpha}. A value this small will cause "
                "result in errors when feature count is 0"
            )
            self.feat_log_probs = np.log(
                                    self.feature_count /
                                    self.feature_count
                                    .sum(axis=1)
                                    .reshape(-1, 1)
                                  )

        # get log priors
        self.class_log_priors = (np.log(self.class_count) -
                                 np.log(self.class_count
                                 .sum(axis=0)
                                 .reshape(-1, 1)))

        def predict(self, X):
            pass


if __name__ == "__main__":
    clf = MultiNayes()
    y = np.array([1, 2, 3, 4, 5, 7])
    clf.label_binarizer(y)
    print(clf.classes.shape[0])
    print(clf.bin_labels)
