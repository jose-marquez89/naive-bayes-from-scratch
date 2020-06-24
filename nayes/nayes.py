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


if __name__ == "__main__":
    clf = MultiNayes()
    y = np.array([1, 0, 1, 0, 0, 0, 1, 1 ,1])
    clf.label_binarizer(y)
    print(clf.classes)
    print(clf.bin_labels)
