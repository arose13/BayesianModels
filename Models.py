import numpy as np
import pandas as pd
from scipy.optimize import fmin
from sklearn.neighbors import KernelDensity
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin


def _np_dropna(a):
    return a[~np.isnan(a).any(axis=1)]


class BayesKDEClassifier(BaseEstimator, ClassifierMixin):
    """
    Bayesian Classifier that uses Kernel Density Estimations to generate the joint distribution

    Parameters:
        - bandwidth: float
        - kernel: for scikit learn KernelDensity
    """
    def __init__(self, bandwidth=0.2, kernel='gaussian'):
        self.classes_, self.models_, self.priors_logp_ = [None] * 3
        self.bandwidth = bandwidth
        self.kernel = kernel

    def fit(self, X, y):
        self.classes_ = np.sort(np.unique(y))
        training_sets = [X[y == yi] for yi in self.classes_]
        self.models_ = [KernelDensity(bandwidth=self.bandwidth, kernel=self.kernel).fit(x_subset)
                        for x_subset in training_sets]

        self.priors_logp_ = [np.log(x_subset.shape[0] / X.shape[0]) for x_subset in training_sets]
        return self

    def predict_proba(self, X):
        logp = np.array([model.score_samples(X) for model in self.models_]).T
        result = np.exp(logp + self.priors_logp_)
        return result / result.sum(1, keepdims=True)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), 1)]


class KDMLEstimator(BaseEstimator):
    """
    Kernel Density Maximum Likelihood Estimator

    It compute an estimate of the joint probability and the fills in blank values with the MLE for that value.
    Good for imputation and regression
    """
    def __init__(self, automatic_optimisation=False, bandwidth=0.2, kernel='gaussian'):
        """
        if automatic_optimisation is given bandwidth is ignored

        :param automatic_optimisation: Whether to find the best bandwidth value
        :param bandwidth: passed to the kernel density estimator
        """
        self.auto_opt = automatic_optimisation
        self.bandwidth, self.kernel = bandwidth, kernel
        self.model = None

    def fit(self, X, y=None):
        print('Before dropna', X.shape)
        X = X.values if isinstance(X, pd.DataFrame) else X
        X = _np_dropna(X)
        print('After dropna', X.shape)

        # Train model
        # TODO check automatic optimisation
        self.model = KernelDensity(self.bandwidth, kernel=self.kernel)
        self.model.fit(X)

        return self

    def transform(self, X):
        # Iterate and fill in all rows
        from tqdm import tqdm
        progress = tqdm(range(len(X)))
        for i in progress: #range(len(X)):
            row = X[i, :].copy()
            missing_data = np.isnan(row)
            initial_guess = np.full(sum(missing_data), 0.0)  # Replace with median at that point

            def _logp(x_guess):
                row[missing_data] = x_guess
                return -self.model.score_samples(row)

            soln = fmin(_logp, initial_guess)
            row[missing_data] = soln
            progress.set_description(f'{row}')
            X[i, :] = row


if __name__ == '__main__':
    from sklearn.datasets import make_classification, make_regression
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import Imputer

    n = 1000

    x, y = make_classification(n, 3, 2, 1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=1992)

    # logit = LogisticRegression()
    # logit.fit(x_train, y_train)
    #
    # bkde = BayesKDEClassifier()
    # bkde.fit(x_train, y_train)
    #
    # print('Logit Train:', logit.score(x_train, y_train))
    # print('Logit Test :', logit.score(x_test, y_test))
    #
    # print('KDE Training:', bkde.score(x_train, y_train))
    # print('KDE Test    :', bkde.score(x_test, y_test))

    x, y, reg_coef_ = make_regression(n, 1, 1, noise=15, coef=True, random_state=True)
    y = y.reshape((n, 1))
    y = (y - y.mean()) / y.std()
    x = np.hstack((x, y))
    print(x.shape, y.shape, reg_coef_.shape)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=1992)

    kdmle = KDMLEstimator()
    kdmle.fit(x_train)

    import seaborn as sns
    import matplotlib.pyplot as graph

    sampled = kdmle.model.sample(20000)
    graph.hist2d(sampled[:, 0], sampled[:, 1], bins=256, cmap='inferno')
    graph.show()
    # sns.jointplot(x[:, 0], x[:, 1], kind='kde')
    # graph.show()
