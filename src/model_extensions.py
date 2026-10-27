import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier

class AuditedMLPClassifier(MLPClassifier):

    def fit(self, X, y, sample_weight=None):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', ConvergenceWarning)
            result = super().fit(X, y, sample_weight=sample_weight)
        self.convergence_warning_count_ = sum((issubclass(w.category, ConvergenceWarning) for w in caught))
        self.convergence_warning_messages_ = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
        return result
