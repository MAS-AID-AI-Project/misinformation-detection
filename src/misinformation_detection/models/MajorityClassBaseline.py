import joblib
from sklearn.dummy import DummyClassifier
from ..eval import expected_cost_metric, operational_efficiency, plot_confusion_matrix, plot_roc_pr 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt 

class MajorityClassBaseline:
    def __init__(
        self,
        strategy="most_frequent",
        model_path=None,
        dummy_params=None
    ):
        """
        Initialize DummyClassifier to always predict the majority class.
        You can provide param dict or load from checkpoint.
        """
        if model_path:
            self.model = joblib.load(model_path)
        else:
            default_args = {"strategy": strategy}
            self.model = DummyClassifier(**(dummy_params or default_args))

    def fit(self, X, y):
        """
        Fit the dummy classifier on training labels.
        X is only used to match shape API compatibility.
        """
        self.model.fit(X, y)

    def predict(self, X):
        """
        Predict with the dummy classifier.
        X shape matches fit; values ignored.
        """
        return self.model.predict(X)

    def most_frequent_class(self):
        """
        Returns the value of the majority class after fitting.
        """
        return self.model.classes_[self.model.class_prior_.argmax()]

    def save(self, model_path):
        """
        Save trained dummy classifier.
        """
        joblib.dump(self.model, model_path)

    def load(self, model_path):
        """
        Load the dummy classifier from disk.
        """
        self.model = joblib.load(model_path)

    def eval(self, X_test, y_test):
        name = "Majority Class Baseline"
        y_pred = self.predict(X_test)
        
        # Standard Metrics Calculation and Print (Kept in class)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, pos_label="fake", zero_division=0)
        rec = recall_score(y_test, y_pred, pos_label="fake", zero_division=0)
        f1 = f1_score(y_test, y_pred, pos_label="fake", zero_division=0)

        print(f"{name} Metrics")
        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall: {rec:.4f}")
        print(f"F1-Score: {f1:.4f}")

        # Custom Metric (Using imported function)
        expected_cost = expected_cost_metric(y_test, y_pred, name=name)

        # Plotting (Using imported function)
        plot_confusion_matrix(y_true=y_test, y_pred=y_pred, 
                              labels=self.model.classes_, 
                              title=f"Confusion Matrix — {name}")

        return {
            "acc": acc,
            "prec": prec,
            "rec": rec,
            "f1": f1,
            "expected_cost": expected_cost,
            }
