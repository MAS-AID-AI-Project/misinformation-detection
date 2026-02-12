import pandas as pd
import json
from ..eval import expected_cost_metric, operational_efficiency, plot_confusion_matrix, plot_roc_pr 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt 

class RuleBasedClassifier:
    def __init__(
        self,
        fake_sources=None,
        real_sources=None,
        length_threshold=None,
        config_path=None
    ):
        """
        Initialize the classifier with domain and length heuristics.
        You can set parameters directly or load from a JSON config.
        """
        if config_path:
            with open(config_path, "r") as f:
                config = json.load(f)
            self.fake_sources = config.get("fake_sources", ["hollywoodlife.com"])
            self.real_sources = config.get("real_sources", ["dailymail.co.uk"])
            self.length_threshold = config.get("length_threshold", 638)
        else:
            self.fake_sources = fake_sources if fake_sources else ["hollywoodlife.com"]
            self.real_sources = real_sources if real_sources else ["dailymail.co.uk"]
            self.length_threshold = length_threshold if length_threshold is not None else 638

    def fit(self, X, y=None):
        """
        Fitting not required for rule-based classifier,
        but included for API compatibility.
        """
        pass

    def predict_row(self, row):
        """
        Apply rules to predict label for a single data row (dict or pandas Series).
        """
        source = row.get("source_domain", None)
        wordcount = row.get("word_count", None)
        if source in self.fake_sources:
            return "fake"
        elif source in self.real_sources:
            return "real"
        else:
            if wordcount is not None and wordcount > self.length_threshold:
                return "real"
            else:
                return "fake"

    def predict(self, X_df):
        """
        Predict labels for the input DataFrame using rules.
        """
        return X_df.apply(self.predict_row, axis=1)

    def save_config(self, config_path):
        """
        Save current rule parameters to a config JSON.
        """
        config = {
            "fake_sources": self.fake_sources,
            "real_sources": self.real_sources,
            "length_threshold": self.length_threshold
        }
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

    def load_config(self, config_path):
        """
        Load rule parameters from config JSON.
        """
        with open(config_path, "r") as f:
            config = json.load(f)
        self.fake_sources = config.get("fake_sources", ["hollywoodlife.com"])
        self.real_sources = config.get("real_sources", ["dailymail.co.uk"])
        self.length_threshold = config.get("length_threshold", 638)

    def eval(self, X_test, y_test):
        name = "Rule-Based Classifier"
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
        # Assuming the RuleBasedClassifier uses specific labels
        labels = ["real", "fake"] 
        plot_confusion_matrix(y_true=y_test, y_pred=y_pred, 
                              labels=labels, 
                              title=f"Confusion Matrix — {name}")

        return {
            "acc": acc,
            "prec": prec,
            "rec": rec,
            "f1": f1,
            "expected_cost": expected_cost,
            }
