import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from ..eval import expected_cost_metric, operational_efficiency, plot_confusion_matrix, plot_roc_pr 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt 


class DecisionTreeBaseline:
    def __init__(self, 
                 max_depth=5, 
                 random_state=42,
                 vectorizer_path=None, 
                 model_path=None,
                 vectorizer_params=None,
                 tree_params=None):
        """
        Initialize a DecisionTreeClassifier and TfidfVectorizer for text classification.

        You can provide params for vectors/tree or pass paths to pretrained objects.
        """
        # Load vectorizer or setup from params
        if vectorizer_path:
            self.vectorizer = joblib.load(vectorizer_path)
        else:
            self.vectorizer = TfidfVectorizer(**(vectorizer_params or {"max_features": 100, "stop_words": "english"}))
        
        # Load model or setup from params
        if model_path:
            self.model = joblib.load(model_path)
        else:
            tree_args = tree_params if tree_params else {"max_depth": max_depth, "random_state": random_state}
            self.model = DecisionTreeClassifier(**tree_args)

    def fit(self, X_text, y):
        """
        Fit model to input data (text and labels).
        Automatically fits TF-IDF vectorizer first if needed.
        """
        X_vec = self.vectorizer.fit_transform(X_text)
        self.model.fit(X_vec, y)

    def transform(self, X_text):
        """
        Transform text with the vectorizer.
        """
        return self.vectorizer.transform(X_text)

    def predict(self, X_text):
        """
        Predict on input text.
        """
        X_vec = self.transform(X_text)
        return self.model.predict(X_vec)

    def save(self, model_path, vectorizer_path):
        """
        Save the trained model and vectorizer.
        """
        joblib.dump(self.model, model_path)
        joblib.dump(self.vectorizer, vectorizer_path)

    def load(self, model_path, vectorizer_path):
        """
        Load model and vectorizer from disk.
        """
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)


        
    def eval(self, X_test, y_test):
        name = "TF-IDF Decision Tree"
        y_pred = self.predict(X_test)
        
        # Standard Metrics Calculation and Print (Kept in class)
        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, pos_label="fake", zero_division=0)
        recall = recall_score(y_test, y_pred, pos_label="fake", zero_division=0)
        f1 = f1_score(y_test, y_pred, pos_label="fake", zero_division=0)

        print(f"{name} Metrics")
        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")

        # Custom Metrics (Using imported functions)
        expected_cost = expected_cost_metric(y_test, y_pred, name=name)

        operational_eff = operational_efficiency(model=self.model, 
                                                 vectorizer=self.vectorizer, 
                                                 X_test=X_test, 
                                                 name=name)

        # Confusion Matrix (Using imported function)
        plot_confusion_matrix(y_true=y_test, y_pred=y_pred, 
                              labels=self.model.classes_, 
                              title=f"Confusion Matrix — {name}")

        # ROC and Precision-Recall (Logic moved to utility function)
        X_test_tfidf = self.vectorizer.transform(X_test)
        probs = self.model.predict_proba(X_test_tfidf)
        fake_idx = list(self.model.classes_).index("fake")
        y_scores = probs[:, fake_idx]

        roc_auc = plot_roc_pr(y_test=y_test, y_scores=y_scores, 
                              pos_label="fake", 
                              title_suffix=name)

        return {
            "acc": acc,
            "prec": precision,
            "rec": recall,
            "f1": f1,
            "expected_cost": expected_cost,
            "operational_eff": operational_eff,
            "roc_auc": roc_auc
            }
