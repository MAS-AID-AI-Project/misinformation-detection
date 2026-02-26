import joblib
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class XGBoostModel:
    def __init__(self, **params):
        self.model = XGBClassifier(
            **params
        )
        self.is_trained = False

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        self.is_trained = True

    def predict(self, X):
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction.")
        return self.model.predict(X)

    def evaluate(self, X_test, y_test, pos_label=1, verbose=True):
        preds = self.predict(X_test)
        acc = accuracy_score(y_test, preds)
        precision = precision_score(y_test, preds, pos_label=pos_label)
        recall = recall_score(y_test, preds, pos_label=pos_label)
        f1 = f1_score(y_test, preds, pos_label=pos_label)
        cm = confusion_matrix(y_test, preds)
        if verbose:
            print(f"Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")
            self.plot_confusion_matrix(cm)
        return acc, precision, recall, f1

    def plot_confusion_matrix(self, cm):
        plt.figure(figsize=(4,3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix - XGBoost')
        plt.tight_layout()
        plt.show()

    def save_model(self, filepath):
        if not self.is_trained:
            raise RuntimeError("Cannot save an untrained model.")
        joblib.dump(self.model, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        self.model = joblib.load(filepath)
        self.is_trained = True
        print(f"Model loaded from {filepath}")
