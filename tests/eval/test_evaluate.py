import pytest
import numpy as np
from unittest.mock import MagicMock
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from misinformation_detection.eval import expected_cost_metric, operational_efficiency, plot_confusion_matrix, plot_roc_pr
from sklearn.metrics import roc_curve, auc


# --- Mocks for Operational Efficiency Testing ---

# A simple mock class for the Vectorizer
class MockVectorizer:
    def transform(self, X):
        # Always return a placeholder array of the correct size
        return np.ones((len(X), 10))

# A simple mock class for the Model (must implement predict_proba)
class MockModel:
    def __init__(self, probabilities):
        self.probabilities = probabilities
        self.classes_ = ['real', 'fake'] # Required for some plotting functions

    def predict_proba(self, X):
        # Returns probabilities for two classes
        # probabilities array is for the positive class (index 1)
        return np.stack([1 - self.probabilities, self.probabilities], axis=1)

# --- Fixtures and Test Data ---

@pytest.fixture
def binary_data():
    """Returns sample true labels, hard predictions, and probability scores."""
    y_true = np.array(['real', 'real', 'fake', 'fake', 'real', 'fake', 'fake', 'real', 'fake', 'fake']) # 4 Real, 6 Fake
    y_pred = np.array(['real', 'real', 'fake', 'fake', 'real', 'fake', 'real', 'fake', 'fake', 'fake']) # 3 Real, 7 Fake
    y_scores = np.array([0.1, 0.2, 0.9, 0.8, 0.6, 0.75, 0.2, 0.7, 0.95, 0.85]) # Probabilities for 'fake'
    
    # Correct Confusion Matrix based on y_true/y_pred:
    # TN (Real -> Real): 3
    # FP (Real -> Fake): 1
    # FN (Fake -> Real): 1
    # TP (Fake -> Fake): 5
    
    return y_true, y_pred, y_scores

# --- Test expected_cost_metric ---

def test_expected_cost_metric_calculation(binary_data, capsys):
    """Tests the core cost calculation logic."""
    y_true, y_pred, _ = binary_data
    
    # Correct Rates & Cost:
    # FP Rate = 1 / (1 + 3) = 0.25
    # FN Rate = 1 / (1 + 5) = 0.16666...
    # Expected Cost (cost_fp=2, cost_fn=5) = (0.25 * 2) + (0.16666... * 5) = 0.5 + 0.83333... = 1.33333...
    
    result = expected_cost_metric(y_true, y_pred, name="Test Model")
    
    assert np.isclose(result, 1.3333333333333333)
    
    # Check printed output for rates
    captured = capsys.readouterr()
    assert "False Positive Rate : 0.2500" in captured.out
    assert "False Negative Rate : 0.1667" in captured.out

def test_expected_cost_metric_custom_costs(binary_data):
    """Tests the cost calculation with custom FP/FN costs."""
    y_true, y_pred, _ = binary_data
    
    # Correct Rates & Custom Cost:
    # FP Rate = 0.25, FN Rate = 0.16666...
    # Custom Cost: cost_fp=10, cost_fn=1. Expected = (0.25 * 10) + (0.16666... * 1) = 2.5 + 0.16666... = 2.66666...
    
    result = expected_cost_metric(y_true, y_pred, name="Custom Cost Test", cost_fp=10, cost_fn=1)
    
    assert np.isclose(result, 2.6666666666666665)

def test_expected_cost_metric_single_class_prediction():
    """Tests the case where y_pred has only one class (e.g., all predicted real)."""
    y_true = np.array(['real', 'real', 'fake', 'fake']) # 2 Real, 2 Fake
    y_pred = np.array(['real', 'real', 'real', 'real']) # All predicted Real
    
    # CM: TN=2, FP=0, FN=2, TP=0
    # FP Rate = 0 / 2 = 0.0
    # FN Rate = 2 / 2 = 1.0
    # Expected Cost = (0.0 * 2) + (1.0 * 5) = 5.0

    result = expected_cost_metric(y_true, y_pred, name="Single Prediction Test")

    # The assertion for 5.0 was correct based on the input data
    assert np.isclose(result, 5.0)

def test_expected_cost_metric_single_class_input():
    """Tests error handling for single-class input (should not raise ValueError)."""
    y_true = np.array(['real', 'real', 'real'])
    y_pred = np.array(['real', 'real', 'real'])
    
    # CM: TN=3, FP=0, FN=0, TP=0 (with implicit 'fake' label being absent)
    # Expected Cost = 0.0
    
    result = expected_cost_metric(y_true, y_pred, name="Single Class Input Test")
    
    assert np.isclose(result, 0.0)

# --- Test operational_efficiency ---

def test_operational_efficiency_calculation():
    """Tests the core calculation of auto-approved percentage."""
    X_test = ["doc1", "doc2", "doc3", "doc4", "doc5", "doc6", "doc7", "doc8", "doc9", "doc10"]
    
    probs_for_fake = np.array([0.1, 0.2, 0.4, 0.5, 0.6, 0.65, 0.75, 0.8, 0.9, 0.25])
    # Auto-approved (Outside [0.3, 0.7]): 0.1, 0.2, 0.75, 0.8, 0.9, 0.25 -> 6 samples
    # Expected Auto-approved % = 6 / 10 = 0.6 (60.00%)
    
    mock_model = MockModel(probabilities=probs_for_fake)
    mock_vectorizer = MockVectorizer()
    
    result = operational_efficiency(
        model=mock_model, 
        vectorizer=mock_vectorizer, 
        X_test=X_test, 
        name="Mock Efficiency Test"
    )
    
    assert np.isclose(result, 0.6)

def test_operational_efficiency_custom_thresholds():
    """Tests calculation with custom low and high thresholds."""
    X_test = ["doc1", "doc2", "doc3", "doc4"]
    probs_for_fake = np.array([0.1, 0.45, 0.55, 0.95])
    
    # Custom thresholds: low=0.4, high=0.8
    # Auto-approved: 0.1, 0.95 -> 2 samples
    # Expected Auto-approved % = 2 / 4 = 0.5 (50.00%)
    
    mock_model = MockModel(probabilities=probs_for_fake)
    mock_vectorizer = MockVectorizer()
    
    result = operational_efficiency(
        model=mock_model, 
        vectorizer=mock_vectorizer, 
        X_test=X_test, 
        name="Custom Threshold Test",
        low_thresh=0.4,
        high_thresh=0.8
    )
    
    assert np.isclose(result, 0.5)

# --- Test Plotting Utilities (Mocking Matplotlib using monkeypatch) ---

def test_plot_confusion_matrix(binary_data, monkeypatch):
    """Tests that plot_confusion_matrix calls the correct plotting methods."""
    y_true, y_pred, _ = binary_data
    labels = np.array(['real', 'fake'])
    
    # Patch the plot and show methods using monkeypatch
    mock_plot = MagicMock()
    mock_show = MagicMock()
    mock_title = MagicMock()

    # Use monkeypatch to redirect the calls
    monkeypatch.setattr('sklearn.metrics.ConfusionMatrixDisplay.plot', mock_plot)
    monkeypatch.setattr('matplotlib.pyplot.show', mock_show)
    monkeypatch.setattr('matplotlib.pyplot.title', mock_title)

    plot_confusion_matrix(y_true, y_pred, labels, title="Test CM")
    
    mock_plot.assert_called_once()
    mock_show.assert_called_once()
    mock_title.assert_called_once_with("Test CM")

def test_plot_roc_pr(binary_data, monkeypatch, capsys):
    """Tests that plot_roc_pr calls the correct plotting methods and returns AUC."""
    y_true, _, y_scores = binary_data
    
    # Mocks for plotting components
    mock_figure = MagicMock()
    mock_show = MagicMock()
    mock_roc_plot = MagicMock()
    mock_pr_plot = MagicMock()

    # Use monkeypatch to redirect the calls
    monkeypatch.setattr('matplotlib.pyplot.figure', mock_figure)
    monkeypatch.setattr('matplotlib.pyplot.show', mock_show)
    monkeypatch.setattr('sklearn.metrics.RocCurveDisplay.plot', mock_roc_plot)
    monkeypatch.setattr('sklearn.metrics.PrecisionRecallDisplay.plot', mock_pr_plot)

    # Calculate expected AUC using sklearn directly
    fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label="fake")
    expected_auc = auc(fpr, tpr)

    result_auc = plot_roc_pr(y_true, y_scores, pos_label="fake", title_suffix="Test Model")
    
    # Assert return value is correct
    assert np.isclose(result_auc, expected_auc)
    
    # Check printed output for AUC
    captured = capsys.readouterr()
    assert f"ROC AUC: {expected_auc:.4f}" in captured.out