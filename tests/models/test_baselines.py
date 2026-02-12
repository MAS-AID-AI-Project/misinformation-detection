import pytest
import numpy as np
import pandas as pd
import json
from unittest.mock import MagicMock, patch

# --- MODEL IMPORTS ---
# The actual classes imported from the package
from misinformation_detection.models import RuleBasedClassifier, MajorityClassBaseline, LogisticRegressionBaseline, DecisionTreeBaseline

# --- MOCKING & EXTERNAL LIB IMPORTS (for speccing mocks only) ---
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

# Mocking evaluation functions (for clean isolation)
expected_cost_metric = MagicMock(return_value=1.5)
operational_efficiency = MagicMock(return_value=0.85)
plot_confusion_matrix = MagicMock()
plot_roc_pr = MagicMock(return_value=0.75)


# --- Fixtures for Test Data (No changes needed) ---

@pytest.fixture
def rule_data_df():
    """DataFrame for testing RuleBasedClassifier logic."""
    data = {
        'source_domain': ['fake.com', 'real.org', 'unknown.com', 'unknown.com', 'fake.com', 'real.org'],
        'word_count': [500, 700, 1000, 300, 638, 637],
        'y_true': ['fake', 'real', 'real', 'fake', 'fake', 'real']
    }
    return pd.DataFrame(data)

@pytest.fixture
def dummy_text_data():
    """Simple text and label data for ML model testing. Majority 'real'."""
    X_train = ["This is real news.", "Fake story about a dog.", "Another true article.", "More real news."]
    y_train = ["real", "fake", "real", "real"] # Majority is 'real' (3 real, 1 fake)
    X_test = ["Test article one", "Test article two"]
    return X_train, y_train, X_test

@pytest.fixture
def tmp_config_path(tmp_path):
    """Creates a temporary path for config files."""
    return tmp_path / "config.json"

@pytest.fixture
def tmp_model_paths(tmp_path):
    """Creates temporary paths for model and vectorizer checkpoints."""
    return tmp_path / "model.joblib", tmp_path / "vectorizer.joblib"


# --- Tests for RuleBasedClassifier ---

def test_rule_classifier_init_defaults():
    """Test initialization with default values."""
    clf = RuleBasedClassifier()
    assert "hollywoodlife.com" in clf.fake_sources
    assert "dailymail.co.uk" in clf.real_sources
    assert clf.length_threshold == 638

def test_rule_classifier_init_custom():
    """Test initialization with custom parameters."""
    clf = RuleBasedClassifier(fake_sources=["bad.net"], length_threshold=100)
    assert "bad.net" in clf.fake_sources
    assert clf.length_threshold == 100

def test_rule_classifier_prediction_logic(rule_data_df):
    """Test prediction logic based on source and length heuristics."""
    clf = RuleBasedClassifier(
        fake_sources=["fake.com"],
        real_sources=["real.org"],
        length_threshold=600  # Custom threshold
    )
    
    # Expected predictions based on rules (threshold 600):
    predictions = clf.predict(rule_data_df)
    expected = pd.Series(["fake", "real", "real", "fake", "fake", "real"])
    pd.testing.assert_series_equal(predictions, expected, check_names=False)

# FIX: Patching the top-level modules now, which is the most reliable way 
# when the specific import path inside the package is unknown or causing issues.
@patch('json.load')
@patch('builtins.open')
def test_rule_classifier_config_save_load_init(mock_open, mock_json_load, tmp_config_path):
    """Test loading classifier configuration during initialization."""
    
    # Configure mock json.load() method
    mock_json_load.return_value = {"fake_sources": ["a.com"], "real_sources": ["dailymail.co.uk"], "length_threshold": 100}
    
    # Initialization of RuleBasedClassifier should trigger json.load
    clf = RuleBasedClassifier(config_path=tmp_config_path)
    
    # Check assertions based on mocked load value
    assert "a.com" in clf.fake_sources
    assert clf.length_threshold == 100
    # Check that mock was called
    mock_json_load.assert_called_once()


# --- Tests for MajorityClassBaseline ---

def test_majority_baseline_fit_predict(dummy_text_data):
    """Test fitting and predicting the majority class."""
    X_train, y_train, X_test = dummy_text_data 
    
    # Test default strategy: most_frequent. Majority is 'real'.
    clf = MajorityClassBaseline()
    clf.fit(X_train, y_train)
    
    # Predict should return 'real' for all test samples
    predictions = clf.predict(X_test)
    assert np.all(predictions == "real")
    assert clf.most_frequent_class() == "real"

@patch('joblib.load')
def test_majority_baseline_load(mock_joblib_load, tmp_model_paths):
    """Test loading a model from disk by checking the mock call."""
    model_path, _ = tmp_model_paths
    
    # Set the return value of the mocked joblib.load
    mock_model = MagicMock(spec=DummyClassifier)
    mock_joblib_load.return_value = mock_model 
    
    # Initialization calls joblib.load
    clf = MajorityClassBaseline(model_path=model_path)
    
    # Assert joblib.load was called once with the correct path
    mock_joblib_load.assert_called_once_with(model_path)
    assert clf.model == mock_model


# --- Tests for LogisticRegressionBaseline and DecisionTreeBaseline ---

# RE-ADDED: @pytest.mark.parametrize is required to loop over the two classes.
@pytest.mark.parametrize("BaselineClass", [LogisticRegressionBaseline, DecisionTreeBaseline])
# FIX: Swapped parameter order in the signature to match the order of decorators.
@patch('joblib.load')
def test_ml_baseline_init_fit_predict(mock_joblib_load, BaselineClass, dummy_text_data):
    """Test ML baselines initialization, fitting, and prediction."""
    X_train, y_train, X_test = dummy_text_data
    
    clf = BaselineClass(vectorizer_params={"max_features": 10}, random_state=42)
    clf.fit(X_train, y_train)
    
    # Test transformation output shape
    X_vec = clf.transform(X_test)
    assert X_vec.shape[0] == len(X_test)
    assert X_vec.shape[1] <= 10 # max_features limit
    
    # Test prediction (check shape/labels only)
    predictions = clf.predict(X_test)
    assert predictions.shape[0] == len(X_test)
    assert all(label in ["real", "fake"] for label in predictions)

@patch('joblib.load')
@patch('joblib.dump')
def test_lr_baseline_save_load(mock_joblib_dump, mock_joblib_load, tmp_model_paths, dummy_text_data):
    """Test persistence for LR model and vectorizer."""
    model_path, vectorizer_path = tmp_model_paths
    
    # 1. Test Save
    clf = LogisticRegressionBaseline(random_state=42)
    clf.fit(dummy_text_data[0], dummy_text_data[1])
    
    # Save calls joblib.dump twice
    clf.save(model_path, vectorizer_path)
    
    # Assert dump was called twice (once for model, once for vectorizer)
    assert mock_joblib_dump.call_count == 2
    
    # 2. Test Load
    mock_lr_model = MagicMock(spec=LogisticRegression)
    mock_vectorizer = MagicMock(spec=TfidfVectorizer)
    
    # Configure joblib.load mock to return the mocks in the correct sequence
    mock_joblib_load.side_effect = [mock_lr_model, mock_vectorizer]
    
    new_clf = LogisticRegressionBaseline()
    new_clf.load(model_path, vectorizer_path)
    
    # Assert load was called twice
    assert mock_joblib_load.call_count == 2
    
    assert new_clf.model == mock_lr_model
    assert new_clf.vectorizer == mock_vectorizer