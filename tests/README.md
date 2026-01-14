# Pytest Ultimate Guide

Pytest is the leading and recommended testing framework for Python, offering simplicity, a low barrier to entry, and powerful advanced features like fixtures. It is designed to scale from small scripts to complex, multi-module applications.

---

## 1. Installation and Execution

### Installation

Pytest is installed using the standard Python package manager:

```bash
pip install pytest
```

### Running Tests

The simplest way to run tests is by navigating to your project's root directory and executing the command.

| Command | Description |
| :--- | :--- |
| `pytest` | Runs all tests in the current directory and subdirectories. |
| `pytest -v` | **Verbose:** Shows detailed names and results for every test run. |
| `pytest -k "substring"` | **Keyword:** Runs only tests whose names contain the given string (e.g., `pytest -k "data_loader"`). |
| `pytest <filename.py>` | Runs only the tests in the specified file. |
| `pytest --maxfail=1` | **Maximum Failures:** Stops the test session after the first failure. |

---

## 2. Test Discovery and Structure

Pytest automatically finds tests based on strict naming conventions.

### File Structure and Naming

| Component | Naming Convention | Example |
| :--- | :--- | :--- |
| **Test Directory** | Typically named `tests/` at the project root. | `tests/` |
| **Test Files** | Must start with `test_` or end with `_test.py`. | `test_data_loader.py` |
| **Test Functions** | Must start with `test_`. | `def test_data_load_count_is_correct():` |
| **Test Classes** | Class names must start with `Test` (e.g., `TestDataLoader`). | `class TestDataLoader:` |

### Example Test File

```python
# tests/test_model_inference.py

# 1. Imports from your source code
# from src.misinformation_detection.model import predict

# 2. Test function structure
def test_predictor_returns_boolean():
    """Test that the prediction function outputs a boolean."""
    # Arrange: Setup inputs
    text = "This is a news article."
    
    # Act: Execute the function
    # result = predict(text)
    
    # Assert: Check the condition
    # assert isinstance(result, bool)
```

---

## 3. The Power of `assert`

Unlike other frameworks that rely on specific methods (like `assertEqual`), Pytest uses Python's built-in **`assert`** keyword. This simplifies test writing and provides superior failure reporting (a "rich traceback").

| Goal | Standard Python `assert` | Pytest Failure Report |
| :--- | :--- | :--- |
| **Equality** | `assert result == expected` | Shows both the value of `result` and `expected`. |
| **Type Check** | `assert isinstance(obj, list)` | Shows the actual type of `obj`. |
| **Exceptions** | `with pytest.raises(ValueError): func_that_fails()` | Confirms the correct exception type was raised. |
| **Membership** | `assert 'key' in dictionary` | Shows the full contents of the dictionary. |

---

## 4. Fixtures (The Cornerstone of Pytest)

Fixtures are functions that provide necessary resources (data, connections, mock objects, etc.) to test functions. They manage the setup and teardown process, ensuring tests are independent and repeatable.

### Defining and Using Fixtures

1.  **Definition:** Use the `@pytest.fixture` decorator.
2.  **Usage:** The test function simply declares the fixture name as an argument; Pytest handles injecting the fixture's return value.

```python
# Fixture Example
@pytest.fixture
def clean_dataframe():
    """Creates a small, clean DataFrame for use in tests."""
    data = {'col_a': [10, 20], 'col_b': ['x', 'y']}
    return pd.DataFrame(data)

def test_data_sums_to_30(clean_dataframe):
    """The clean_dataframe is automatically passed in."""
    assert clean_dataframe['col_a'].sum() == 30
```

### The `tmp_path` Fixture

Pytest provides built-in fixtures. **`tmp_path`** is essential for data science projects, as it automatically creates a unique temporary directory for each test that uses it, guaranteeing clean, isolated file operations.

### Sharing Fixtures with `conftest.py`

When fixtures are needed by multiple test files, they should be defined in a special file named **`conftest.py`** in the `tests/` directory. Pytest automatically discovers fixtures in this file—you do not need to import them.

---

## 5. Best Practices for AI/ML Projects

1.  **Use Fixtures for Mock Data:** Never load production data in unit tests. Use fixtures to create tiny, representative mock datasets that ensure tests are fast and reproducible.
2.  **Test for Data Shape/Type:** Always use assertions to confirm that function outputs (like a loaded DataFrame or a processed array) have the expected shape, columns, and data types.
3.  **Test Boundary Cases:** Use **parameterization** (via `@pytest.mark.parametrize`) to test a function with many inputs, including edge cases (e.g., empty data, zero values, long strings).
4.  **Isolate Component Tests:** Ensure your unit tests for one module **do not** automatically run another module's code. Use fixtures or mocking techniques to pass in dependencies, ensuring you are testing the unit in isolation.

```python
# Example of testing exceptions (essential for validation)
def test_loader_missing_file_raises_error(tmp_path):
    """
    Tests that the function raises a FileNotFoundError when the data path is invalid.
    """
    # Use tmp_path, which is an empty directory, so files will be missing
    import os
    with pytest.raises(FileNotFoundError):
        load_fakenewsnet_data(tmp_path)
```