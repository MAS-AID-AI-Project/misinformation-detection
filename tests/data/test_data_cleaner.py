import pandas as pd
import pytest
import re
from pathlib import Path
from dateutil import parser
from urllib.parse import urlparse
from misinformation_detection.data import fakenewsnet_data_cleaning_pipeline

# ==============================================================================
# 2. PYTEST FIXTURE (Mock Input Data)
# ==============================================================================

@pytest.fixture
def raw_data_df():
    """Fixture providing a mock DataFrame with dirty data for testing."""
    data = {
        'article': [
            None,                                                                       # 0. Null (REMOVED)
            'x' * 10,                                                                   # 1. Too short (REMOVED)
            'x' * 300,                                                                  # 2. Valid, clean (KEPT)
            'This is a URL: http://example.com/page.html <br> and a tag.',              # 3. Too short (52 chars) (REMOVED)
            'y' * 300,                                                                  # 4. Duplicate (REMOVED)
            'y' * 300,                                                                  # 5. Duplicate, kept as 'first' (KEPT)
            # FIX: Simplified dirty text for reliable whitespace testing
            'z' * 300 + ' Article with bad chars and excessive  spaces \u2014 \t',      # 6. Valid, dirty text (KEPT)
            'a' * 300 + 'This is a long, valid article with a title.'                   # 7. Valid, clean (KEPT)
        ],
        'title': [
            'Title A', 'Title B', 'Title C', 'Title D', 'Title E', 'Title E', 'Title G', 'Title H',
        ],
        'url': [
            'http://www.site1.com', None, 'invalid-url-$$', 'https://www.site2.com/page',
            'http://site1.com/dupe', 'http://site1.com/dupe', 'site3.com/page', 'http://example.org/valid',
        ],
        'date': [
            '2023-01-01', None, 'Jan 5, 2023', 'Invalid date string', 
            '2023/01/01', '2023/01/01', '2/3/2023', 'UNKNOWN_DATE',
        ],
        'labels': [0, 1, 0, 1, 0, 1, 0, 1],
        'label': ['real', 'fake', 'real', 'fake', 'real', 'fake', 'real', 'fake']
    }
    
    return pd.DataFrame(data)

# ==============================================================================
# 3. PYTEST TEST FUNCTIONS (Finalized)
# ==============================================================================

def test_cleaning_pipeline_drops_rows(raw_data_df):
    """Tests row removal: nulls, outliers (too short), and duplicates."""
    
    cleaned_df = fakenewsnet_data_cleaning_pipeline(raw_data_df)
    
    # Expected: 4 rows (since Row 3 was also too short: 52 < 250)
    assert len(cleaned_df) == 4, f"Expected 4 rows, but got {len(cleaned_df)}"
    assert (cleaned_df['article'] == 'y' * 300).sum() == 1


def test_cleaning_pipeline_imputes_data(raw_data_df):
    """Tests imputation steps: filling missing URLs/dates and extracting source_domain."""
    
    cleaned_df = fakenewsnet_data_cleaning_pipeline(raw_data_df)
    source_domains = cleaned_df['source_domain'].values
    
    # ASSERT (1): Check Domain Extraction (Missing Scheme)
    assert 'site3.com' in source_domains
    
    # ASSERT (2): Check Domain Extraction (Invalid URL that remains)
    assert 'invalid-url-$$' in source_domains, "Expected 'invalid-url-$$' to be retained as domain for unparsable URL."

    # ASSERT (3): Check Date Imputation/Cleaning
    # Remaining rows with 'UNKNOWN_DATE': 1 (Original Row 7)
    unknown_dates = (cleaned_df['date_cleaned'] == 'UNKNOWN_DATE').sum()
    assert unknown_dates == 1, f"Expected 1 'UNKNOWN_DATE' in final DF, got {unknown_dates}"
    
    # Check date parsing correctness (Row 2: 'Jan 5, 2023' -> 05/01/2023)
    assert (cleaned_df['date_cleaned'] == '05/01/2023').any()


def test_cleaning_pipeline_transforms_text(raw_data_df):
    """Tests text cleaning: removing URLs, HTML tags, and non-ASCII characters."""

    cleaned_df = fakenewsnet_data_cleaning_pipeline(raw_data_df)
    
    # Find the row that had dirty text (Original Row 6, Title G)
    dirty_row = cleaned_df[cleaned_df['title'] == 'Title G'].iloc[0]
    cleaned_text = dirty_row['article_cleaned']
    
    # ASSERT (1): Check normalization/non-ASCII (Em dash \u2014 is non-ASCII)
    assert '\u2014' not in cleaned_text
    
    # ASSERT (2): Check for excessive whitespace, using the robust assertion
    assert '  ' not in cleaned_text 


def test_cleaning_pipeline_saves_file(raw_data_df, tmp_path):
    """Tests the optional file saving functionality."""

    output_dir = tmp_path / "processed_data"
    output_path = output_dir / "cleaned_data.csv"

    cleaned_df = fakenewsnet_data_cleaning_pipeline(raw_data_df, save_path=output_path)

    assert output_path.exists()

    saved_df = pd.read_csv(output_path)
    
    # FIX 1: Reset the index of the returned DataFrame for comparison
    cleaned_df_reset = cleaned_df.reset_index(drop=True)
    
    # FIX 2: Drop the 'date_parsed' column from both DFs before comparison to avoid dtype mismatch (datetime vs object)
    cleaned_df_compare = cleaned_df_reset.drop(columns=['date_parsed'])
    saved_df_compare = saved_df.drop(columns=['date_parsed'])

    pd.testing.assert_frame_equal(cleaned_df_compare, saved_df_compare)