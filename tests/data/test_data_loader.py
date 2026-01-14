import os
import pandas as pd
from pathlib import Path
import pytest
import io
from misinformation_detection.data import load_fakenewsnet_data



# ==============================================================================
# PYTEST FIXTURES (Setup Code)
# ==============================================================================

@pytest.fixture
def mock_fakenewsnet_dir(tmp_path: Path):
    """
    A Pytest fixture that creates a temporary directory structure 
    and mock CSV files for testing data loading.
    
    tmp_path is a built-in Pytest fixture that provides a temporary path.
    """
    
    # 1. Define the necessary nested directories based on the project's relative path:
    # "../../raw/FakeNewsNet/" 
    # We create the structure: {tmp_path}/raw/FakeNewsNet/
    fake_raw_dir = tmp_path / "raw" / "FakeNewsNet"
    fake_raw_dir.mkdir(parents=True, exist_ok=True)

    # 2. Define mock data
    gossip_content = (
        "id,article,title\n"
        "1,text_gossip_a,Gossip Title 1\n"
        "2,text_gossip_b,Gossip Title 2\n"
        "3,text_gossip_c,Gossip Title 3\n"
    )
    politifact_content = (
        "id,article,title,source\n"
        "4,text_politi_x,Politi Title 1,NYT\n"
        "5,text_politi_y,Politi Title 2,Fox\n"
    )

    # 3. Write mock data to files
    (fake_raw_dir / "gossipcop.csv").write_text(gossip_content, encoding='utf-8')
    (fake_raw_dir / "politifact.csv").write_text(politifact_content, encoding='utf-8')

    # The fixture returns the directory that load_fakenewsnet_data needs to access
    # We return the path to the 'FakeNewsNet' folder
    return fake_raw_dir


# ==============================================================================
# PYTEST TEST FUNCTIONS
# ==============================================================================

def test_load_fakenewsnet_default_config(mock_fakenewsnet_dir):
    """
    Tests that the function correctly loads and combines the default datasets 
    (gossipcop and politifact).
    """
    
    # ARRANGE
    # The 'mock_fakenewsnet_dir' provides the path to the directory containing the CSVs.
    # In the test environment, this path is passed as 'raw_data_dir'.
    
    # ACT
    combined_df = load_fakenewsnet_data(mock_fakenewsnet_dir)
    
    # ASSERT (1): Check the returned type
    assert isinstance(combined_df, pd.DataFrame)
    
    # ASSERT (2): Check the total number of rows (3 from gossipcop + 2 from politifact)
    assert len(combined_df) == 5, f"Expected 5 rows, but got {len(combined_df)}"
    
    # ASSERT (3): Check that the combined columns are correct (id, article, title, source)
    expected_cols = {'id', 'article', 'title', 'source'}
    # Politifact has 'source'; gossipcop does not. Pandas fills missing cols with NaN.
    assert set(combined_df.columns) == expected_cols
    
    # ASSERT (4): Check data content to ensure concatenation worked
    assert 'text_gossip_a' in combined_df['article'].values
    assert 'text_politi_y' in combined_df['article'].values
    
    # ASSERT (5): Check if NaN values were introduced correctly (gossipcop rows should have NaN in 'source')
    assert combined_df['source'].isna().sum() == 3


def test_load_fakenewsnet_custom_datasets(mock_fakenewsnet_dir):
    """
    Tests that the function can load data when custom datasets are specified.
    In this test, we only ask for 'gossipcop'.
    """
    
    # ARRANGE
    custom_datasets = {
        "gossip_only": Path("gossipcop.csv"),
    }
    
    # ACT
    combined_df = load_fakenewsnet_data(
        raw_data_dir=mock_fakenewsnet_dir, 
        datasets=custom_datasets
    )
    
    # ASSERT (1): Check the total number of rows (only 3 from gossipcop)
    assert len(combined_df) == 3
    
    # ASSERT (2): Check columns are correct for only gossipcop data
    assert set(combined_df.columns) == {'id', 'article', 'title'}
    

def test_load_fakenewsnet_missing_file_raises_error(tmp_path):
    """
    Tests that the function raises an error if a required file is missing.
    We use the clean 'tmp_path' fixture and don't create any files.
    """
    
    # We expect a FileNotFoundError when the function attempts pd.read_csv()
    with pytest.raises(FileNotFoundError):
        # ACT
        # Pass a path that exists, but contains none of the default files
        load_fakenewsnet_data(tmp_path)