import os
from pathlib import Path
import pandas as pd

def load_fakenewsnet_data(
        raw_data_dir, datasets=None, verbose=False
    ):
    """
    Load and combine the FakeNewsNet datasets (GossipCop and PolitiFact).

    Args:
        raw_data_dir (str or Path): Directory containing the CSV files
        [Optional] datasets (dict): Dictionary storing the name (str or Path) of the CSV files storing each component of the dataset
            When not provided, the csv files are assumed to be:
            - gossipcop.csv
            - politifact.csv  
        
    Returns:
        pd.DataFrame: Combined dataframe with columns ['article', 'dataset_source', 'label', ...]
    """

    if datasets is None:
        datasets = {
            "gossipcop": Path("gossipcop.csv"),
            "politifact": Path("politifact.csv"),
        }

    datasets_df = {}

    for name, path in datasets.items():
        df = pd.read_csv(Path(raw_data_dir) / Path(path))
        datasets_df[name] = df

    fnn_df = pd.concat(datasets_df.values(), ignore_index=True)
    if verbose:
        print(f"Loaded {len(fnn_df):,} articles from GossipCop and PolitiFact")
    return fnn_df