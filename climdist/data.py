import pandas as pd

def load(which, readability, heading2):

    """Load the Rigasche Zeitung data.
        which: str, 'main', 'sample'
        readability: bool, include the OCR readability column
        heading2: bool, include the heading2 column"""

    if which == 'main':
        df = pd.read_parquet('../data/processed/RZ_processed.parquet')

        if readability == True:
            df['readability'] = pd.read_parquet('../data/processed/RZ_readability.parquet')
        if heading2 == True:
            df['heading2'] = pd.read_parquet('../data/processed/RZ_heading2.parquet')

    elif which == 'sample':
        df = pd.read_parquet('../data/processed/RZ_sample.parquet')

    return df
