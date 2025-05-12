from data.extract_lakefs import ColumnTransformer
import pandas as pd
import dask.dataframe as dd

def test_explode_column():
    # Create sample data
    data = {
        'full_name': ['John Doe', 'Jane Smith', 'Bob Wilson'],
        'tags': ['python,data,ml', 'ai,python', 'ml,data']
    }
    df = pd.DataFrame(data)
    ddf = dd.from_pandas(df, npartitions=2)

    # Initialize transformer
    transformer = ColumnTransformer()

    # Test exploding
    exploded_df = transformer.explode_column(ddf, 'tags', separator=',')
    result = exploded_df.compute()

    # Assertions
    assert len(result) == 7  # Total number of individual tags
    assert set(result['tags'].unique()) == {'python', 'data', 'ml', 'ai'} 