from data.extract_lakefs import ColumnTransformer
import pandas as pd
import dask.dataframe as dd
from typing import List

def test_combined_transformations():
    # Create sample data
    data = {
        'full_name': ['John Doe', 'Jane Smith', 'Bob Wilson'],
        'old_id': [1, 2, 3],
        'tags': ['python,data,ml', 'ai,python', 'ml,data']
    }
    df = pd.DataFrame(data)
    ddf = dd.from_pandas(df, npartitions=2)

    # Initialize transformer
    transformer = ColumnTransformer()

    def name_splitter(full_name: str) -> List[str]:
        return full_name.split(' ')

    # Create meta for the expected DataFrame structure after transformations
    meta = pd.DataFrame({
        'user_id': pd.Series(dtype='int64'),
        'first_name': pd.Series(dtype='object'),
        'last_name': pd.Series(dtype='object'),
        'tags': pd.Series(dtype='object')
    })

    # Test combined transformations
    df_transformed = (ddf
        .pipe(transformer.rename_columns, {'old_id': 'user_id'})
        .pipe(lambda df: transformer.split_column(df, 'full_name', ['first_name', 'last_name'], name_splitter, meta=meta))
        .pipe(transformer.explode_column, 'tags', ',')
    )
    result = df_transformed.compute()

    # Assertions
    assert 'user_id' in result.columns
    assert 'first_name' in result.columns
    assert 'last_name' in result.columns
    assert len(result) == 7  # Total number of rows after exploding
    assert set(result['tags'].unique()) == {'python', 'data', 'ml', 'ai'}