from data.extract_lakefs import ColumnTransformer
import pandas as pd
import dask.dataframe as dd
from typing import List

def test_split_column():
    # Create sample data
    data = {
        'full_name': ['John Doe', 'Jane Smith', 'Bob Wilson'],
    }
    df = pd.DataFrame(data)
    ddf = dd.from_pandas(df, npartitions=2)
    # Initialize transformer
    transformer = ColumnTransformer()

    def name_splitter(full_name: str) -> List[str]:
        if pd.isna(full_name):
            return ['', '']  # Return empty strings for NA values
        print(full_name.split(' '))
        return full_name.split(' ')

    # Test splitting
    df_transformed = (ddf
        .pipe(lambda df: transformer.split_column(df, 'full_name', ['first_name', 'last_name'], name_splitter))
    )

    result = df_transformed.compute()
    print(result)
    # Assertions
    # assert 'first_name' in result.columns
    # assert 'last_name' in result.columns
    # assert result['first_name'].tolist() == ['John', 'Jane', 'Bob']
    # assert result['last_name'].tolist() == ['Doe', 'Smith', 'Wilson'] 


if __name__ == "__main__":
    test_split_column()