from abc import ABC, abstractmethod
import numpy as np
import dask.dataframe as dd
import pandas as pd
import ast
from typing import Callable,Dict,Optional,Union
import pandas as pd
import dask.dataframe as dd
from typing import Union, Dict, Optional, List

class DaskDataProcess:
    """Abstract base class for data processing tasks."""
    def __init__(self, new_expected_columns: Dict[str, str]):
  
        self.new_expected_columns = new_expected_columns
        self.inferred_dtypes: Dict[str, str] = {}
        
    def infer_meta(
        self,
        default_dtype: str = "object",
        df: Union[pd.DataFrame, dd.DataFrame] = None,
        dask_data_process: Optional["DaskDataProcess"] = None,
        meta: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:

        if meta is not None:
            # Make a copy of the provided meta dictionary.
            inferred_dtypes = meta.dtypes.to_dict()
        elif dask_data_process is not None:
            # Inherit meta from the provided DaskDataProcess instance.
            inferred_dtypes = dask_data_process.inferred_dtypes.copy()
        else:
            # Infer dtypes from the input DataFrame.
            inferred_dtypes = {col: df[col].dtype.name for col in df.columns}

        assert inferred_dtypes is not None
        # Add any new expected columns with default dtype if not already in the inferred meta.
        for col in self.new_expected_columns:
            if col not in inferred_dtypes:
                inferred_dtypes[col] = default_dtype
        
        # Update the instance variable.
        self.inferred_dtypes.update(inferred_dtypes)
        return self._get_meta_pd()
    
    def _get_meta_pd(self) -> pd.DataFrame:

        # Create an empty Series for each column using the correct dtype.
        return pd.DataFrame({col: pd.Series([], dtype=dtype) for col, dtype in self.inferred_dtypes.items()})
    
    def _add_to_meta(self, new_expected_columns: Dict[str, str]) -> pd.DataFrame:

        for col, dtype in new_expected_columns.items():
            if col not in self.inferred_dtypes:
                self.inferred_dtypes[col] = dtype
        return self._get_meta_pd()
    
    def __call__(
        self,
        df: Union[pd.DataFrame, dd.DataFrame] = None,
        meta: Optional[Dict[str, str]] = None,
        dask_data_process: Optional["DaskDataProcess"] = None
    ) -> Dict[str, object]:

        meta_df = self.infer_meta(df = df, meta=meta, dask_data_process=dask_data_process)
        return {"func": self.process, "meta": meta_df}
    
    def process(self, df: Union[pd.DataFrame, dd.DataFrame]) -> Union[pd.DataFrame, dd.DataFrame]:
        raise NotImplementedError("Subclasses should implement this method.")



class ExplodeProcess(DaskDataProcess):
    """Handles exploding list-like columns."""
   
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        column = list(self.new_expected_columns.keys())[0]
        df[column] = df[column].apply(lambda x: list(eval(x)) if isinstance(x, str) else x)
        return df.explode(column)

class ExtractNestedProcess(DaskDataProcess):
    """Handles extracting key-value pairs from a nested dictionary column."""
    def __init__(self, new_expected_columns:dict , nested_column:str):
        super().__init__(new_expected_columns)
        self.nested_column = nested_column

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        # Convert string representations of dictionaries to actual dictionaries
        df[self.nested_column] = df[self.nested_column].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        # Ensure all values are dictionaries
        if not df[self.nested_column].apply(lambda x: isinstance(x, dict)).all():
            raise TypeError(f"Column '{self.nested_column}' must contain dictionaries.")

        # Extract keys
        for key in self.new_expected_columns.keys():
            df[key] = df[self.nested_column].apply(lambda x: x.get(key, None))

        return df


def agregation_function(df: pd.DataFrame) -> List[np.ndarray]:
    user_text = (df["question"] + df["context"]).to_numpy()
    assistant_text = df["text"].to_numpy()
    system_text = np.full_like(user_text, " ")
    return user_text, assistant_text , system_text


class AddConversation(DaskDataProcess):
    """Adds a conversation column to the DataFrame."""
    def __init__(self, agregation_function: Callable[[pd.DataFrame], pd.DataFrame], new_expected_columns: Dict[str, str] = {"conversation": "object"}):
        super().__init__(new_expected_columns)
        self.agregation_function = agregation_function
    
    def process(self, df: pd.DataFrame | dd.DataFrame) -> pd.DataFrame:
        # Concatenate "question" and "context" in a vectorized way.
        user_text, assistant_text = self.agregation_function(df)

        # Build the conversation column using a list comprehension.
        # Each row gets a list of two dictionaries: one for the user and one for the assistant.
        df["conversation"] = [
            [{"role": "user", "content": user}, {"role": "assistant", "content": assist}]
            for user, assist in zip(user_text, assistant_text)
        ]
        return df

class IncludeOnlyCleaner(DaskDataProcess):
    """Keeps only specified columns and drops the rest."""
    def process(self, df: pd.DataFrame | dd.DataFrame) -> pd.DataFrame:
        """Drops columns that are NOT in `columns_to_keep`."""
        return df[list(self.new_expected_columns.keys())] 

    def infer_meta(self,df: pd.DataFrame | dd.DataFrame , **kwargs) -> pd.DataFrame:
        return pd.DataFrame({col: pd.Series(dtype=dtype) for col, dtype in self.new_expected_columns.items()})  
    
    

# get_text_from_answer = ExtractNestedProcess(new_expected_columns={"text": "object"}, nested_column="answers")(df)
# get_text_from_array_text = ExplodeProcess(new_expected_columns={"text": "string"})(meta=get_text_from_answer["meta"])
# get_human = AddConversation(agregation_function=agregation_function)(meta=get_text_from_array_text["meta"])
# new_df = df .map_partitions(**get_text_from_answer)\
#             .map_partitions(**get_text_from_array_text)\
#             .map_partitions(**get_human)\
#             .map_partitions(lambda df: df[["conversation"]])
