from enum import Enum
from datetime import datetime
from datasets import load_dataset, load_dataset_builder
import dask.dataframe as dd
import pandas as pd
from tqdm import tqdm
from loguru import logger
from itertools import islice

# Configure logger to show debug messages
logger.remove()
logger.add(lambda msg: print(msg), level="DEBUG")

from data.utils.lakefs import LakeFsDataset
from data.utils.hugging_face import get_info

def process_and_upload_chunk(ddf, chunk_index, lakefs_client,tx, directory: str):
    # Debug: Print column names and their types before processing
    logger.debug(f"Column names before processing: {ddf.columns}")
    logger.debug(f"Column types: {ddf.dtypes}")

    # Convert all column names to strings
    ddf = ddf.rename(columns=lambda x: str(x))
    
    # Debug: Print column names after conversion
    logger.debug(f"Column names after string conversion: {ddf.columns}")

    repo = lakefs_client.repo_manager.repo_name
    
    path = f"{repo}/{tx.branch.id}/{directory}/chunk_{chunk_index}"
    print("---\nUploading chunk to path:", path, "\n---")
    try:
        ddf.to_parquet(
            path,
            engine="pyarrow",
            write_metadata_file=True,
            filesystem=lakefs_client.fs,
            overwrite=True
        )
        logger.success(f"Uploaded chunk {chunk_index} to {path}")
    except Exception as e:
        logger.error(f"Error writing chunk {chunk_index}: {str(e)}")
        # Debug: Print problematic data
        logger.debug(f"First few rows of data:\n{ddf.head()}")
        raise
    
def stream_and_upload_from_hf_to_lakefs(hf_dataset_name, name:str, dataset: LakeFsDataset, split: str=None, npartitions=2, chunk_size=2000, start=None, end=None, data_dir=None):
    """
    Stream data from Hugging Face, process it in chunks, and upload it to GCS.
    """
    
    directory = dataset.dataset.get_path()
    lakefs_client = dataset.lakefs_client
    repo = lakefs_client.repo_manager.repo_name
    branch = lakefs_client.branch_manager.current_branch

    with lakefs_client.fs.transaction(repo,branch) as tx:
        logger.info(f"Loading dataset from Hugging Face {hf_dataset_name} with data_dir {data_dir} and split {split}")
        dataset = load_dataset(hf_dataset_name, name=name, data_dir=data_dir, split=split, streaming=True, trust_remote_code=True)
        if start is not None and end is not None:
            dataset = islice(dataset, start, end)

        buffer = []
        chunk_index = 0

        # Debug: Print first record structure
        for record in tqdm(dataset, desc="Streaming from Hugging Face"):
            if chunk_index == 0 and len(buffer) == 0:
                logger.debug(f"First record structure:\n{record}")
            
            buffer.append(record)

            if len(buffer) >= chunk_size:
                df = pd.DataFrame(buffer)
                # Debug: Print DataFrame info before conversion to dask
                logger.debug(f"DataFrame info before dask conversion:\n{df.info()}")
                
                ddf = dd.from_pandas(df, npartitions=npartitions)
                process_and_upload_chunk(ddf, chunk_index, lakefs_client,tx, directory+f"/{split}")

                buffer.clear()
                chunk_index += 1

        if buffer:
            df = pd.DataFrame(buffer)
            ddf = dd.from_pandas(df, npartitions=npartitions)
            process_and_upload_chunk(ddf, chunk_index, lakefs_client,tx, directory+f"/{split}")

        tx.commit(f"Uploaded dataset from huggingface {hf_dataset_name} , {split} to lakefs in {datetime.now()}")
        logger.success(f"Uploaded dataset from huggingface {hf_dataset_name} to lakefs")
        return {"address": lakefs_client.path + f"/{directory}/{split}"}

    