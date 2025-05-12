from enum import Enum
from datetime import datetime
from datasets import load_dataset, load_dataset_builder
import dask.dataframe as dd
import pandas as pd
from tqdm import tqdm
from loguru import logger
from itertools import islice

from data.utils.lakefs import LakeFsDataset
from data.utils.hugging_face import get_info

def process_and_upload_chunk(ddf, chunk_index,lakefs_client,directory:str):

    path = f"{lakefs_client.path}/{directory}/chunk_{chunk_index}"
    ddf.to_parquet(
        path,
        engine="pyarrow",
        write_metadata_file=True,
        filesystem=lakefs_client.fs,
        overwrite=True
    )

    logger.success(f"Uploaded chunk {chunk_index} to {path}")
    
def stream_and_upload_from_hf_to_lakefs(hf_dataset_name,dataset:LakeFsDataset,split:str=None ,npartitions=2, chunk_size=2000, start=None, end=None,data_dir=None ):
    """
    Stream data from Hugging Face, process it in chunks, and upload it to GCS.
    """
    
    directory = dataset.dataset.get_path()
    lakefs_client = dataset.lakefs_client

    with lakefs_client.fs.transaction(lakefs_client.repo_manager.repo_name, lakefs_client.branch_manager.current_branch) as tx:
        dataset = load_dataset(hf_dataset_name, data_dir=data_dir, split=split, streaming=True,trust_remote_code=True)
        if start is not None and end is not None:
            dataset = islice(dataset, start, end)

        buffer = []
        chunk_index = 0

        for record in tqdm(dataset, desc="Streaming from Hugging Face"):
            buffer.append(record)

            if len(buffer) >= chunk_size:
                df = pd.DataFrame(buffer) 
                ddf = dd.from_pandas(df, npartitions=npartitions)
                process_and_upload_chunk(ddf, chunk_index,lakefs_client,directory+f"/{split}")

                buffer.clear()
                chunk_index += 1

        if buffer:
            df = pd.DataFrame(buffer)
            ddf = dd.from_pandas(df, npartitions=npartitions)
            process_and_upload_chunk(ddf, chunk_index,lakefs_client,directory+f"/{split}")

        tx.commit(f"Uploaded dataset from huggingface {hf_dataset_name} to lakefs in {datetime.now()}")
        logger.success(f"Uploaded dataset from huggingface {hf_dataset_name} to lakefs")
        return {"address":lakefs_client.path + f"/{directory}/{split}"}

    