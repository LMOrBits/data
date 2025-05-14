
from pathlib import Path
from typing import List, Optional

from loguru import logger

from data.utils.lakefs import LakeFsEmbeding


def dfs_list_files_and_folders(directory_path):
    directory = Path(directory_path)
    if not directory.is_dir():
        raise ValueError(f"The path {directory} is not a valid directory.")

    all_items = []

    def dfs(current_path):
        for item in current_path.iterdir():
            all_items.append(item)
            if item.is_dir():
                dfs(item)

    dfs(directory)
    return all_items


def ingest_data(lakefs_dataset: LakeFsEmbeding ,data_path:Path , commit_message:str = "ingeste vectordb data"):
    """
    Ingest data to lakefs
    """
    repo = lakefs_dataset.repo
    branch = lakefs_dataset.branch
    lakefs_client = lakefs_dataset.lakefs_client
    desired_path = data_path/lakefs_dataset.prefix
    if not desired_path.is_dir():
        raise ValueError(f"The path {desired_path} is does not exist.")
    with lakefs_client.fs.transaction(repo,branch) as tx:
        dirs= {relative_path:relative_path.relative_to(desired_path) for relative_path in dfs_list_files_and_folders(desired_path)}
        for absolute_paths, relative_path in dirs.items():
            logger.info(f"Uploading {absolute_paths} to {relative_path}")
            lakefs_client.fs.put_file(str(absolute_paths) ,f"{lakefs_dataset.repo}/{tx.branch.id}/{lakefs_dataset.prefix}/{relative_path}")
        commit = tx.commit(message=commit_message)
    return commit 

def get_vectordb_data(lakefs_dataset: LakeFsEmbeding, data_path:Path , force:bool = False , commit_hash:Optional[str] = None):
    """
    Get data from lakefs
    """
    lakefs_client = lakefs_dataset.lakefs_client
    # Check if the path is a directory
    if not data_path.is_dir():
        data_path.mkdir(parents=True, exist_ok=True)
    # Check if the path is empty
    if not force and any((data_path/lakefs_dataset.prefix).iterdir()):
        raise ValueError(f"The path {data_path} is not empty. Use force=True to overwrite.")
    if force:
        import shutil
        for item in (data_path/lakefs_dataset.prefix).iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
    address = lakefs_dataset.get_path(id=commit_hash) if commit_hash else lakefs_dataset.get_path()
    lakefs_client.fs.get(address,str(data_path),recursive=True)
    