
from pathlib import Path
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
    if not (data_path/"vectordb").is_dir():
        raise ValueError(f"The path {data_path}/vectordb is does not exist.")
    with lakefs_client.fs.transaction(repo,branch) as tx:
        dirs= {relative_path:relative_path.relative_to(data_path/"vectordb") for relative_path in dfs_list_files_and_folders(data_path/"vectordb")}
        for absolute_paths, relative_path in dirs.items():
            lakefs_client.fs.put_file(str(absolute_paths) ,f"{lakefs_dataset.repo}/{tx.branch.id}/{lakefs_dataset.prefix}/{relative_path}")
        tx.commit(message="Add training data")

def get_vectordb_data(lakefs_dataset: LakeFsEmbeding, data_path:Path , force:bool = False):
    """
    Get data from lakefs
    """
    lakefs_client = lakefs_dataset.lakefs_client
    # Check if the path is a directory
    if not data_path.is_dir():
        data_path.mkdir(parents=True, exist_ok=True)
    # Check if the path is empty
    if not force and any(data_path.iterdir()):
        raise ValueError(f"The path {data_path} is not empty. Use force=True to overwrite.")
    if force:
        import shutil
        for item in data_path.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
    
    lakefs_client.fs.get(lakefs_dataset.get_path(),str(data_path),recursive=True)
    