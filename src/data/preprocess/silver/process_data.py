import os

from pathlib import Path

import dask.dataframe as dd
from loguru import logger
from dask.distributed import Client
from typing import List 
from data.preprocess.text_dataset_cleaners import DatasetCleanerManager

def process_raw_data(
    df_partition: dd.core.DataFrame, dataset_cleaner_manager: DatasetCleanerManager, column_names: List[str]
) -> dd.core.DataFrame:
    # Apply the dataset cleaner manager to all specified columns in parallel
    processed_partition = df_partition.map_partitions(
        lambda df: df.assign(**{col: df[col].apply(dataset_cleaner_manager) for col in column_names}),
        meta={col: 'object' for col in column_names}
    )
    return processed_partition


# def process_data(config: DataProcessingConfig) -> None:
#     logger.info("Processing raw data...")

#     processed_data_save_dir = config.processed_data_save_dir

#     cluster = custom_instantiate(config.dask_cluster)
#     client = Client(cluster)  # type: ignore

#     try:
#         dataset_reader_manager = instantiate(config.dataset_reader_manager)
#         dataset_cleaner_manager = instantiate(config.dataset_cleaner_manager)

#         df = dataset_reader_manager.read_data(config.dask_cluster.n_workers)

#         logger.info("Cleaning data...")
#         df = df.assign(
#             cleaned_text=df.map_partitions(
#                 process_raw_data, dataset_cleaner_manager=dataset_cleaner_manager, meta=("text", "object")
#             )
#         )
#         df = df.compute()

#         train_parquet_path = os.path.join(processed_data_save_dir, "train.parquet")
#         dev_parquet_path = os.path.join(processed_data_save_dir, "dev.parquet")
#         test_parquet_path = os.path.join(processed_data_save_dir, "test.parquet")

#         train_df = df[df["split"] == "train"]
#         dev_df = df[df["split"] == "dev"]
#         test_df = df[df["split"] == "test"]

#         train_df = filter_based_on_minimum_number_of_words(train_df, min_nrof_words=config.min_nrof_words)
#         dev_df = filter_based_on_minimum_number_of_words(dev_df, min_nrof_words=config.min_nrof_words)
#         test_df = filter_based_on_minimum_number_of_words(test_df, min_nrof_words=config.min_nrof_words)

#         train_df.to_parquet(train_parquet_path)
#         dev_df.to_parquet(dev_parquet_path)
#         test_df.to_parquet(test_parquet_path)

#         docker_info = {"docker_image": config.docker_image_name, "docker_tag": config.docker_image_tag}
#         docker_info_save_path = os.path.join(processed_data_save_dir, "docker_info.yaml")

#         write_yaml_file(docker_info_save_path, docker_info)

#         logger.info("Data processing finished!")

#     finally:
#         logger.info("Closing dask client and cluster...")
#         client.close()  # type: ignore
#         cluster.close()


if __name__ == "__main__":
    process_data()
