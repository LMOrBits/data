from loguru import logger
from typing import Optional
from datasets import  load_dataset_builder, get_dataset_config_names, DownloadMode, load_dataset, DatasetInfo

def get_info(hf_dataset_name:str,data_dir:Optional[str]=None , config_name:Optional[str]=None):
    config_names = get_dataset_config_names(hf_dataset_name)
    logger.info(f"Config names: {config_names}")
    if len(config_names) == 1:
        default_config_name = config_names[0]
        if config_name is None:
            config_name = default_config_name
        else:
            assert config_name in config_names, f"Config name {config_name} not found in dataset {hf_dataset_name}"
    else:
        assert config_name is not None, f"Dataset has multiple configs, please specify a config name from the following list\n{config_names}"

    builder = load_dataset_builder(hf_dataset_name,
                                   data_dir=data_dir,
                                   trust_remote_code=True,
                                   name=config_name,
                                   )
    return builder.info , config_name



def get_splits(hf_dataset_name:str,data_dir:Optional[str]=None , config_name:Optional[str]=None):
    info, config_name = get_info(hf_dataset_name,data_dir,config_name)
    return info.splits


def get_one_sample(hf_dataset_name:str,data_dir:Optional[str]=None , config_name:Optional[str]=None):
    ds = load_dataset(hf_dataset_name, name=config_name, data_dir=data_dir, trust_remote_code=True)
    samples = {split: ds[split][0] for split in ds.keys()}
    return samples