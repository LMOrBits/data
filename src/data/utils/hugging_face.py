from datasets import  load_dataset_builder


def get_info(hf_dataset_name:str,data_dir:str=None):
    builder = load_dataset_builder(hf_dataset_name,data_dir=data_dir,trust_remote_code=True)
    return builder.info
