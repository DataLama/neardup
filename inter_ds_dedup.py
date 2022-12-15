import os
import argparse
from hydra import initialize, initialize_config_dir, compose
from omegaconf import OmegaConf
from pathlib import Path

from datasets import Dataset, concatenate_datasets
from transformers import AutoTokenizer

from neardup.build import MinHashIndex
from neardup.utils import find_duplicate_components
from neardup.search import inter_search


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Source data type.",
    )
    args = parser.parse_args()

    # load configs
    with initialize_config_dir(version_base=None, config_dir="/root/neardup/config"):
        config = compose(config_name='config.yaml', overrides=[f'data={args.source}'])
    
    # import data generator
    if args.source=='sbst':
        from scripts.sbst import data_gen

    #### =================== load source dataset ===================
    fn_list = [fn for fn in Path(config[args.source].source_data_dir).glob('*.csv')]

    ds_list = []
    for fn in fn_list:
        ds = Dataset.from_generator(data_gen, gen_kwargs={"fn":fn})
        ds_list.append(ds)
    
    ds = concatenate_datasets(ds_list)
    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model_name_or_path)

    ####

    # build index
    index_builder = MinHashIndex(
        ds=ds,
        input_key=config[args.source].input_key,
        tokenizer=tokenizer,
        sep=config.sep,
        n_gram=config[args.source].n_gram,
        index_name=config[args.source].index_name,
        num_perm=config[args.source].num_perm,
        threshold=config[args.source].threshold,
        seed=config.seed,
    )
    index = index_builder.build_index()

    # inter query get duplicate set
    queried_ds = inter_search(index, index_builder.embedded_ds, config.seed)
    dup_set = find_duplicate_components(queried_ds)

    # filter the duplicate set
    final_data = ds.filter(
        lambda _, idx: idx not in dup_set,
        num_proc=4,
        with_indices=True,
        desc="Filtering duplicates...",
    )

    # save the final data
    os.makedirs(f"{config.data_dir}/inter", exist_ok=True)
    size = final_data.to_parquet(f'{config.data_dir}/inter/dedup_{args.source}.parquet')
    print(f"Final data size: {size/1e+6:.2f}MB.")
    

