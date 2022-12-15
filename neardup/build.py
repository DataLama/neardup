from typing import Generator, Union, Dict
import pickle
import os
from tqdm.auto import tqdm

from datasets import Dataset, load_from_disk
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
from datasketch import MinHash, MinHashLSH, LeanMinHash

from .process import get_minhash
from .utils import batch_iterator

os.environ["TOKENIZERS_PARALLELISM"] = 'false'

class MinHashIndex:
    """MinHash + LSH based index."""
    def __init__(
        self, 
        ds:Dataset, 
        input_key:str,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        sep:str,
        n_gram:int,
        index_name:str,
        num_perm:int,
        threshold:float,
        seed:int,
    ):
        """
        Args:
            
        """
        self.ds = ds
        self.input_key = input_key
        self.tokenizer = tokenizer
        self.sep = sep
        self.n_gram = n_gram
        self.index_name = index_name
        self.num_perm = num_perm
        self.threshold = threshold
        self.seed = seed
        self.embedding_dir = f"cache/{index_name}/embeddings"
        self.index_fn = f"cache/{index_name}/index.pkl"

        os.makedirs(f"cache/{index_name}", exist_ok=True)

        if os.path.exists(self.index_fn):
            with open(self.index_fn, "rb") as f:
                self._index = pickle.load(f)
            self.embedded_ds = load_from_disk(self.embedding_dir)
        else:
            self._index = MinHashLSH(threshold=threshold, num_perm=num_perm)
    
    def build_index(self) -> MinHashLSH:
        """Build the index."""

        if not os.path.exists(self.index_fn):
            # build minhash embeddings.
            self.embedded_ds = self.ds.map(
                get_minhash,
                fn_kwargs=dict(
                    num_perm=self.num_perm,
                    seed=self.seed,
                    tokenizer=self.tokenizer,
                    input_key=self.input_key,
                    n_gram=self.n_gram,
                    sep=self.sep
                ),
                remove_columns=self.ds.column_names,
                num_proc=8,
                with_indices=True, 
                batched=True,
                desc=f"MinHashing..."
            )
            batch_embeddings = batch_iterator(self.embedded_ds)

            # save the embedding_ds.
            self.embedded_ds.save_to_disk(self.embedding_dir)

            # build the index.
            with tqdm(total=len(self.embedded_ds), desc="Build the index ...") as pbar:
                for batch in batch_embeddings:
                    for _id, signature in zip(batch['id'], batch['signature']):
                        self._index.insert(_id, LeanMinHash(seed=self.seed, hashvalues=signature))
                        pbar.update(1)

            # save the index.
            with open(self.index_fn, "wb") as f:
                pickle.dump(self._index, f)

        return self.index

    @property
    def index(self) -> MinHashLSH:
        if len(self._index.keys) == 0:
            raise ValueError("Index is Empty.")
        else:
            return self._index



# class NearDupBuilder:
#     """NearDup Index Builder.
    
#     Use MinHash+LSH algorithm in datasketch and get scalability with redis.
#     """
#     def __init__(
#         self, 
#         ds:Dataset, 
#         input_key:str,
#         tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
#         seed:int
#     ):
#         """
#         Args:
            
#         """
#         self.ds = ds
#         self.input_key = input_key
#         self.tokenizer = tokenizer
#         self.seed = seed
        
#     def process_minhashing(
#         self,
#         num_perm:int,
#         n_gram:int,
#         sep:str
#     ):
#         """Minhash with huggingface's tokenizer"""
#         embedded_ds = self.ds.map(
#             get_minhash,
#             fn_kwargs=dict(
#                 num_perm=num_perm,
#                 seed=self.seed,
#                 tokenizer=self.tokenizer,
#                 input_key=self.input_key,
#                 n_gram=n_gram,
#                 sep=sep
#             ),
#             # input_columns=[self.input_key],
#             # remove_columns=[self.input_key],
#             num_proc=4,
#             with_indices=True, 
#             batched=True,
#             desc=f"MinHashing..."
#         )
#         return embedded_ds
    
#     def build_index(
#         self,
#         threshold:float,
#         num_perm:int,
#         n_gram:int,
#         sep:str,
#         index_name:str,
#         reuse_index:bool,
#         redis_config:Dict = dict()
#     ):
#         # build minhash embeddings.
#         self.embedded_ds = self.process_minhashing(num_perm, n_gram, sep)
#         batch_embedd = batch_iterator(self.embedded_ds)
        
#         # connect to redis
#         if redis_config == dict():
#             raise NotImplementedError(f"No use of redis is not implemented.")
#         else:
#             lsh = MinHashLSH(
#                     threshold=threshold, num_perm=num_perm, storage_config={
#                     'type': 'redis',
#                     'basename': index_name.encode(),
#                     'redis': redis_config,
#                     }
#             )
        
#         # build index
#         if not lsh.is_empty() and reuse_index:
#             print("== Reuse the prebuilted index. ==")
#         else:
#             print("== Build the index. ==")
#             # lsh insert.
#             with lsh.insertion_session() as session:
#                 with tqdm(total=len(self.embedded_ds), desc="Insert to redis ...") as pbar:                    
#                     for batch in batch_embedd:
#                         for _id, signature in zip(batch['id'], batch['signature']):
#                             if _id in lsh:
#                                 continue
#                             session.insert(
#                                 _id,
#                                 LeanMinHash(seed=self.seed, hashvalues=signature),
#                                 check_duplication=False,
#                             )
#                             pbar.update(1)
#         return lsh
    
#     def query(self, index, idx):
#         """Query the minhash index."""
#         return {
#         "neighbors": [
#                 dup_idx
#                 for dup_idx in index.query(
#                     LeanMinHash(seed=self.seed, hashvalues=self.embedded_ds[idx]['signature']),
#                 )
#                 if dup_idx != idx  # exclude itself
#             ],
#             "id": idx,
#         }
            
#     def process_batch_query(self, index):
#         queried_ds = self.embedded_ds.map(
#             get_query,
#             fn_kwargs=dict(
#                 index=index,
#                 seed=self.seed,
#             ),
#             num_proc=4, 
#             remove_columns='signature',
#             with_indices=True,
#             batched=True,
#             desc=f"Batch querying..."
#         )
#         return queried_ds