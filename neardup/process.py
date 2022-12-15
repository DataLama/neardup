# process functions for hugggingface datasets map.
from typing import List, Dict, Union
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

from datasketch import MinHash, MinHashLSH, LeanMinHash

from .utils import ngrams

def get_minhash(
        data:Dict[str, Union[str, List]], 
        idx:int,
        num_perm:int, 
        seed:int,
        tokenizer:Union[PreTrainedTokenizer, PreTrainedTokenizerFast], 
        input_key:int, 
        n_gram:int, 
        sep:str
    ) -> Dict[str, Union[str, List]]:
    """Shingle the text to token and Calculate the minhash signature.
    
    Args:
        data (Dict[str, Union[str, List]]): The data to be processed in huggingface datasets.map function.
        idx (int): The index of the data in huggingface datasets.map function (must with_indices=True).
        num_perm (int): The number of permutation in minhash.
        seed (int): The seed for minhash.
        tokenizer (Union[PreTrainedTokenizer, PreTrainedTokenizerFast]): The tokenizer to tokenize the text.
        input_key (int): The key of the text in data.
        n_gram (int): The n-gram of the text.
        sep (str): The separator for ngram.
    """
    if isinstance(idx, int):
        m = MinHash(num_perm=num_perm, seed=seed)
        tokens = ngrams(tokenizer.tokenize(data[input_key]), n_gram, sep)
        m.update_batch([token.encode("utf-8") for token in tokens])
        signitures = m.hashvalues
        
    elif isinstance(idx, list):
        signitures = []
        for text in data[input_key]:
            m = MinHash(num_perm=num_perm, seed=seed)
            tokens = ngrams(tokenizer.tokenize(text), n_gram, sep)
            m.update_batch([token.encode("utf-8") for token in tokens])
            signitures.append(m.hashvalues)
            
    return {"id": idx, "signature": signitures}

def inter_query(
        data:Dict[str, Union[str, List]], 
        idx:int,
        index:MinHashLSH, 
        seed:int
    ) -> Dict[str, Union[str, List]]:
    """Query the index and get the neighbors."""
    if isinstance(idx, int):
        neighbors = [dup_idx for dup_idx in index.query(LeanMinHash(seed=seed, hashvalues=data['signature'])) 
                     if dup_idx != data['id']]
    elif isinstance(idx, list):
        neighbors = []
        for _id, s in zip(data['id'], data['signature']):
            neighbors.append([dup_idx for dup_idx in index.query(LeanMinHash(seed=seed, hashvalues=s)) 
                              if dup_idx != _id])
    
    return {"id": data["id"], "neighbors": neighbors}