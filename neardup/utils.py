import os
from typing import List, Union, Set

from datasets import Dataset
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

import networkit as nk
from tqdm import tqdm

def ngrams(tokens: List[str], ngram: int = 1, sep: str = " ") -> List[str]:
    """Generate ngrams from tokens.
    
    Args:
        tokens: List of tokens
        ngram: ngram size (default = 1)
        sep: seperator of ngram (default = " ")
    
    Usage of 'sep':
        # If you tokenize the tokens with BBPE tokenizer...
        # If you want to convert tokens to string you need seperator.
        # I use "EleutherAI/polyglot-ko-1.3b" for this case.
    
        text = "진짜 존나 여쿨느낌 개좋아용오유유ㅜㅜㅜ"
        tokens = tokenizer.tokenize(x)
        tokens
        >>> ['ì§Ħì§ľ',
             'Ġì¡´',
             'ëĤĺ',
             'ĠìĹ¬',
             'ì¿¨',
             'ëĬĲ',
             'ëĤĮ',
             'Ġê°ľ',
             'ì¢ĭ',
             'ìķĦ',
             'ìļ©',
             'ìĺ¤',
             'ìľł',
             'ìľł',
             'ãħ',
             'ľ',
             'ãħ',
             'ľ',
             'ãħ',
             'ľ']
        
        # bbpe token's ngram
        sep = " "
        my_ngrams = ngrams(tokens, ngram=3, sep)
        my_ngrams
        >>> ['ì§Ħì§ľ Ġì¡´ ëĤĺ',
         'ĠìĹ¬ ì¿¨ ëĬĲ',
         'ëĤĮ Ġê°ľ ì¢ĭ',
         'ìķĦ ìļ© ìĺ¤',
         'ìľł ìľł ãħ',
         'ľ ãħ ľ',
         'ãħ ľ']
        
        # Convert ngrams to human-readable format.
        view_bbpe_ngrams(tokenizer, my_ngrams, sep)
        >>> ['진짜 존나', ' 여쿨느', '낌 개좋', '아용오', '유유�', '�ㅜ', 'ㅜ']
    """
    ngrams = [sep.join(tokens[i : i + ngram]) for i in range(0, len(tokens), ngram)]
    return ngrams


def view_bbpe_ngrams(tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
                     , ngrams: List[str], sep: str) -> List[str]:
    """Convert ByteLevel Ngrams to human-readable format.
    
    Args:
        tokenizer: huggingface's Pretrained Tokenizers.
        ngrams: List of ngrams.
        sep: seperator used in ngram.
    """
    return list(map(lambda x:tokenizer.convert_tokens_to_string(x.split(sep)), my_ngrams))

def batch_iterator(
    dataset: Dataset,
    batch_size: int = 1000,
):
    for i in range(0, len(dataset), batch_size):
        yield dataset[i : i + batch_size]
        

def find_duplicate_components(
    records,
    input_graph: str = None,
    output_graph: str = None,
) -> Set[int]:
    """
    Find the duplicate components in a graph.
    Parameters
    ----------
    records : Iterable | Dataset
        The dataset that contains the neighbors.
    input_graph : str | None, optional
        The path to the input graph, by default None
    output_graph : str | None, optional
        The path to the output graph, by default None
    Returns
    -------
    Set[int]
        The set of duplicate components.
    Examples
    --------
    >>> records = [{"__id__": 0, "__neighbors__": [1]}, {"__id__": 1, "__neighbors__": [0]}]
    >>> find_duplicate_components(records)
    {1}
    """
    if input_graph is not None:
        g = nk.readGraph(str(input_graph), nk.Format.NetworkitBinary)
    else:
        g = nk.graph.Graph()
        for record in tqdm(records, desc="Constructing graph..."):
            for y in record["neighbors"]:
                g.addEdge(record["id"], y, addMissing=True)

        if output_graph is not None:
            if os.path.exists(output_graph):
                os.remove(output_graph)
            nk.writeGraph(g, str(output_graph), nk.Format.NetworkitBinary)

    to_remove: Set[int] = set()
    cc = nk.components.ConnectedComponents(g)
    cc.run()
    for component in tqdm(cc.getComponents(), desc="Iterating over components..."):
        component = sorted(component)
        to_remove.update(component[1:])

    return to_remove