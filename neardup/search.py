
from .process import inter_query

def inter_search(index, embedded_ds, seed):
    queried_ds = embedded_ds.map(
        inter_query,
        fn_kwargs=dict(
            index=index,
            seed=seed,
        ),
        num_proc=4, 
        remove_columns='signature',
        with_indices=True,
        batched=True,
        desc=f"Batch querying..."
    )
    return queried_ds