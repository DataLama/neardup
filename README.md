# NearDup
Unofficial Implementation of NearDup in [Deduplicating Training Data Makes Language Models Better](https://arxiv.org/pdf/2107.06499.pdf) 

## Why?
* NLU 문제를 해결하기 위한 데이터 중복제거는 `MinHash Near Deduplication`이 더 효과적일 것으로 예상됨.
  * 논문에서 실험하지는 못했지만, memorization 능력이 필요한 task (document retrieval or closed-book question answering)은 substring deduplication이 오히려 성능을 하락시킬 수도 있을 것이라고 discuss함.
* HITL(Human in the Loop)을 통한 SL 데이터셋을 구축하는 과정에서 `MinHash Near Deduplication`은 구조적으로 유사한 데이터셋을 제거하여, 좀 더 다양한 형태의 데이터셋을 탐색하는데, 도움이 될 것으로 예상.

## Features
* MinHash Near Deduplication
  * inner-dataset deduplication
  * intra-dataset deduplication

## Structure

---
## Reference
* [text-dedup](https://github.com/ChenghaoMou/text-dedup)
* [ExactSubstr deduplication](https://github.com/google-research/deduplicate-text-datasets)
* [datasketch](https://github.com/ekzhu/datasketch)
