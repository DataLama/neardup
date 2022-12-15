# This code is for data from (https://github.com/bab2min/corpus)

from kiwipiepy import Kiwi

def data_gen(fn):
    """generator for bab2min's naver shopping."""
    kiwi = Kiwi()
    id2label = ["부정", "부정", "중립", "긍정", "긍정"]
    text_set = set()
    with open(fn) as f:
        for line in f.readlines():
            label, text = line.split('\t')
            for sentence in kiwi.split_into_sents(text.strip(), return_sub_sents=False):
                if sentence.text in text_set:
                    continue
                else:
                    text_set.add(sentence.text)
                    yield {
                        "sentence": sentence.text,
                        "label": id2label[int(label)-1]
                    }
