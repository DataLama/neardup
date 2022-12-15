import emoji
import pandas as pd
from kiwipiepy import Kiwi

def data_gen(fn):
    """generator for bab2min's naver shopping."""
    kiwi = Kiwi()
    text_set = set()
    label_dict = {"O":"중립"}
    df = pd.read_csv(fn)
    df = df.loc[~df.text.duplicated()]
    
    for data in df.to_dict('r'):
        text = emoji.emojize(data['text'])
        for sentence in kiwi.split_into_sents(text, return_sub_sents=False):
            if sentence.text in text_set:
                continue
            else:
                text_set.add(sentence.text)
                yield {
                    "sentence": sentence.text,
                    "label": label_dict[data['label']] if data['label'] == 'O' else data['label']
                }
