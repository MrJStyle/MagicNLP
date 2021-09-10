from functools import partial
from pathlib import Path
from typing import Union, List, Dict, Tuple

from numpy import ndarray
from paddle.fluid.reader import DataLoader
from paddlenlp.data import Dict as PDict, Pad, Stack
from paddlenlp.datasets import MapDataset
from paddlenlp.transformers import BertTokenizer


class Data:
    def __init__(self, tag_path: str):
        self.tag_index_map, self.index_tag_map = self.load_label_vocab(tag_path)
        self.tag_nums = len(self.tag_index_map)

    @staticmethod
    def load_label_vocab(path: str):
        tag_index_map = {}
        index_tag_map = {}

        with Path(path).open() as f:
            for index, tag in enumerate(f.readlines()):
                tag = tag.strip("\n")
                tag_index_map[tag] = index
                index_tag_map[index] = tag

        return tag_index_map, index_tag_map

    @staticmethod
    def load_dataset(file_path: Union[str, List[str]]) -> Union[MapDataset, List[MapDataset]]:
        """
        Parameters
        ----------
        file_path: 标注好的数据文件路径
        """
        def read(path):
            with Path(path).open() as f:
                next(f)
                for line in f.readlines():
                    words, labels = line.strip("\n").split("\t")
                    words = words.split("\002")
                    labels = labels.split("\002")
                    yield {"words": words, "labels": labels}

        if isinstance(file_path, str):
            return MapDataset(list(read(file_path)))
        elif isinstance(file_path, List):
            return [MapDataset(list(read(p))) for p in file_path]
        else:
            raise TypeError("Please check type of file_path")

    def tokenize_and_align_labels(
            self,
            example: Dict,
            tokenizer: BertTokenizer,
            no_entity_id: int,
            max_seq_len: int = 128
    ) -> Dict:
        words, labels = example["words"], example["labels"]
        tokenized_input = tokenizer(
            words,
            return_length=True,
            is_split_into_words=True,
            max_seq_len=max_seq_len
        )

        # -2 for [CLS] and [SEP]
        if len(tokenized_input['input_ids']) - 2 < len(labels):
            labels = labels[:len(tokenized_input['input_ids']) - 2]
        labels = [self.tag_index_map[label] for label in labels]
        tokenized_input['labels'] = [no_entity_id] + labels + [no_entity_id]
        tokenized_input['labels'] += [no_entity_id] * (
                len(tokenized_input['input_ids']) - len(tokenized_input['labels'])
        )
        return tokenized_input

    @staticmethod
    def make_batch(examples) -> Tuple[ndarray]:
        return PDict({
            "input_ids": Pad(axis=0, pad_val=0),
            "token_type_ids": Pad(axis=0, pad_val=0),
            "seq_len": Stack(),
            "labels": Pad(axis=0, pad_val=-1)}
        )(examples)

    def make_dataloader(
            self,
            ds: MapDataset,
            tokenizer: BertTokenizer,
            no_entity_id: int,
            max_seq_len: int,
            batch_size: int,
            is_test: bool = False
    ) -> DataLoader:
        trans_func = partial(
            self.tokenize_and_align_labels,
            tokenizer=tokenizer,
            no_entity_id=no_entity_id,
            max_seq_len=max_seq_len,
        )

        ds.map(trans_func)

        return DataLoader(
            dataset=ds,
            batch_size=batch_size,
            collate_fn=self.make_batch,
            shuffle=False if is_test else True
        )