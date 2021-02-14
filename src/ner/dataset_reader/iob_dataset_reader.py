from typing import Dict, Iterator, List

from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import SequenceLabelField, TextField
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token
from overrides import overrides


@DatasetReader.register('iob_reader')
class IobDatasetReader(DatasetReader):
    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__()
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    @overrides
    def text_to_instance(self, tokens: List[Token], tags: List[str] = None) -> Instance:
        sentence_field = TextField(tokens, self.token_indexers)
        fields = {"tokens": sentence_field}

        if tags:
            label_field = SequenceLabelField(tags, sequence_field=sentence_field)
            fields["tags"] = label_field

        return Instance(fields)

    @overrides
    def _read(self, file_path: str) -> Iterator[Instance]:
        with open(file_path, "r") as f:
            docs = [line.split("\n") for line in f.read().split("\n\n")
                    if line != ""]
            docs = [[l.split("\t") for l in line if l != ""] for line in docs]

        for sent in docs:
            tokens = [Token(t[0]) for t in sent]
            labels = [t[1] for t in sent]

            yield self.text_to_instance(tokens, labels)
