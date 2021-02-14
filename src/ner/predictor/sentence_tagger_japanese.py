from typing import Dict, List
import json

import numpy
from allennlp.common.util import JsonDict
from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import SequenceLabelField, TextField, FlagField
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor
from overrides import overrides
from src.japanese_tokenizer.mecab_tokenizer import MecabTokenizer


@Predictor.register("mecab_sentence_tagger")
class MecabSentenceTaggerPredictor(Predictor):
    def __init__(
        self,
        model: Model,
        dataset_reader: DatasetReader,
        dic_path: str,
        use_user_dic: bool = False,
        user_dic_path: str = None,
    ) -> None:
        super().__init__(model, dataset_reader)
        self._tokenizer = MecabTokenizer(dic_path, use_user_dic, user_dic_path)

    def predic(self, sentence: str) -> JsonDict:
        return self.predict_json({"sentence": sentence})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        sentence = json_dict["sentence"]
        tokens = self._tokenizer.tokenize(sentence)
        return self._dataset_reader.text_to_instance(tokens)

    @overrides
    def predictions_to_labeled_instances(
        self, instance: Instance, outputs: Dict[str, numpy.ndarray]
    ) -> List[Instance]:
        predicted_tags = outputs["tags"]
        predicted_spans = []

        i = 0
        while i < len(predicted_tags):
            tag = predicted_tags[i]
            # if its a U, add it to the list
            if tag[0] == "U":
                current_tags = [t if idx == i else "O" for idx, t in enumerate(predicted_tags)]
                predicted_spans.append(current_tags)
            # if its a B, keep going until you hit an L.
            elif tag[0] == "B":
                begin_idx = i
                while tag[0] != "L":
                    i += 1
                    tag = predicted_tags[i]
                end_idx = i
                current_tags = [
                    t if begin_idx <= idx <= end_idx else "O"
                    for idx, t in enumerate(predicted_tags)
                ]
                predicted_spans.append(current_tags)
            i += 1

        # Creates a new instance for each contiguous tag
        instances = []
        for labels in predicted_spans:
            new_instance = instance.duplicate()
            text_field: TextField = instance["tokens"]
            new_instance.add_field(
                "tags", SequenceLabelField(labels, text_field), self._model.vocab
            )
            new_instance.add_field("ignore_loss_on_o_tags", FlagField(True))
            instances.append(new_instance)

        return instances

    @overrides
    def dump_line(self, outputs: JsonDict) -> str:
        return json.dumps(outputs, ensure_ascii=False) + '\n'

