from typing import List, Optional

from overrides import overrides

from allennlp.data.tokenizers.token_class import Token
from allennlp.data.tokenizers.tokenizer import Tokenizer

try:
    import MeCab
except ImportError:
    print("Please install MeCab")
    raise ImportError


@Tokenizer.register("mecab")
class MecabTokenizer(Tokenizer):
    def __init__(self,
                 dic_path: str = None,
                 user_dic_path: str = None):

        dic_args = ""
        if dic_path is not None:
            dic_args += f' -d {dic_path}'
        if user_dic_path is not None:
            dic_args += f' -u {user_dic_path}'

        print(f'-Owakati{dic_args}')
        self.tokenizer = MeCab.Tagger(f'-Owakati{dic_args}')

    def tokenize(self, text: str) -> List[Token]:
        tokens = self.tokenizer.parse(text).split(' ')
        return [Token(t) for t in tokens]

