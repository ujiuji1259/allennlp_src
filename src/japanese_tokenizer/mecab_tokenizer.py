import os
from typing import List, Optional

from overrides import overrides

from allennlp.data.tokenizers.token_class import Token
from allennlp.data.tokenizers.tokenizer import Tokenizer

try:
    import MeCab
except ImportError:
    print("Please install MeCab")
    raise ImportError

DIC = os.environ.get('MECAB_DIC', '')


@Tokenizer.register("mecab")
class MecabTokenizer(Tokenizer):
    def __init__(self,
                 dic_path: str,
                 use_user_dic: bool = False,
                 user_dic_path: str = None):

        if not use_user_dic:
            self.tokenizer = MeCab.Tagger(f'-Owakati -d {dic_path}')
        else:
            assert user_dic_path is not None, "Please specify user dictionary path if you add user dictionary to system dictionary"
            self.tokenizer = MeCab.Tagger(f'-Owakati -d {dic_path} -u {user_dic_path}')

    def tokenize(self, text: str) -> List[Token]:
        tokens = self.tokenizer.parse(text).split(' ')
        return [Token(t) for t in tokens]

