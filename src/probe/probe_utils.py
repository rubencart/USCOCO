import re
from abc import ABC
from typing import Dict, List, Sequence, Union

from config import ProbeConfig


class ConstituentDictionary(ABC):
    NOT_A_CONSTITUENT = "[NO]"
    PAD = "[PAD]"
    NT_RE = re.compile("NT\((.*)\)")

    dict: Dict[str, int]
    tokens: List[str]

    def __len__(self):
        return len(self.tokens) - 1  # exclude pad so it can't be predicted

    @staticmethod
    def build_tag_dict(cfg: ProbeConfig):
        if cfg.learn_tags:
            return ConstituentTagDictionary(cfg.learn_negative_constituents)
        else:
            return IsConstituentDictionary()

    def encode(
        self, tags: Union[List[str], List[List[str]]], with_NT=False
    ) -> Union[List[int], List[List[int]]]:
        if isinstance(tags[0], str):
            return [self.dict[t] for t in tags]
        else:
            if not isinstance(tags[0], Sequence):
                raise ValueError()
            return [self.encode(lst, with_NT) for lst in tags]

    def decode(
        self, ids: Union[List[int], List[List[int]]], to_NT=False
    ) -> Union[List[str], List[List[str]]]:
        if isinstance(ids[0], int):
            return [self.tokens[i] for i in ids]
        else:
            if not isinstance(ids[0], Sequence):
                raise ValueError()
            return [self.decode(lst, to_NT) for lst in ids]


class IsConstituentDictionary(ConstituentDictionary):
    CONSTITUENT = "[YES]"

    def __init__(self):
        self.tokens = [self.CONSTITUENT, self.NOT_A_CONSTITUENT, self.PAD]
        self.dict = {t: i for i, t in enumerate(self.tokens)}
        self.pad_id = self.dict[self.PAD]


class ConstituentTagDictionary(ConstituentDictionary):
    NT_TAGS = [
        "ADJP",
        "ADVP",
        "CONJP",
        "FRAG",
        "INTJ",
        "LST",
        "NAC",
        "NP",
        "NX",
        "PP",
        "PRN",
        "PRT",
        "QP",
        "RRC",
        "S",
        "SBAR",
        "SBARQ",
        "SINV",
        "SQ",
        "UCP",
        "VP",
        "WHADJP",
        "WHADVP",
        "WHNP",
        "WHPP",
        "X",
    ]

    def __init__(self, learn_negative=False):
        self.tokens = [f"NT({t})" for t in self.NT_TAGS]
        if learn_negative:
            self.tokens += [self.NOT_A_CONSTITUENT]
        self.tokens.append(self.PAD)
        self.dict = {t: i for i, t in enumerate(self.tokens)}
        self.pad_id = self.dict[self.PAD]
