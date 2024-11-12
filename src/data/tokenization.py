import logging
import pickle
import re
from abc import ABC
from typing import Any, List, Union

import clip
import torch
from config import TextEncoderConfig
from probe import probe_utils
from transformers import AutoTokenizer, GPT2Tokenizer, PreTrainedTokenizer

logger = logging.getLogger("pytorch_lightning")

LENGTH_TOKEN = "<len>"
LEAKY_TOKEN = "<leaky>"


class Tokenizer(ABC):
    tokenizer: Union[PreTrainedTokenizer, Any]

    def __init__(self, cfg: TextEncoderConfig):
        self.len_token_id = -1
        self.leaky_token_id = -1
        self.pad_token_id = -1
        self.bos_token_id = -1
        self.eos_token_id = -1
        self.cfg = cfg
        self.num_words = cfg.num_words

    def __call__(self, text: List[str], **kwargs):
        raise NotImplementedError

    def encode(self, *args, **kwargs):
        return self.tokenizer.encode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    def __len__(self):
        return len(self.tokenizer)

    @classmethod
    def build_tokenizer(cls, cfg: TextEncoderConfig, **kwargs) -> "Tokenizer":
        if cfg.text_encoder in (
            "huggingface",
            "vokenization",
            "visualbert",
            "rp_huggingface",
            "rp_transformer",
            "gpt2_bllip",
        ):
            return HuggingFaceTokenizer(cfg, **kwargs)
        elif cfg.text_encoder == "attn_gan":
            return DamsmTokenizer(cfg)
        elif cfg.text_encoder == "tg":
            return TGTokenizer(cfg, use_tg=True)
        elif cfg.text_encoder == "plm":
            return TGTokenizer(cfg, use_tg=False)
        else:
            assert cfg.text_encoder == "sent_clip", cfg.text_encoder
            return ClipTokenizer(cfg)


class DamsmTokenizer(Tokenizer):
    # CAP_WORD2INDEX = 'cap_word2index.pt'
    # CAP_INDEX2WORD = 'cap_index2word.pt'
    # LABEL_WORD2INDEX = 'label_word2index.pt'
    # LABEL_INDEX2WORD = 'label_index2word.pt'

    def __init__(self, cfg: TextEncoderConfig):
        super().__init__(cfg)

        # path = cfg.text_encoder.attn_gan_text_encoder_path
        # with open(os.path.join(path, self.CAP_WORD2INDEX), 'rb') as fin:
        #     self.cap_word2index = dill.load(fin)
        # with open(os.path.join(path, self.CAP_INDEX2WORD), 'rb') as fin:
        #     self.cap_index2word = dill.load(fin)
        # with open(os.path.join(path, self.LABEL_WORD2INDEX), 'rb') as fin:
        #     self.label_word2index = dill.load(fin)
        # with open(os.path.join(path, self.LABEL_INDEX2WORD), 'rb') as fin:
        #     self.label_index2word = dill.load(fin)
        with open(cfg.attn_gan_vocab_path, "rb") as f:
            self.vocab: Vocabulary = pickle.load(f)

        self.original_length = len(self.vocab)

        # self.cap_word2index[LENGTH_TOKEN] = max(self.cap_word2index.values()) + 1
        # self.cap_index2word[self.cap_word2index[LENGTH_TOKEN]] = LENGTH_TOKEN

        self.pad_token_id = self.vocab("<pad>")
        self.eos_token_id = self.vocab("</s>")
        self.bos_token_id = self.vocab("<s>")

        if cfg.add_len_token:
            self.vocab.add_word(LENGTH_TOKEN)
            self.len_token_id = self.vocab(LENGTH_TOKEN)
        # self.len_token_id = self.cap_word2index[LENGTH_TOKEN]
        # self.eos_token_id = self.cap_word2index["<eos>"]

    def __call__(self, text: List[str], **kwargs):
        tok_captions = []
        for caption in text:
            tok_caption = [self.clean_number(w) for w in caption.strip().lower().split()]
            tok_caption = [self.vocab(token) for token in tok_caption]
            tok_captions.append(torch.tensor(tok_caption, dtype=torch.long))

        input_ids = torch.nn.utils.rnn.pad_sequence(
            tok_captions, batch_first=True, padding_value=self.pad_token_id
        )
        return {
            "input_ids": input_ids,
            "attention_mask": input_ids.ne(self.pad_token_id).long(),
        }

    @staticmethod
    def clean_number(w):
        new_w = re.sub("[0-9]{1,}([,.]?[0-9]*)*", "N", w)
        return new_w

    def __len__(self):
        return len(self.vocab)


class Vocabulary(object):
    """Just for AttnGAN dict, from VPCFG project"""

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx["<unk>"]
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


class ClipTokenizer(Tokenizer):
    def __init__(self, cfg: TextEncoderConfig):
        super().__init__(cfg)
        logger.info("Initializing CLIP tokenizer")

        self.tokenizer = clip.clip._tokenizer
        self.pad_token_id = 0
        self.bos_token_id = self.tokenizer.encoder["<|startoftext|>"]
        self.eos_token_id = self.tokenizer.encoder["<|endoftext|>"]

    def __call__(self, text: List[str], **kwargs):
        """
        :return: format: {'input_ids': padded LongTensor shape BS x L
                          'attention_mask': padded LongTensor shape BS x L with 0 where padding }
        """
        # todo max len
        input_ids = clip.tokenize(text)
        return {
            "input_ids": input_ids,
            "attention_mask": input_ids.ne(self.pad_token_id).long(),
        }

    def __len__(self):
        return len(self.tokenizer.encoder)


class HuggingFaceTokenizer(Tokenizer):
    def __init__(self, cfg: TextEncoderConfig, **kwargs):
        super().__init__(cfg)
        logger.info("Initializing Huggingface tokenizer: %s" % self.cfg.hf_model_name_or_path)

        # if 'roberta' in self.cfg.hf_model_name_or_path:
        #     TokenizerClass = RobertaTokenizer
        # else:
        #     TokenizerClass = AutoTokenizer

        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            (
                self.cfg.hf_tokenizer_model_name_or_path
                if self.cfg.hf_tokenizer_model_name_or_path is not None
                else self.cfg.hf_model_name_or_path
            ),
            cache_dir=self.cfg.cache_dir if self.cfg.cache_dir else None,
            local_files_only=cfg.huggingface_offline,
            **kwargs,
        )

        additional_special_tokens = []
        if cfg.add_len_token:
            additional_special_tokens.append(LENGTH_TOKEN)
        if additional_special_tokens:
            self.tokenizer.add_special_tokens({
                "additional_special_tokens": additional_special_tokens
            })
            logger.info(
                "Added special tokens: %s, %s"
                % (
                    self.tokenizer.additional_special_tokens,
                    self.tokenizer.additional_special_tokens_ids,
                )
            )
            self.len_token_id = (
                self.tokenizer.get_vocab()[LENGTH_TOKEN]
                if LENGTH_TOKEN in additional_special_tokens
                else self.len_token_id
            )
            self.leaky_token_id = (
                self.tokenizer.get_vocab()[LEAKY_TOKEN]
                if LEAKY_TOKEN in additional_special_tokens
                else self.leaky_token_id
            )

        if (
            self.tokenizer.pad_token is None
            and cfg.text_encoder in ("huggingface", "gpt2_bllip")
            and ("gpt2" in cfg.hf_model_name_or_path or cfg.use_llama)
        ):
            logger.info("Huggingface tokenizer: setting pad token")
            self.tokenizer.pad_token = self.tokenizer.eos_token
            # self.config.pad_token_id = self.config.eos_token_id

        self.pad_token_id = self.tokenizer.pad_token_id
        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id

    def __call__(self, text: List[str], **kwargs):
        """
        :param text:
        :param kwargs:
        :return: format: {'input_ids': padded LongTensor shape BS x L
                          'attention_mask': padded LongTensor shape BS x L with 0 where padding }
        """
        return self.tokenizer(
            text,
            truncation=True,  # 'longest_first',
            max_length=self.num_words,
            return_tensors="pt",
            padding=True,
            **kwargs,
        )


class TGTokenizer(Tokenizer):
    """
    When passing a custom tokenizer, for now only works with huggingface tokenizers
        of which the tokenize method
        allows an add_prefix_space argument, e.g. BPE based GPT-2 and RoBERTa
    """

    def __init__(
        self, cfg: TextEncoderConfig, tokenizer: PreTrainedTokenizer = None, use_tg: bool = True
    ):
        super().__init__(cfg)
        logger.info("Initializing TG tokenizer: %s" % self.cfg.architecture)
        self.use_tg = use_tg  # False for PLM

        if tokenizer is None:
            self.tokenizer = GPT2Tokenizer.from_pretrained(
                self.cfg.architecture,
                cache_dir=self.cfg.cache_dir,
                local_files_only=cfg.huggingface_offline,
                # add_prefix_space=True,
            )
        else:
            self.tokenizer = tokenizer

        if self.tokenizer.pad_token is None and isinstance(self.tokenizer, GPT2Tokenizer):
            logger.info("Huggingface tokenizer: setting pad token")
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # specify special tokens
        self.SPECIAL_BRACKETS = ["-LRB-", "-RRB-", "-LCB-", "-RCB-"]

        # specify GEN actions
        # '<|endoftext|>' not included
        self.GEN_VOCAB = (
            self.tokenizer.convert_ids_to_tokens(range(len(self.tokenizer) - 1))
            + self.SPECIAL_BRACKETS
        )

        # specify non-GEN parsing actions
        self.NT_CATS = probe_utils.ConstituentTagDictionary.NT_TAGS
        self.REDUCE = (
            ["REDUCE({})".format(nt) for nt in self.NT_CATS] if self.use_tg else ["REDUCE()"]
        )
        self.ROOT = "[START]"

        self.NT_ACTIONS = ["NT({})".format(cat) for cat in self.NT_CATS]
        self.NT_ACTIONS_SET = set(self.NT_ACTIONS)
        self.NT_ACTIONS2NT_CAT = dict([["NT({})".format(cat), cat] for cat in self.NT_CATS])
        self.ACTIONS_SET = set(
            self.NT_ACTIONS + self.REDUCE
        )  # the set of non-terminal actions and reduce

        self.w_boundary_char = b"\xc4\xa0".decode()

        self.a2str = {}
        for cat in self.NT_CATS:
            a = "NT({})".format(cat)
            self.a2str[a] = "(" + cat

        self.constituent_tokens = (
            self.SPECIAL_BRACKETS + self.NT_ACTIONS + self.REDUCE + [self.ROOT]
        )
        self.num_added_toks = self.tokenizer.add_tokens(self.constituent_tokens)

        self.GEN_ids = self.tokenizer.convert_tokens_to_ids(self.GEN_VOCAB)
        self.NT_ids = self.tokenizer.convert_tokens_to_ids(self.NT_ACTIONS)
        self.REDUCE_ids = self.tokenizer.convert_tokens_to_ids(self.REDUCE)

        self.pad_token_id = self.tokenizer.pad_token_id
        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id

    def __call__(self, text: List[str], **kwargs):
        bs = len(text)
        line_batch = [self.ROOT + " " + line for line in text]
        words_batch = [line.strip().split() for line in line_batch]
        i_tokens_batch = [
            [
                (i, token, token not in self.constituent_tokens)
                for (i, word) in enumerate(words)
                # for token in self.tokenizer.tokenize(word)]
                for token in self.tokenizer.tokenize(word, add_prefix_space=True)
            ]
            for words in words_batch
        ]
        word_idx_batch, tokens_batch, const_mask = zip(*[
            zip(*i_tokens_sample) for i_tokens_sample in i_tokens_batch
        ])

        lengths = torch.tensor([len(tokens) for tokens in tokens_batch])
        batch_max_len = max(lengths)

        # gpt2 uses eos as bos and pad
        pad_token = self.tokenizer.pad_token
        tokens_padded_batch = [
            list(tokens) + [pad_token for _ in range(batch_max_len - len(tokens))]
            for tokens in tokens_batch
        ]
        const_padded_mask = [
            list(c_mask) + [False for _ in range(batch_max_len - len(c_mask))]
            for c_mask in const_mask
        ]
        constituent_mask = torch.tensor(const_padded_mask)

        ids_batch = [
            self.tokenizer.convert_tokens_to_ids(tokens_padded)
            for tokens_padded in tokens_padded_batch
        ]
        input_ids = torch.tensor(ids_batch)

        out_mask = (
            torch.arange(batch_max_len).type_as(lengths).repeat((bs, 1)) < lengths.unsqueeze(1)
        ).int()

        return {
            "input_ids": input_ids,
            "attention_mask": out_mask,
            "tokens": tokens_batch,
            "words": words_batch,
            "token_2_word_idx": list(word_idx_batch),
            "constituent_mask": constituent_mask,
        }


class VokenizationTokenizer(Tokenizer):
    pass
