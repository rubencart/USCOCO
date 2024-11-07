import itertools
import json
import logging
import random
from abc import ABC
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, cast

import h5py
import torch
from config import Config
from data.singleton import S
from data.tokenization import TGTokenizer, Tokenizer
from data.tree_datasets import CocoNotatedSyntaxTreeDataset
from data.tree_utils import (
    extract_spans_and_tags,
    extract_spans_and_tags_other_tokenizer,
    extract_spans_and_tags_plm_tokenizer,
    extract_spans_and_tags_rb_tokenizer,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset

from probe import probe_utils

logger = logging.getLogger("pytorch_lightning")


class CocoProbeDataset(Dataset, ABC):
    h5_ds: Optional[h5py.Dataset] = None
    h5_file: Optional[h5py.File] = None

    def __init__(
        self,
        cfg: Config,
        sentences_path: str,
        split: str,
        negative_constituents: Optional[Literal["all", "as_positives"]] = None,
    ):
        self.cfg = cfg
        self.h5_path = cfg.probe.h5_path
        self.h5_index_path = cfg.probe.h5_index_path
        self.sentences_path = sentences_path
        self.split = split
        self.prefix = {
            "train": "train",
            "newval": "new_val",
            "val": "test_indom",
            "absurd": "test_abs",
        }[split]
        self.prefix_4_ds = self.prefix
        self.negative_constituents = (
            negative_constituents
            if negative_constituents
            else self.cfg.probe.learn_which_negative_constituents
        )

        with open(self.h5_index_path, "r") as f:
            self.h5_index = json.load(f)
        self.caption_ids = list(self.h5_index[self.prefix].keys())

        self.syntax_ds = CocoNotatedSyntaxTreeDataset(
            cfg, sentences_path, use_tg=True, index_by_capt_id=True, right_branching=False
        )
        self.tag_dict = probe_utils.ConstituentDictionary.build_tag_dict(cfg.probe)
        self.layer = cfg.probe.current_layer
        logger.info("Initializing Probe dataset for layer %s" % self.layer)
        self.before_proj = False
        if self.layer > cfg.detr.encoder_layers:
            logger.info("Initializing Probe dataset for layer before projection to encoder dim")
            self.before_proj = True
            self.prefix_4_ds += "_inp"
        with h5py.File(self.h5_path, "r", swmr=False, libver="latest") as f:
            self.h5_ds_shape = f[self.prefix_4_ds].shape

    def init_h5(self, prefix: str):
        logger.info("Initializing h5")
        self.h5_file = h5py.File(self.h5_path, "r", swmr=False, libver="latest")
        self.h5_ds = self.h5_file[prefix]

    def __getitem__(self, item: int):  # -> ProbeSample:
        if self.h5_file is None:
            self.init_h5(self.prefix_4_ds)

        capt_id = self.caption_ids[item]
        h5_idx, n_tokens = self.h5_index[self.prefix][capt_id]

        embs = cast(h5py.Dataset, self.h5_ds)[h5_idx]  # .transpose(0, 1)
        if not self.before_proj:
            embs = embs[self.layer]
        embs = embs[:n_tokens]

        syntax = self.syntax_ds[int(capt_id)][0]

        return {
            "caption_id": capt_id,
            "tg_caption": syntax["caption"],
            "n_tokens": n_tokens,
            "embeddings": torch.from_numpy(embs),  # .unsqueeze(0),
            "image_id": syntax["image_id"],
            # 'tree_actions': ' '.join(syntax['actions']),
            "tree_actions": syntax["actions"],
        }

    def __del__(self):
        if self.h5_file is not None:
            logger.info("Closing h5")
            self.h5_file.close()

    def __len__(self):
        # return len(self.h5_ds)
        return len(self.h5_index[self.prefix])

    def collate(self, list_of_samples: List[Dict[str, Any]]):
        pass

    def get_spans_and_tags(self, tg_tokens, other_tokens, constituent_mask=None):
        spans_and_tags = self.tree_to_span_idcs_tags(tg_tokens, other_tokens, constituent_mask)

        l, r, tags = zip(*[zip(*sample) for sample in spans_and_tags])
        spans = tuple(tuple(zip(ls, rs)) for ls, rs in zip(l, r))

        if not self.cfg.probe.learn_tags:
            tags = [[self.tag_dict.CONSTITUENT for _ in t] for t in tags]

        if self.cfg.probe.learn_negative_constituents:
            spans, tags = self.append_negative_spans(spans, tags, other_tokens)

        return spans, tags

    def append_negative_spans(self, spans, tags, tokens):
        new_spans, new_tags = [], []
        for sp, t, tkns in zip(spans, tags, tokens):
            nb = -1 if self.negative_constituents == "all" else len(sp)
            neg = self.get_negative_span_idcs(sp, tkns, nb=nb)
            new_spans.append(list(sp) + neg)
            new_tags.append(list(t) + [self.tag_dict.NOT_A_CONSTITUENT for _ in range(len(neg))])
        return new_spans, new_tags

    def get_negative_span_idcs(
        self, spans: Tuple[Tuple[int, int]], tokens: Tuple[str], nb=-1
    ) -> List[Tuple[int, int]]:
        length = max(r for (_, r) in spans)
        # combinations = list(itertools.combinations(range(length + 1), r=2))
        combinations = list(itertools.combinations(range(length), r=2))
        # exclude terminal spans
        filtered = self.filter_negative_span_idcs(combinations, spans, tokens)

        # print('POS')
        # print('\n'.join([f'{(l,r)} {tokens[l:r]}' for (l, r) in spans]))
        # print('NEG')
        # print('\n'.join([f'{(l,r)} {tokens[l:r]}' for (l, r) in filtered]))

        if nb > 0:
            filtered = random.sample(filtered, k=min(nb, len(filtered)))
        return filtered

    def tree_to_span_idcs_tags(self, tg_tokens, other_tokens, constituent_mask=None):
        raise NotImplementedError

    def filter_negative_span_idcs(
        self, idcs: List[Tuple[int, int]], positives: Tuple[Tuple[int, int]], tokens: Tuple[str]
    ) -> List[Tuple[int, int]]:
        raise NotImplementedError

    @staticmethod
    def _even_nb_of_id_reduces(tokens: Tuple[str, ...]) -> bool:
        right_red, idx = tokens[-1], -1
        while tokens[idx - 1] == right_red:
            idx -= 1
        return idx % 2 == 0


class TGCocoProbeDataset(CocoProbeDataset):
    def __init__(
        self,
        cfg: Config,
        tokenizer: Tokenizer,
        # h5_path: str,
        # h5_index_path: str,
        sentences_path: str,
        split: str,
        # num_words: int,
        negative_constituents: Optional[Literal["all", "as_positives"]] = None,
    ):
        super().__init__(
            cfg,
            # cfg.probe.h5_path,
            # cfg.probe.h5_index_path,
            sentences_path,
            split,
            negative_constituents,
        )
        # self.tokenizer = tokenizer
        if not isinstance(tokenizer, TGTokenizer):
            raise TypeError
        self.tg_tokenizer = tokenizer

        # def __getitem__(self, item: int):  # -> ProbeSample:

    #     capt_id, n_tokens, embs = super().__getitem__(item)

    def collate(self, list_of_samples: List[Dict[str, Any]]) -> Dict:
        """
        Span indexes into embeddings[constituent_mask]!
        """
        captions, img_ids, capt_ids, tree_actions, embeddings, n_tokens = zip(
            *(
                (
                    s["tg_caption"],
                    s["image_id"],
                    s["caption_id"],
                    s["tree_actions"],
                    s["embeddings"],
                    s["n_tokens"],
                )
                for s in list_of_samples
            )
        )

        tokenized = self.tg_tokenizer([" ".join(list(ta)) for ta in tree_actions])
        spans, tags = self.get_spans_and_tags(
            tokenized["tokens"], tokenized["tokens"], constituent_mask=tokenized["constituent_mask"]
        )
        tags = self.tag_dict.encode(tags, with_NT=True)

        # assert torch.equal(n_tokens.int(), tokenized['attention_mask'].int().sum(-1))
        pad_fn = partial(nn.utils.rnn.pad_sequence, batch_first=True)
        result = {
            "captions": list(captions),
            "img_ids": list(img_ids),
            "caption_ids": list(capt_ids),
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "constituent_mask": tokenized["constituent_mask"],
            "embeddings": pad_fn(embeddings, padding_value=0.0),
            "spans": spans,
            "tags": tags,
        }
        return result

    def tree_to_span_idcs_tags(self, _, tokens, constituent_mask=None):
        sp_and_tags = [
            extract_spans_and_tags(t, include_parens=self.cfg.probe.tg_include_parens)
            for t in tokens
        ]
        # not necessary anymore since const tokens get attention weight 0 in probe_model.py!
        # sp_and_tags = [tree_utils.filter_non_words(s, constituent_mask[i, :len(t)])
        #                for i, (s, t) in enumerate(zip(sp_and_tags, tg_tokens))]
        return sp_and_tags

    def filter_negative_span_idcs(
        self, idcs: List[Tuple[int, int]], positives: Tuple[Tuple[int, int]], tokens: Tuple[str]
    ):
        return [
            (l, r)
            for (l, r) in idcs
            if (
                (l, r) not in positives
                and r - l > 1
                and "[START]" not in tokens[l]
                and not self.cfg.probe.tg_harder_negatives
                or (
                    (
                        not self.cfg.probe.tg_include_parens
                        and "NT(" not in tokens[l]
                        and "REDUCE(" not in tokens[r - 1]
                    )
                    or (
                        self.cfg.probe.tg_include_parens
                        and "NT(" in tokens[l]
                        and "REDUCE(" in tokens[r - 1]
                        and self._even_nb_of_id_reduces(tokens[:r])
                    )
                )
            )
        ]


class LMCocoProbeDataset(CocoProbeDataset):
    def __init__(
        self,
        cfg: Config,
        tokenizer: Tokenizer,
        sentences_trees_path: str,
        split: str,
        negative_constituents: Optional[Literal["all", "as_positives"]] = None,
    ):
        super().__init__(
            cfg,
            sentences_trees_path,
            split,
            negative_constituents,
        )
        self.tokenizer = tokenizer
        # give this class' tokenizer to TG tokenizer so tokens between
        # constituent labels (so words) get tokenized
        #   by this class' tokenizer (so they can later be matched to
        #   the tokenized sentence by this class' tokenizer
        #   to extract constituents and their tags)
        self.tg_tokenizer = TGTokenizer(cfg.text_encoder, self.tokenizer.tokenizer)

        self.image_dir, self.captions_path = {
            "train": (cfg.train_image_dir, cfg.train_captions),
            "newval": (cfg.train_image_dir, cfg.train_captions),
            "val": (cfg.val_image_dir, cfg.val_captions),
            "absurd": (cfg.val_image_dir, cfg.uscoco_captions),
        }[split]
        self.captions_ds = {
            "train": S.TrainCaptions,
            "newval": S.TrainCaptions,
            "val": S.ValCaptions,
            "absurd": S.AbsurdCaptions,
        }[split](self.image_dir, self.captions_path)

    def __getitem__(self, item: int):  # -> ProbeSample:
        super_item = super().__getitem__(item)

        ann = self.captions_ds.coco.loadAnns([int(super_item["caption_id"])])[0]
        caption = ann["caption"]
        # because we do this in data_module.py, models are trained like this
        caption = caption.strip().rstrip(".")

        return {
            **super_item,
            "caption": caption,
        }

    def collate(self, list_of_samples: List[Dict[str, Any]]) -> Dict:
        """
        spans indexes into embeddings straight, not into embeddings[constituent_mask]!
        """
        captions, tg_captions, img_ids, capt_ids, tree_actions, embeddings, n_tokens = zip(
            *(
                (
                    s["caption"],
                    s["tg_caption"],
                    s["image_id"],
                    s["caption_id"],
                    s["tree_actions"],
                    s["embeddings"],
                    s["n_tokens"],
                )
                for s in list_of_samples
            )
        )
        tg_tokenized = self.tg_tokenizer([" ".join(list(ta)) for ta in tree_actions])
        # tok_tg_capt = self.tokenizer(list(tg_captions))
        tokenized = self.tokenizer(list(captions))
        tokens = [self.tokenizer.tokenizer.tokenize(caption) for caption in captions]
        spans, tags = self.get_spans_and_tags(tg_tokenized["tokens"], tokens)
        tags = self.tag_dict.encode(tags, with_NT=True)

        pad_fn = partial(nn.utils.rnn.pad_sequence, batch_first=True)
        result = {
            "captions": list(captions),
            "img_ids": list(img_ids),
            "caption_ids": list(capt_ids),
            "embeddings": pad_fn(embeddings, padding_value=0.0),
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "constituent_mask": tg_tokenized["constituent_mask"],
            "spans": spans,
            "tags": tags,
        }
        return result

    def tree_to_span_idcs_tags(self, tg_tokens, other_tokens, constituent_mask=None):
        return [
            extract_spans_and_tags_other_tokenizer(tg, t) for tg, t in zip(tg_tokens, other_tokens)
        ]

    def filter_negative_span_idcs(
        self, idcs: List[Tuple[int, int]], positives: Tuple[Tuple[int, int]], tokens: Tuple[str]
    ):
        return [(l, r) for (l, r) in idcs if ((l, r) not in positives and r - l > 1)]


class PLMCocoProbeDataset(CocoProbeDataset):
    def __init__(
        self,
        cfg: Config,
        tokenizer: Tokenizer,
        sentences_trees_path: str,
        split: str,
        negative_constituents: Optional[Literal["all", "as_positives"]] = None,
    ):
        super().__init__(
            cfg,
            sentences_trees_path,
            split,
            negative_constituents,
        )
        self.tokenizer = tokenizer
        # give this class' tokenizer to TG tokenizer so tokens between
        # constituent labels (so words) get tokenized
        #   by this class' tokenizer (so they can later be matched to
        #   the tokenized sentence by this class' tokenizer
        #   to extract constituents and their tags)
        self.tg_tokenizer = TGTokenizer(cfg.text_encoder, use_tg=True)
        self.captions_ds = CocoNotatedSyntaxTreeDataset(
            cfg, sentences_trees_path, use_tg=False, index_by_capt_id=True, right_branching=False
        )

    def __getitem__(self, item: int):  # -> ProbeSample:
        super_item = super().__getitem__(item)

        syntax = self.captions_ds[int(super_item["caption_id"])][0]

        return {
            # 'caption_id': capt_id,
            # 'tg_caption': syntax['caption'],
            # 'n_tokens': n_tokens,
            # 'embeddings': torch.from_numpy(embs),  # .unsqueeze(0),
            # 'image_id': syntax['image_id'],
            # 'tree_actions': syntax['actions'],
            **super_item,
            "plm_caption": syntax["caption"],
            "plm_tree_actions": syntax["actions"],
        }

    def collate(self, list_of_samples: List[Dict[str, Any]]) -> Dict:
        """
        spans indexes into embeddings straight, not into embeddings[constituent_mask]!
        """
        (
            captions,
            tg_captions,
            img_ids,
            capt_ids,
            tg_tree_actions,
            plm_tree_actions,
            embeddings,
            n_tokens,
        ) = zip(
            *(
                (
                    s["plm_caption"],
                    s["tg_caption"],
                    s["image_id"],
                    s["caption_id"],
                    s["tree_actions"],
                    s["plm_tree_actions"],
                    s["embeddings"],
                    s["n_tokens"],
                )
                for s in list_of_samples
            )
        )
        tg_tokenized = self.tg_tokenizer([" ".join(list(ta)) for ta in tg_tree_actions])
        plm_tokenized = self.tokenizer([" ".join(list(ta)) for ta in plm_tree_actions])
        # tokens = [self.tokenizer.tokenizer.tokenize(caption) for caption in captions]
        spans, tags = self.get_spans_and_tags(tg_tokenized["tokens"], plm_tokenized["tokens"])
        tags = self.tag_dict.encode(tags, with_NT=True)

        pad_fn = partial(nn.utils.rnn.pad_sequence, batch_first=True)
        result = {
            "captions": list(captions),
            "img_ids": list(img_ids),
            "caption_ids": list(capt_ids),
            "embeddings": pad_fn(embeddings, padding_value=0.0),
            "input_ids": plm_tokenized["input_ids"],
            "attention_mask": plm_tokenized["attention_mask"],
            "constituent_mask": plm_tokenized["constituent_mask"],
            "spans": spans,
            "tags": tags,
        }
        return result

    def tree_to_span_idcs_tags(self, tg_tokens, other_tokens, constituent_mask=None):
        return [
            extract_spans_and_tags_plm_tokenizer(
                tg, include_parens=self.cfg.probe.tg_include_parens
            )
            for tg in tg_tokens
        ]

    def filter_negative_span_idcs(
        self, idcs: List[Tuple[int, int]], positives: Tuple[Tuple[int, int]], tokens: Tuple[str]
    ):
        return [
            (l, r)
            for (l, r) in idcs
            if (
                (l, r) not in positives
                and r - l > 1
                and "[START]" not in tokens[l]
                and not self.cfg.probe.tg_harder_negatives
                or (
                    (
                        not self.cfg.probe.tg_include_parens
                        and "NT(" not in tokens[l]
                        and "REDUCE(" not in tokens[r - 1]
                    )
                    or (
                        self.cfg.probe.tg_include_parens
                        and "NT(" in tokens[l]
                        and "REDUCE(" in tokens[r - 1]
                    )
                )
            )
        ]


class RBCocoProbeDataset(CocoProbeDataset):
    def __init__(
        self,
        cfg: Config,
        tokenizer: Tokenizer,
        sentences_trees_path: str,
        split: str,
        negative_constituents: Optional[Literal["all", "as_positives"]] = None,
    ):
        super().__init__(
            cfg,
            sentences_trees_path,
            split,
            negative_constituents,
        )
        self.tg_tokenizer = tokenizer
        self.right_branch_ds = CocoNotatedSyntaxTreeDataset(
            cfg, sentences_trees_path, use_tg=True, right_branching=True, index_by_capt_id=True
        )

    def __getitem__(self, item: int):  # -> ProbeSample:
        super_item = super().__getitem__(item)
        # capt_id = self.caption_ids[item]

        rb_syntax = self.right_branch_ds[int(super_item["caption_id"])][0]

        return {
            **super_item,
            "rb_caption": rb_syntax["caption"],
            "rb_tree_actions": rb_syntax["actions"],
        }

    def collate(self, list_of_samples: List[Dict[str, Any]]) -> Dict:
        """
        spans indexes into embeddings straight, not into embeddings[constituent_mask]!
        """
        captions, img_ids, capt_ids, tree_actions, rb_tree_actions, embeddings, n_tokens = zip(
            *(
                (
                    s["rb_caption"],
                    s["image_id"],
                    s["caption_id"],
                    s["tree_actions"],
                    s["rb_tree_actions"],
                    s["embeddings"],
                    s["n_tokens"],
                )
                for s in list_of_samples
            )
        )
        # only for supervision
        tg_tokenized = self.tg_tokenizer([" ".join(list(ta)) for ta in tree_actions])

        rb_tokenized = self.tg_tokenizer([" ".join(list(ta)) for ta in rb_tree_actions])

        spans, tags = self.get_spans_and_tags(tg_tokenized["tokens"], rb_tokenized["tokens"])
        tags = self.tag_dict.encode(tags, with_NT=True)

        pad_fn = partial(nn.utils.rnn.pad_sequence, batch_first=True)
        result = {
            "captions": list(captions),
            "img_ids": list(img_ids),
            "caption_ids": list(capt_ids),
            "embeddings": pad_fn(embeddings, padding_value=0.0),
            "input_ids": rb_tokenized["input_ids"],
            "attention_mask": rb_tokenized["attention_mask"],
            "constituent_mask": rb_tokenized["constituent_mask"],
            "spans": spans,
            "tags": tags,
        }
        return result

    def tree_to_span_idcs_tags(self, tg_tokens, rb_tokens, constituent_mask=None):
        sp_and_tags = [
            extract_spans_and_tags_rb_tokenizer(
                tg, rb, include_parens=self.cfg.probe.tg_include_parens
            )
            for tg, rb in zip(tg_tokens, rb_tokens)
        ]
        # not necessary anymore since const tokens get attention weight 0 in probe_model.py!
        # sp_and_tags = [tree_utils.filter_non_words(s, constituent_mask[i, :len(t)])
        #                for i, (s, t) in enumerate(zip(sp_and_tags, tg_tokens))]
        return sp_and_tags

    def filter_negative_span_idcs(
        self, idcs: List[Tuple[int, int]], positives: Tuple[Tuple[int, int]], tokens: Tuple[str]
    ):
        return [
            (l, r)
            for (l, r) in idcs
            if (
                (l, r) not in positives
                and r - l > 1
                and "[START]" not in tokens[l]
                and not self.cfg.probe.tg_harder_negatives
                or (
                    (
                        not self.cfg.probe.tg_include_parens
                        and "NT(" not in tokens[l]
                        and "REDUCE(" not in tokens[r - 1]
                    )
                    or (
                        self.cfg.probe.tg_include_parens
                        and "NT(" in tokens[l]
                        and "REDUCE(" not in tokens[r - 1]
                        # and 'REDUCE(' in tokens[r - 1]
                        # and self._even_nb_of_id_reduces(tokens[:r])
                    )
                )
            )
        ]


def train_dataloader(ds: CocoProbeDataset, cfg: Config) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=cfg.probe.batch_size,
        pin_memory=cfg.cuda and cfg.pin_memory,
        drop_last=False,
        num_workers=cfg.num_workers if not cfg.debug else cfg.debug_num_workers,
        collate_fn=ds.collate,
        # **kwargs,
        shuffle=True,
    )


def inference_dataloader(ds: CocoProbeDataset, cfg: Config) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=cfg.probe.batch_size,
        pin_memory=cfg.cuda and cfg.pin_memory,
        num_workers=cfg.num_workers if not cfg.debug else cfg.debug_num_workers,
        drop_last=False,
        collate_fn=ds.collate,
        # **kwargs,
        shuffle=False,
        persistent_workers=cfg.persistent_workers,
    )


def build_dataloaders(
    cfg: Config, tokenizer: Tokenizer
) -> Dict[str, Union[DataLoader, probe_utils.ConstituentDictionary, int]]:
    DsType: Any
    if cfg.use_tg and "TG" in cfg.probe.h5_path and not cfg.text_encoder.tg_right_branching:
        DsType = TGCocoProbeDataset
    elif cfg.use_tg and "TG" in cfg.probe.h5_path and cfg.text_encoder.tg_right_branching:
        DsType = RBCocoProbeDataset
    elif cfg.use_plm and "qian" in cfg.probe.h5_path:
        DsType = PLMCocoProbeDataset
    elif (cfg.text_encoder.text_encoder == "gpt2_bllip" and "qian-base" in cfg.probe.h5_path) or (
        cfg.text_encoder.text_encoder == "huggingface"
        and "gpt2" in cfg.text_encoder.hf_model_name_or_path
        and "gpt2" in cfg.probe.h5_path
    ):
        DsType = LMCocoProbeDataset
    else:
        raise NotImplementedError
    logger.info("Using dataset type: %s" % DsType)

    ds_dict = {}
    splits = ("newval", "val", "absurd")
    ann_files = {
        "newval": cfg.ann_trees_train_json,
        "train": cfg.ann_trees_train_json,
        "val": cfg.ann_trees_val_json,
        "absurd": cfg.ann_trees_absurd_json,
    }
    for split in splits:
        ds = DsType(
            cfg,
            tokenizer,
            ann_files[split],
            split,
            negative_constituents=cfg.probe.eval_which_negative_constituents,
        )
        ds_dict[split.replace("valtest", "val")] = inference_dataloader(ds, cfg)
        ds_dict["tag_dict"] = ds.tag_dict
        ds_dict["emb_dim"] = ds.h5_ds_shape[-1]
    if cfg.do_train:
        tds = DsType(
            cfg,
            tokenizer,
            cfg.ann_trees_train_json,
            "train",
            negative_constituents=cfg.probe.learn_which_negative_constituents,
        )
        ds_dict["train"] = train_dataloader(tds, cfg)
    return ds_dict
