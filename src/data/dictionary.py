import logging
from typing import List, Union

import numpy as np
import torch
from config import Config
from fairseq.data import Dictionary
from torch import LongTensor, Tensor

logger = logging.getLogger("pytorch_lightning")


class CategoryDictionary(Dictionary):
    def __init__(self, cfg: Config, categories, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = cfg
        self.cat_label_to_coco_ids = None
        self.cat_label_to_coco_names = None
        self.coco_ids_to_coco_names = None
        self.nobj_coco_id = cfg.nobj_id
        self.nobj_label_id = None
        self.coco_categories = categories
        self.max_coco_category = max([cat["id"] for cat in categories.values()])

        self.build_category_dictionary()

    def build_category_dictionary(self):
        categories_by_dct_idx = {}
        for idx, cat in self.coco_categories.items():
            idx_in_dct = self.add_symbol(cat["name"])
            cat["idx_in_dct"] = idx_in_dct
            categories_by_dct_idx[idx_in_dct] = cat

        # todo add category for <NOT AN OBJECT> ??? OR only if num_queries not fixed
        not_an_obj_idx_in_dct = self.add_symbol("<NOBJ>")
        categories_by_dct_idx[not_an_obj_idx_in_dct] = {
            "name": "<NOBJ>",
            "id": self.cfg.nobj_id,
            "supercategory": "<NOBJ>",
            "idx_in_dct": not_an_obj_idx_in_dct,
        }

        self.set_cat_labels_to_coco_ids_map(categories_by_dct_idx)
        self.nobj_label_id = not_an_obj_idx_in_dct
        # do not call finalize!

    def set_cat_labels_to_coco_ids_map(self, categories_by_dct_idx):
        self.cat_label_to_coco_ids = [
            (
                categories_by_dct_idx[i]["id"]
                if i in categories_by_dct_idx and "id" in categories_by_dct_idx[i]
                else self.symbols[i]
            )
            for i in range(len(self))
        ]
        self.cat_label_to_coco_names = [
            categories_by_dct_idx[i]["name"] if i in categories_by_dct_idx else self.symbols[i]
            for i in range(len(self))
        ]
        self.coco_ids_to_coco_names = {
            cat["id"] if "id" in cat else cat["name"]: cat["name"]
            for cat in categories_by_dct_idx.values()
        }

    def convert_cat_labels_to_coco_ids(
        self,
        cat_labels: List[Union[List[int], int]],
        as_tensor=False,
        exclude_nobj=False,
        map_special_to=False,
        special=-1,
    ):
        if not len(cat_labels) > 0 or not (
            isinstance(cat_labels[0], List)
            or (isinstance(cat_labels[0], np.ndarray) and cat_labels[0].ndim > 0)
            or (isinstance(cat_labels[0], Tensor) and cat_labels[0].dim() > 0)
        ):
            if not self.cfg.old_coco_label_convert:
                return self.convert_cat_labels_to_coco_ids(
                    [cat_labels], as_tensor, exclude_nobj, map_special_to, special
                )[0]
            else:
                return self.convert_cat_labels_to_coco_ids([cat_labels])[0]
        # map_fn = partial(torch.as_tensor, dtype=torch.long) if as_tensor else id
        return (
            [
                [
                    (
                        self.cat_label_to_coco_ids[int(lab)]
                        if (not map_special_to or int(lab) >= self.nspecial)
                        else special
                    )
                    for lab in row
                    if (not exclude_nobj or int(lab) != self.nobj_label_id)
                ]
                for row in cat_labels
            ]
            if not self.cfg.old_coco_label_convert
            else [
                [
                    self.cat_label_to_coco_ids[int(lab)]
                    for lab in row
                    if (not exclude_nobj or int(lab) != self.nobj_label_id)
                ]
                for row in cat_labels
            ]
        )

    def convert_coco_ids_to_coco_names(self, coco_ids: List[Union[List[int], int]]):
        if not len(coco_ids) > 0 or not (
            isinstance(coco_ids[0], List)
            or (isinstance(coco_ids[0], np.ndarray) and coco_ids[0].ndim > 0)
            or (isinstance(coco_ids[0], Tensor) and coco_ids[0].dim() > 0)
        ):
            return self.convert_coco_ids_to_coco_names([coco_ids])[0]
        return [
            [
                self.coco_ids_to_coco_names[lab] if lab in self.coco_ids_to_coco_names else lab
                for lab in row
            ]
            for row in coco_ids
        ]

    def convert_cat_labels_to_coco_names(self, cat_labels):
        if not len(cat_labels) > 0 or not (
            isinstance(cat_labels[0], List)
            or (isinstance(cat_labels[0], np.ndarray) and cat_labels[0].ndim > 0)
            or (isinstance(cat_labels[0], Tensor) and cat_labels[0].dim() > 0)
        ):
            return self.convert_cat_labels_to_coco_names([cat_labels])[0]
        # map_fn = partial(torch.as_tensor, dtype=torch.long) if as_tensor else id
        return [[self.cat_label_to_coco_names[int(lab)] for lab in row] for row in cat_labels]


class PositionDictionary(Dictionary):
    def __init__(
        self,
        cfg: Config,
        pad="<pad>",
        eos="</s>",
        unk="<unk>",
        bos="<s>",
        mask="<mask>",
        extra_special_symbols=None,
    ):
        super().__init__(pad, eos, unk, bos, mask, extra_special_symbols, skip_specials=True)
        self.n_positions = cfg.nb_of_pos_bins
        self.quantize_resolution = (1 / cfg.nb_of_pos_bins) + 1e-7
        self.pos_bins = np.arange(0.0, 1.01, self.quantize_resolution)
        logger.info(
            f"Using bins to quantize bbox positions: {self.pos_bins}, nb: {len(self.pos_bins) - 1}"
        )
        # self.nb_pos_tokens = cfg.nb_of_pos_bins
        assert len(self) == 0
        # padding index is not in self.symbols/self.indices, but can still be retrieved with .pad()
        self.pad_index = len(self.pos_bins) - 1

        for pos in range(self.n_positions):
            self.add_symbol(pos)

    def add_symbol(self, *args, **kwargs):
        super().add_symbol(*args, **kwargs)
        assert self.symbols == list(range(len(self)))

    def encode(self, cont_coords: List) -> List:
        if isinstance(cont_coords[0], List) or (
            isinstance(cont_coords, Tensor) and cont_coords.dim() > 1
        ):
            return [self.encode(coords) for coords in cont_coords]

        return self.quantize(cont_coords).tolist()

    def encode_tensor(self, cont_coords: Tensor) -> Tensor:
        if isinstance(cont_coords[0], List) or (
            isinstance(cont_coords, Tensor) and cont_coords.dim() > 1
        ):
            return torch.stack([self.encode_tensor(coords) for coords in cont_coords])

        return torch.LongTensor(self.quantize(cont_coords.cpu()))

    def quantize(self, coords):
        return np.digitize(np.array(coords).clip(0, 1), self.pos_bins) - 1

    def decode(self, encoded_coords: List, drop_special=False) -> List:
        if isinstance(encoded_coords[0], List):
            return [self.decode(coords) for coords in encoded_coords]

        return [
            (
                self.symbols[c] * self.quantize_resolution + self.quantize_resolution / 2
                if c < len(self.pos_bins) - 1
                else "<pad>"
            )
            for c in encoded_coords
            if not drop_special or c < len(self.pos_bins) - 1
        ]

    def decode_tensor(self, encoded_coords: LongTensor) -> Tensor:
        return encoded_coords.float() * self.quantize_resolution + self.quantize_resolution / 2
