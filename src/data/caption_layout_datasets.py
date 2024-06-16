import copy
import json
import logging
import math
import os
import pickle
import random
from abc import ABC
from functools import partial
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from config import Config
from torch import Tensor, nn
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import CocoCaptions, CocoDetection
from tqdm import tqdm

from data.dictionary import CategoryDictionary, PositionDictionary
from data.singleton import S
from data.spurious_bbox_filter import SpuriousFilter

# Vocabulary import needed to load pickle
from data.tokenization import Tokenizer, Vocabulary

logger = logging.getLogger("pytorch_lightning")


def crop_pad_normalize(instances, extra_cpn_pad=0.04):
    if len(instances) > 0:
        og_bboxes = np.array([inst["bbox"] for inst in instances], dtype=float)
        bboxes = og_bboxes.copy()
        x_min, x_max = np.min(bboxes[:, 0]), np.max(bboxes[:, 0] + bboxes[:, 2])
        y_min, y_max = np.min(bboxes[:, 1]), np.max(bboxes[:, 1] + bboxes[:, 3])

        # add extra padding around the centered boxes on all sides,
        # so no box coordinates are equal to 0 or 1,
        #  which a sigmoid can't predict
        extra_pad = extra_cpn_pad * (x_max - x_min)
        if x_max - x_min > y_max - y_min:
            D = x_max - x_min
            P = ((x_max - x_min) - (y_max - y_min)) / 2
            x_min_n, _ = x_min, x_max
            y_min_n, _ = y_min - P, y_max + P
        else:
            D = y_max - y_min
            P = ((y_max - y_min) - (x_max - x_min)) / 2
            x_min_n, _ = x_min - P, x_max + P
            y_min_n, _ = y_min, y_max

        D += 2 * extra_pad
        bboxes[:, 0], bboxes[:, 1] = (
            bboxes[:, 0] - x_min_n + extra_pad,
            bboxes[:, 1] - y_min_n + extra_pad,
        )
        bboxes = bboxes / max(D, 1e-4)

        for i, inst in enumerate(instances):
            inst["bbox_cpn"] = bboxes[i].tolist()

    return instances


def compute_mean_std_dimensions(cfg: Config, smartfilter: SpuriousFilter):
    ds = S.TrainDetection(cfg.train_image_dir, cfg.train_instances, transform=transforms.ToTensor())

    xs, ys, ws, rs = [], [], [], []
    for img_id in tqdm(ds.coco.imgs.keys()):
        img = ds.coco.loadImgs(img_id)[0]
        instances = ds.coco.loadAnns(ds.coco.getAnnIds(img_id))
        h, w = img["height"], img["width"]

        _, instances, _, _ = CocoInstancesDataset.process_instances(
            instances,
            filter_crowds=cfg.filter_out_crowds,
            smartfilter=smartfilter,
            img_transform=None,
            cpn=crop_pad_normalize,
            clamp_neg=True,
            move_neg=False,
            image=None,
            height=h,
            width=w,
            cpn_pad=cfg.cpn_extra_pad,
        )
        for inst in instances:
            x, y, w, h = inst["bbox_cpn"]
            xs.append(x)
            ys.append(y)
            ws.append(w)
            rs.append(h / w)

    return {
        "x": (np.mean(xs), np.std(xs)),
        "y": (np.mean(ys), np.std(ys)),
        "w": (np.mean(ws), np.std(ws)),
        "r": (np.mean(rs), np.std(rs)),
    }


class CocoInstancesDataset(Dataset, ABC):
    def __init__(
        self,
        cfg: Config,
        image_dir: str,
        instances_file: str,
        tokenizer: Tokenizer,
        inference: bool = False,
        split="train",
        smartfilter=None,
        image_transform=None,
        _preprocess_instance_ds=True,
        use_ids_from_file=None,
        rprecision=False,
    ):
        """
        https://cocodataset.org/#format-data
        """
        self.smartfilter = smartfilter
        self.image_transform = image_transform
        self.instance_ds: CocoDetection = {
            "train": S.TrainDetection,
            "newval": S.TrainDetection,
            "val": S.ValDetection,
            "absurd": S.AbsurdDetection,
        }[split](image_dir, instances_file, transform=transforms.ToTensor())
        self.split = split

        self.image_ids = self.instance_ds.ids
        if use_ids_from_file is not None and isinstance(use_ids_from_file, str):
            with open(use_ids_from_file, "r") as f:
                self.image_ids = json.load(f)
        elif use_ids_from_file is not None and isinstance(use_ids_from_file, List):
            self.image_ids = use_ids_from_file

        self.image_id_to_instance_idx = {}
        for idx, img_id in enumerate(self.instance_ds.ids):
            self.image_id_to_instance_idx[int(img_id)] = idx

        self.tokenizer = tokenizer
        self.cfg = cfg
        self.inference = inference

        self.categories = self.instance_ds.coco.cats
        self.n_categories = len(self.categories)
        self.att_mask_pad_id = 0  # huggingface models
        if self.tokenizer is not None:
            self.txt_pad_id = self.tokenizer.pad_token_id

        self.n_positions = cfg.nb_of_pos_bins
        self.category_dict = CategoryDictionary(cfg, self.categories)
        self.pos_dict = PositionDictionary(cfg)
        self.pos_cont_pad_id = cfg.pos_cont_pad_id

        self.rprecision = rprecision

        with open(self.cfg.new_valset_ids) as json_file:
            self.new_valset_ids = json.load(json_file)

        if _preprocess_instance_ds:
            self._preprocess_instance_ds()
        if cfg.shuffle_sentences or cfg.shuffle_sentences_eval:
            self.shuffle_seeds = dict(
                zip(self.image_ids, random.sample(range(0, 1000000), len(self.image_ids)))
            )

        self.norm_mean_std = False
        if cfg.model.obj_gan:
            assert cfg.use_cpn
            # self.dimension_mean_stds = ObjGAN.read_mean_std(cfg.obj_gan.mean_std_path)
            if os.path.exists(cfg.obj_gan.mean_std_path):
                logger.info("loading mean std dimensions from %s" % cfg.obj_gan.mean_std_path)
                with open(cfg.obj_gan.mean_std_path) as f:
                    self.dimension_mean_stds = json.load(f)
            else:
                logger.info(
                    "computing mean std dimensions and saving to %s" % cfg.obj_gan.mean_std_path
                )
                self.dimension_mean_stds = compute_mean_std_dimensions(cfg, smartfilter)
                with open(cfg.obj_gan.mean_std_path, "w") as f:
                    json.dump(self.dimension_mean_stds, f)
            self.norm_mean_std = not self.rprecision

    def _preprocess_instance_ds(self):
        logger.info("preprocessing instances in CocoInstancesDataset...")

        if self.cfg.num_allowed_objects_in_image > 0:
            self.image_ids = self.filter_images_with_many_objects(self.image_ids)

        if self.cfg.use_new_valset and self.split != "newval":
            self.image_ids = [i for i in self.image_ids if i not in self.new_valset_ids]

        logger.info("Data in dataset: \t {}/{}".format(len(self.image_ids), len(self.instance_ds)))
        logger.info("preprocessing done!")

    def filter_images_with_many_objects(self, image_ids):
        return list(
            filter(
                lambda img_id: self._check_num_objects_for_image(img_id)
                <= self.cfg.num_allowed_objects_in_image,
                image_ids,
            )
        )

    @staticmethod
    def apply_image_transform(image, instances, transform):
        if transform is not None:
            bboxes = [inst["bbox"] for inst in instances]
            bboxes = [
                (
                    min(max(1e-5, x), image.shape[2]),
                    min(max(1e-5, y), image.shape[1]),
                    min(max(1e-5, w), max(1e-5, image.shape[2] - min(max(0, x), image.shape[2]))),
                    min(max(1e-5, h), max(1e-5, image.shape[1] - min(max(0, y), image.shape[1]))),
                )
                for (x, y, w, h) in bboxes
            ]
            ids = [inst["id"] for inst in instances]
            try:
                transformed = transform(
                    image=image.numpy().transpose(1, 2, 0), bboxes=bboxes, ids=ids
                )
                transformed_image = torch.Tensor(transformed["image"].transpose(2, 0, 1))
                bboxes, ids = transformed["bboxes"], transformed["ids"]
                transformed_instances = []
                for inst in copy.deepcopy(instances):
                    if inst["id"] in ids:
                        inst["bbox"] = bboxes[ids.index(inst["id"])]
                        transformed_instances.append(inst)
                return transformed_image, transformed_instances
            except Exception:
                logger.exception("Error while transforming instances %s" % ids)
        return image, instances

    def _check_num_objects_for_image(self, image_id):
        instance_ds, _ = self.get_instance_ds_and_idx_for_id(image_id)
        instances = instance_ds.coco.loadAnns(instance_ds.coco.getAnnIds(image_id))
        if self.smartfilter is not None:
            img_ann = instance_ds.coco.loadImgs(image_id)[0]
            height, width = img_ann["height"], img_ann["width"]
            instances = self.smartfilter(instances, height, width)
        return len(instances)

    def get_instance_ds_and_idx_for_id(self, img_id):
        # if img_id not in self.image_ids: raise ValueError
        return self.instance_ds, self.image_id_to_instance_idx

    def get_processed_instances_for_id(self, image_id):
        instance_ds, instance_idx = self.get_instance_ds_and_idx_for_id(image_id)
        image_dct = instance_ds.coco.loadImgs(image_id)[0]
        w, h = image_dct["width"], image_dct["height"]

        if self.split != "absurd":
            image, instances = instance_ds[instance_idx[image_id]]
            sf, clamp_neg, move_neg = self.smartfilter, True, False
        else:
            instances = instance_ds.coco.loadAnns(instance_ds.coco.getAnnIds(image_id))
            image, sf, clamp_neg, move_neg = None, None, False, True

        if self.cfg.old_preprocessing:
            clamp_neg, move_neg = False, True

        cpn = None
        if self.cfg.use_cpn:
            cpn = crop_pad_normalize

        image, instances, h, w = self.process_instances(
            instances,
            self.cfg.filter_out_crowds,
            sf,
            self.image_transform,
            cpn,
            clamp_neg,
            move_neg,
            image,
            h,
            w,
            self.cfg.cpn_extra_pad,
        )
        return image, instances, w, h

    @staticmethod
    def process_instances(
        instances,
        filter_crowds=True,
        smartfilter=None,
        img_transform=None,
        cpn=None,
        clamp_neg=False,
        move_neg=False,  # norm_mean_std=False,
        image=None,
        height=None,
        width=None,
        cpn_pad=0.02,
    ):
        if filter_crowds:
            instances = [inst for inst in instances if inst["iscrowd"] == 0]
        if smartfilter is not None:
            if height is None or width is None:
                _, height, width = image.shape
            instances = smartfilter(instances, height, width)
        if clamp_neg:
            instances = CocoInstancesDataset.clamp_boxes_to_hw(instances, height, width)
        if move_neg:
            instances = CocoInstancesDataset.move_negative_boxes(instances)
        if img_transform is not None:
            image, instances = CocoInstancesDataset.apply_image_transform(
                image, instances, img_transform
            )
            _, height, width = image.shape
        if cpn is not None:
            instances = crop_pad_normalize(instances, cpn_pad)
        return image, instances, height, width

    @staticmethod
    def clamp_boxes_to_hw(instances, img_h, img_w):
        for instance in instances:
            x, y, w, h = instance["bbox"]
            nx = min(max(0.0, x), img_w)
            ny = min(max(0.0, y), img_h)
            nw = min(max(0.0, w), max(0.0, img_w - nx))
            nh = min(max(0.0, h), max(0.0, img_h - ny))
            # nx = min(max(1e-5, x), img_w)
            # ny = min(max(1e-5, y), img_h)
            # nw = min(max(1e-5, w), max(1e-5, img_w - nx))
            # nh = min(max(1e-5, h), max(1e-5, img_h - ny))
            instance["bbox_before_clamp"] = instance["bbox"]
            instance["bbox_clamped"] = any([x != nx, y != ny, w != nw, h != nh])
            instance["bbox"] = [nx, ny, nw, nh]
        return instances

    @staticmethod
    def move_negative_boxes(instances):
        move_right, move_down = 0, 0
        for instance in instances:
            x, y, _, _ = instance["bbox"]
            if x < 0:
                move_right = max(move_right, math.abs(x))
            if y < 0:
                move_down = max(move_down, math.abs(y))
        for instance in instances:
            x, y, w, h = instance["bbox"]
            instance["bbox"] = [x + move_right, y + move_down, w, h]
        return instances

    def __len__(self):
        return len(self.image_ids)

    def tokenize_labels_positions(
        self,
        image_size,
        instances,
        cutoff_number=True,
        add_full_img_box=False,
        already_normalized=False,
    ):
        full_img_box = add_full_img_box or self.rprecision

        labels = [self.category_dict.bos()]
        bboxes_quant = (
            [4 * [self.pos_dict.pad()]]
            if not full_img_box
            else [self.pos_dict.encode([0.5, 0.5, 1.0, 1.0])]
        )
        bboxes_cont = [4 * [self.pos_cont_pad_id]] if not full_img_box else [[0.5, 0.5, 1.0, 1.0]]
        replaced = [False]

        img_height, img_width = image_size

        if self.cfg.order_labels_by == "big_to_small":

            def sort_by_key(i):
                if "area" in i:
                    return i["area"]
                else:
                    bbox_key = "bbox_cpn" if "bbox_cpn" in i else "bbox"
                    return i[bbox_key][2] * i[bbox_key][3]

            reverse = True
        else:
            assert self.cfg.order_labels_by == "left_to_right", self.cfg.order_labels_by

            def sort_by_key(i):
                return i["bbox"][0] + (i["bbox"][2] / 2)

            reverse = False
        sorted_instances = sorted(instances, key=sort_by_key, reverse=reverse)

        for instance in sorted_instances:
            if instance["category_id"] == self.category_dict.nobj_coco_id:
                continue
            if self.cfg.filter_out_crowds and instance["iscrowd"] != 0:
                continue
            if cutoff_number and len(labels) > self.cfg.model.max_target_positions:
                break

            # box coordinates are measured from the top left image corner and are 0-indexed
            # https://cocodataset.org/#format-data
            if self.cfg.use_cpn or already_normalized:
                # already normalized by CPN
                px, py, pw, ph = (
                    instance["bbox_cpn"] if "bbox_cpn" in instance else instance["bbox"]
                )
            else:
                x, y, w, h = instance["bbox"]  # [x,y,width,height]
                px, py, pw, ph = x / img_width, y / img_height, w / img_width, h / img_height

            # this should not have any effect if CPN is enabled
            py_4_clamp = px if self.cfg.old_preprocessing_clamp else py
            px, py = min(max(px, 0.0), 1.0), min(max(py_4_clamp, 0.0), 1.0)
            pw, ph = min(max(pw, 0.0), 1.0 - px), min(max(ph, 0.0), 1.0 - py)
            # center_x, center_y
            bbox_cont = [px + pw / 2, py + ph / 2, pw, ph]

            if not self.cfg.old_preprocessing_skip_zero and (pw < 1e-8 or ph < 1e-8):
                logger.error(
                    f'\nBbox with zero width or height skipped. Image id: {instance["image_id"]}, '
                    f'Instance id: {instance["id"] if "id" in instance else "?"}, Box: {bbox_cont}'
                )
                continue
            # list of 4 ints
            bbox_quantized: List = self.pos_dict.encode(
                bbox_cont.cpu() if isinstance(bbox_cont, Tensor) else bbox_cont
            )
            instance.update({
                "bbox_prop": [px, py, pw, ph],
                "bbox_prop_center": bbox_cont,
                "bbox_quantized": bbox_quantized,
            })
            if self.norm_mean_std:
                mx, my = bbox_cont[:2]
                r = ph / pw
                bbox_norm_mean_std = [
                    (mx - self.dimension_mean_stds["x"][0]) / self.dimension_mean_stds["x"][1],
                    (my - self.dimension_mean_stds["y"][0]) / self.dimension_mean_stds["y"][1],
                    (pw - self.dimension_mean_stds["w"][0]) / self.dimension_mean_stds["w"][1],
                    (r - self.dimension_mean_stds["r"][0]) / self.dimension_mean_stds["r"][1],
                ]
                instance["bbox_norm_mean_std"] = bbox_norm_mean_std
                bboxes_cont.append(bbox_norm_mean_std)
            else:
                bboxes_cont.append(bbox_cont)

            bboxes_quant.append(bbox_quantized)
            replaced.append(instance["replaced"] if self.split == "absurd" else False)

            category_info = self.categories[instance["category_id"]]
            label = self.category_dict.index(category_info["name"])
            instance.update({"category": category_info, "label": label})
            labels.append(label)

            instance.pop("segmentation", None)

        labels += [self.category_dict.eos()]
        bboxes_quant += [4 * [self.pos_dict.pad()]]  # or eos?
        bboxes_cont += [4 * [self.pos_cont_pad_id]]
        replaced += [False]

        return bboxes_quant, bboxes_cont, labels, replaced, sorted_instances

    @staticmethod
    def _sort_by_len(
        labels,
        bboxes,
        bboxes_cont,
        captions,
        img_shapes,
        img_ids,
        tokenized,
        caption_ids,
        instances,
    ):
        tokenized = [dict(zip(tokenized, t)) for t in zip(*tokenized.values())]
        (
            labels,
            bboxes,
            bboxes_cont,
            captions,
            img_shapes,
            img_ids,
            tokenized,
            caption_ids,
            instances,
        ) = zip(
            *sorted(
                zip(
                    labels,
                    bboxes,
                    bboxes_cont,
                    captions,
                    img_shapes,
                    img_ids,
                    tokenized,
                    caption_ids,
                    instances,
                ),
                key=lambda tup: tup[-3]["attention_mask"].sum(-1),
                reverse=True,
            )
        )
        tokenized = {k: torch.stack([d[k] for d in tokenized], dim=0) for k in tokenized[0]}
        return (
            labels,
            bboxes,
            bboxes_cont,
            captions,
            img_shapes,
            img_ids,
            tokenized,
            caption_ids,
            instances,
        )


class CocoInstancesAndAnyCaptionsDataset(CocoInstancesDataset, ABC):
    def __init__(
        self,
        cfg: Config,
        image_dir: str,
        instances_file: str,
        tokenizer: Tokenizer,
        inference: bool = False,
        split="train",
        smartfilter=None,
        image_transform=None,
        _preprocess_instance_ds=True,
        use_ids_from_file=None,
        rprecision=False,
    ):
        super().__init__(
            cfg,
            image_dir,
            instances_file,
            tokenizer,
            inference,
            split,
            smartfilter,
            image_transform,
            _preprocess_instance_ds,
            use_ids_from_file,
            rprecision=rprecision,
        )
        self.load_span_tree_pos_embs = cfg.detr.load_span_tree_pos_embs
        self.syntax_tree_ds = None
        if self.load_span_tree_pos_embs:
            with open(cfg.vocab_file, "rb") as f:
                self.vocab = pickle.load(f)
            captions_file, dataset_class = {
                "train": (cfg.syntax_train_json, S.SyntaxTrainCaptions),
                "newval": (cfg.syntax_train_json, S.SyntaxTrainCaptions),
                "val": (cfg.syntax_val_json, S.SyntaxValCaptions),
                "absurd": (cfg.syntax_absurd_json, S.SyntaxAbsurdCaptions),
            }[split]
            self.syntax_tree_ds = dataset_class(
                captions_file,
                self.vocab,
                n=cfg.lt.n,
                k=cfg.lt.k,
                mask_value=cfg.latent_tree_h5_mask_value,
            )

    def join_with_latent_tree_positions(self, image_id, anns):
        # span trees ~ titov
        if self.load_span_tree_pos_embs:
            ds = self.get_syntax_ds_for_id(image_id)
            sp_tree_indices = ds.get_indices_for_image_id(image_id, max_size=self.cfg.lt.max_nodes)
            sp_tree_items = [ds[tree_idx] for tree_idx in sp_tree_indices]
            sp_tree_items = {sp_t["caption_id"]: sp_t for sp_t in sp_tree_items}
            anns = [
                {
                    **t,
                    "node_mask": sp_tree_items[t["caption_id"]]["node_mask"],
                    "node_pos": sp_tree_items[t["caption_id"]]["node_pos"],
                    "span_list": sp_tree_items[t["caption_id"]]["no_enc_pos"],
                    "span_list_tokens": [
                        ds.vocab.idx2word[t.item()]
                        for t in sp_tree_items[t["caption_id"]]["tokens"]
                    ],
                }
                for t in anns
                if t["caption_id"] in sp_tree_items
            ]
        return anns

    def get_captions_ds_for_id(self, img_id):
        # if img_id not in self.image_ids: raise ValueError
        return self.captions_ds

    def get_syntax_ds_for_id(self, img_id):
        # if img_id not in self.image_ids: raise ValueError
        return self.syntax_tree_ds

    def pick_caption(self, caption_ids: Sequence[int]) -> int:
        return (
            random.choice(range(len(caption_ids))) if not self.inference else len(caption_ids) - 1
        )


class CocoInstancesAndCaptionsDataset(CocoInstancesAndAnyCaptionsDataset):
    def __init__(
        self,
        cfg: Config,
        image_dir: str,
        captions_file: str,
        instances_file: str,
        tokenizer: Tokenizer,
        inference: bool = False,
        split="train",
        smartfilter=None,
        image_transform=None,
        _preprocess_instance_ds=True,
        use_ids_from_file=None,
        rprecision=False,
    ):
        super().__init__(
            cfg,
            image_dir,
            instances_file,
            tokenizer,
            inference,
            split,
            smartfilter,
            image_transform,
            _preprocess_instance_ds,
            use_ids_from_file,
            rprecision=rprecision,
        )
        self.captions_ds: CocoCaptions = {
            "train": S.TrainCaptions,
            "newval": S.TrainCaptions,
            "val": S.ValCaptions,
            "absurd": S.AbsurdCaptions,
        }[split](image_dir, captions_file)
        # assert len(self.captions_ds) == len(self.instance_ds)
        self.image_id_to_caption_idx = {}
        for idx, img_id in enumerate(self.captions_ds.ids):
            self.image_id_to_caption_idx[int(img_id)] = idx

        self.sort_by_caption_len = cfg.text_encoder.sort_by_caption_len

    def __getitem__(self, index):
        img_id = self.image_ids[index]
        captions_ds = self.get_captions_ds_for_id(img_id)
        aids = captions_ds.coco.getAnnIds(imgIds=[img_id])
        anns = captions_ds.coco.loadAnns(aids)
        anns = self.join_with_latent_tree_positions(
            img_id, [{**ann, "caption_id": ann["id"]} for ann in anns]
        )
        if len(anns) == 0:
            logger.error("Length 0")
        ann = anns[self.pick_caption([ann["id"] for ann in anns])]
        image, instances, img_width, img_height = self.get_processed_instances_for_id(img_id)

        caption, caption_id = ann["caption"], ann["id"]
        # img_filename = self.captions_ds.coco.loadImgs(self.captions_ds.ids[index])[0]['file_name']

        caption = caption.strip().rstrip(".")
        if (
            (self.split != "train" and self.cfg.shuffle_sentences_eval)
            or self.split == "train"
            and self.cfg.shuffle_sentences
        ):
            shuffled_caption = caption.split(" ")
            random.Random(self.shuffle_seeds[img_id]).shuffle(shuffled_caption)
            caption = " ".join(shuffled_caption)
        bboxes, bboxes_cont, labels, replaced, _ = self.tokenize_labels_positions(
            [img_height, img_width], instances
        )
        nb_clamped = len([i for i in instances if "bbox_clamped" in i and i["bbox_clamped"]])

        return {
            **ann,
            "caption": caption,
            "img_id": img_id,
            "caption_id": caption_id,
            "img_shape": [img_height, img_width],
            "labels": labels,
            "bboxes": bboxes,
            "bboxes_cont": bboxes_cont,
            "instances": instances,
            "nb_clamped": nb_clamped,
            "replaced": replaced,
        }

    def collate(self, list_of_samples: List[Dict[str, Any]]):
        (
            labels,
            bboxes,
            bboxes_cont,
            captions,
            img_shapes,
            img_ids,
            caption_ids,
            instances,
            nb_clamped,
            replaced,
        ) = zip(
            *(
                (
                    torch.as_tensor(s["labels"], dtype=torch.long),
                    torch.as_tensor(s["bboxes"], dtype=torch.long),
                    torch.as_tensor(s["bboxes_cont"], dtype=torch.float),
                    s["caption"],
                    torch.as_tensor(s["img_shape"], dtype=torch.long),
                    s["img_id"],
                    s["caption_id"],
                    s["instances"],
                    s["nb_clamped"],
                    torch.as_tensor(s["replaced"], dtype=torch.bool),
                )
                for s in list_of_samples
            )
        )
        tokenized = self.tokenizer(list(captions))
        pad_fn = partial(nn.utils.rnn.pad_sequence, batch_first=True)

        node_pos, node_mask = None, None
        span_list = None
        span_list_tokens = None
        if self.load_span_tree_pos_embs:
            node_pos, node_mask = zip(*((s["node_pos"], s["node_mask"]) for s in list_of_samples))
            node_pos, node_mask = pad_fn(node_pos, padding_value=0), pad_fn(
                node_mask, padding_value=0
            )
            span_list = [s["span_list"] for s in list_of_samples]
            span_list_tokens = [s["span_list_tokens"] for s in list_of_samples]

        result = {
            "captions": captions,
            "img_ids": img_ids,
            "caption_ids": caption_ids,
            "instances": instances,
            "img_shapes": torch.stack(img_shapes),
            "bboxes": pad_fn(bboxes, padding_value=self.pos_dict.pad()),
            "bboxes_cont": pad_fn(bboxes_cont, padding_value=self.pos_cont_pad_id),
            "labels": pad_fn(labels, padding_value=self.category_dict.pad()),
            "tree_node_pos": node_pos,
            "tree_node_mask": node_mask,
            "nb_clamped": nb_clamped,
            "replaced": pad_fn(replaced, padding_value=False),
            "span_list": span_list,
            "span_list_tokens": span_list_tokens,
            **tokenized,
        }
        return result


class CocoInstancesAndNotatedSyntaxTreeDataset(CocoInstancesAndAnyCaptionsDataset):
    def __init__(
        self,
        cfg: Config,
        image_dir: str,
        captions_file: str,
        instances_file: str,
        tokenizer: Tokenizer,
        inference: bool = False,
        split="train",
        smartfilter=None,
        image_transform=None,
        _preprocess_instance_ds=True,
        use_ids_from_file=None,
        rprecision=False,
    ):
        super().__init__(
            cfg,
            image_dir,
            instances_file,
            tokenizer,
            inference,
            split,
            smartfilter,
            image_transform,
            _preprocess_instance_ds=False,
            use_ids_from_file=use_ids_from_file,
            rprecision=rprecision,
        )
        logger.info(
            "Building syntax tree dataset from silver-truth PLM/TG annotation file: %s"
            % captions_file
        )
        self.captions_ds = {
            "train": S.NotatedSyntaxTrainCaptions,
            "newval": S.NotatedSyntaxTrainCaptions,
            "val": S.NotatedSyntaxValCaptions,
            "absurd": S.NotatedSyntaxAbsurdCaptions,
        }[split](cfg, captions_file)

        if _preprocess_instance_ds:
            self._preprocess_instance_ds()

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        # 'image_id', 'caption', 'trees', 'caption_id'
        tree_items = self.get_captions_ds_for_id(image_id)[image_id]
        # span trees ~ titov
        tree_items = self.join_with_latent_tree_positions(image_id, tree_items)
        tree_item = tree_items[self.pick_caption([t["caption_id"] for t in tree_items])]

        image, instances, img_width, img_height = self.get_processed_instances_for_id(image_id)
        bboxes, bboxes_cont, labels, replaced, _ = self.tokenize_labels_positions(
            [img_height, img_width], instances
        )

        return {
            "img_shape": [img_height, img_width],
            "labels": labels,
            "bboxes": bboxes,
            "bboxes_cont": bboxes_cont,
            "replaced": replaced,
            **tree_item,
        }

    def collate(self, list_of_samples: List[Dict[str, Any]]):
        captions, labels, bboxes, bboxes_cont, img_shapes, img_ids, capt_ids, trees, replaced = zip(
            *(
                (
                    s["caption"],
                    torch.as_tensor(s["labels"], dtype=torch.long),
                    torch.as_tensor(s["bboxes"], dtype=torch.long),
                    torch.as_tensor(s["bboxes_cont"], dtype=torch.float),
                    torch.as_tensor(s["img_shape"], dtype=torch.long),
                    s["image_id"],
                    s["caption_id"],
                    s["trees"],
                    torch.as_tensor(s["replaced"], dtype=torch.bool),
                )
                for s in list_of_samples
            )
        )

        tokenized = self.tokenizer(list(trees))
        pad_fn = partial(nn.utils.rnn.pad_sequence, batch_first=True)

        node_pos, node_mask = None, None
        span_list = None
        span_list_tokens = None
        if self.load_span_tree_pos_embs:
            node_pos, node_mask = zip(*((s["node_pos"], s["node_mask"]) for s in list_of_samples))
            node_pos, node_mask = pad_fn(node_pos, padding_value=0), pad_fn(
                node_mask, padding_value=0
            )
            span_list = [s["span_list"] for s in list_of_samples]
            span_list_tokens = [s["span_list_tokens"] for s in list_of_samples]

        try:
            result = {
                "captions": captions,
                "img_ids": list(img_ids),
                "caption_ids": list(capt_ids),
                "img_shapes": torch.stack(img_shapes),
                "bboxes": pad_fn(bboxes, padding_value=self.pos_dict.pad()),
                "bboxes_cont": pad_fn(bboxes_cont, padding_value=self.pos_cont_pad_id),
                "labels": pad_fn(labels, padding_value=self.category_dict.pad()),
                "replaced": pad_fn(replaced, padding_value=False),
                "tree_node_pos": node_pos,
                "tree_node_mask": node_mask,
                # 'input_ids': list(trees),
                # 'attention_mask': torch.tensor([0]),
                "span_list": span_list,
                "span_list_tokens": span_list_tokens,
                **tokenized,
            }
            return result
        except Exception as e:
            print(bboxes)
            raise e
