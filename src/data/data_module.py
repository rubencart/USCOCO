import json
import logging
import random
from typing import Union

import albumentations as A
import pytorch_lightning as pl
from config import Config
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from data.caption_layout_datasets import (
    CocoInstancesAndCaptionsDataset,
    CocoInstancesAndNotatedSyntaxTreeDataset,
)
from data.singleton import S
from data.spurious_bbox_filter import (
    CompoundSpuriousFilter,
    SingleSpuriousFilter,
    SpuriousFilter,
)
from data.tokenization import Tokenizer

logger = logging.getLogger("pytorch_lightning")


class COCODataModule(pl.LightningDataModule):
    """https://pytorch-lightning.readthedocs.io/en/stable/datamodules.html"""

    coco_val: Union[CocoInstancesAndCaptionsDataset, Subset, None] = None
    coco_train: Union[CocoInstancesAndCaptionsDataset, Subset, None] = None
    coco_comp_test: Union[CocoInstancesAndCaptionsDataset, None] = None
    coco_new_val: Union[CocoInstancesAndCaptionsDataset, None] = None

    image_transform: Union[A.Compose, None] = None
    smartfilter: Union[SpuriousFilter, None] = None

    def __init__(self, cfg: Config, tokenizer: Tokenizer):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = tokenizer

    def setup(self, stage=None):
        assert stage == "fit" or stage == "test" or stage == "validate" or stage is None, stage

        if self.cfg.use_smart_filter:
            self.smartfilter = self.get_smartfilter()

        if self.cfg.use_image_transforms:
            self.image_transform = self.get_image_transforms()

        # Assign train/val datasets for use in dataloaders
        if self.cfg.use_plm or self.cfg.use_tg:
            ds_class = CocoInstancesAndNotatedSyntaxTreeDataset
            train_capt_file = self.cfg.ann_trees_train_json
            val_capt_file = self.cfg.ann_trees_val_json
            absurd_capt_file = self.cfg.ann_trees_absurd_json
        else:
            assert self.cfg.captions == "coco"
            ds_class = CocoInstancesAndCaptionsDataset
            train_capt_file = self.cfg.train_captions
            val_capt_file = self.cfg.val_captions
            absurd_capt_file = self.cfg.uscoco_captions

        if stage == "fit" or stage is None:
            logger.info("building original val set")
            self.coco_val = self.get_val_ds(ds_class, val_capt_file)
            self.val_set_for_val = self.coco_val

            logger.info("building absurd test set")
            self.absurd_test_set = self.get_absurd_ds(ds_class, absurd_capt_file)

            if self.cfg.overfit_on_val_samples > 0:
                overfit_subset_idcs = random.sample(
                    range(len(self.coco_val)), self.cfg.overfit_on_val_samples
                )
                logger.info("Overfitting on val samples, using same val subset for training")
                self.coco_train = Subset(
                    self.coco_val,
                    indices=self.cfg.overfit_times_in_epoch * overfit_subset_idcs,
                )
                self.val_set_for_val = self.coco_val = Subset(
                    self.coco_train,
                    indices=range(self.cfg.overfit_on_val_samples),
                )
                self.ds_for_attributes = self.coco_train.dataset
            else:
                logger.info("building train set")
                self.coco_train = self.get_train_ds(ds_class, train_capt_file)
                self.ds_for_attributes = self.coco_train

                if self.cfg.use_new_valset:
                    logger.info("building new val set")
                    self.coco_new_val = self.get_new_val_ds(ds_class, train_capt_file)
                    self.val_set_for_val = self.coco_new_val

        if stage == "validate":
            if self.cfg.use_new_valset:
                logger.info("building new val set for validating")
                self.val_set_for_val = self.coco_new_val = self.get_new_val_ds(
                    ds_class, train_capt_file
                )
            else:
                logger.info("building original val set for validating")
                self.val_set_for_val = self.coco_val = self.get_val_ds(ds_class, val_capt_file)
            self.ds_for_attributes = self.val_set_for_val

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            if self.cfg.use_new_valset:
                logger.info("building new val set for testing")
                self.coco_new_val = self.get_new_val_ds(ds_class, train_capt_file)
            logger.info("building original val set as in-domain test set")
            self.coco_val = self.get_val_ds(ds_class, val_capt_file)
            logger.info("building compositional test set")
            logger.info("building absurd test set")
            self.absurd_test_set = self.get_absurd_ds(ds_class, absurd_capt_file)

            if self.cfg.save_probe_embeddings_train:
                logger.info("building train set")
                self.coco_train = self.get_train_ds(ds_class, train_capt_file)

            self.ds_for_attributes = self.coco_val

        self.categories = self.ds_for_attributes.categories
        self.category_dict = self.ds_for_attributes.category_dict
        self.pos_dict = self.ds_for_attributes.pos_dict

        self.n_categories = self.ds_for_attributes.n_categories
        self.n_positions = self.ds_for_attributes.n_positions

        self.att_mask_pad_id = self.ds_for_attributes.att_mask_pad_id
        if self.tokenizer is not None:
            self.txt_pad_id = self.ds_for_attributes.txt_pad_id

    def get_image_transforms(self):
        image_transforms = []
        if self.cfg.p_HorizontalFlip > 0.0:
            image_transforms.append(A.HorizontalFlip(p=self.cfg.p_HorizontalFlip))
        if self.cfg.p_RandomSizedBBoxSafeCrop > 0.0:
            image_transforms.append(
                A.RandomSizedBBoxSafeCrop(
                    self.cfg.height_RandomSizedBBoxSafeCrop,
                    self.cfg.width_RandomSizedBBoxSafeCrop,
                    p=self.cfg.p_RandomSizedBBoxSafeCrop,
                )
            )
        if self.cfg.p_RandomScale > 0.0:
            image_transforms.append(
                A.RandomScale(self.cfg.max_RandomScale, p=self.cfg.p_RandomScale)
            )
        assert len(image_transforms) > 0
        image_transform = A.Compose(
            image_transforms,
            bbox_params=A.BboxParams(
                format="coco",
                label_fields=["ids"],
                min_visibility=self.cfg.transform_min_visibility,
            ),
            p=1.0,
        )
        return image_transform

    def get_smartfilter(self):
        instance_ds = S.TrainDetection(
            self.cfg.train_image_dir, self.cfg.train_instances, transform=transforms.ToTensor()
        )
        sfs = [
            SingleSpuriousFilter(
                discrimination_type=self.cfg.smart_filter_discrimination_type[i],
                min_relative_size=self.cfg.smart_filter_min_relative_size[i],
                max_deviation=self.cfg.smart_filter_max_deviation[i],
                instance_ds=instance_ds,
                img_dir=self.cfg.train_image_dir,
                annFile=self.cfg.train_instances,
                normalize_dist=self.cfg.smart_filter_normalize_dist[i],
                distr_type=self.cfg.smart_filter_distr_type[i],
                dist_type=self.cfg.smart_filter_dist_type[i],
                pretrained_spurious_filter_file=self.cfg.smart_filter_pretrained_file[i],
                save_file=self.cfg.smart_filter_save_file[i],
                filter_crowds=self.cfg.filter_out_crowds,
            )
            for i in range(self.cfg.nb_smart_filters)
        ]
        smartfilter = CompoundSpuriousFilter(sfs, self.cfg.smart_filter_compound_combination_type)
        return smartfilter

    def get_new_val_ds(self, ds_class, train_capt_file):
        return ds_class(
            self.cfg,
            self.cfg.train_image_dir,
            train_capt_file,
            self.cfg.train_instances,
            self.tokenizer,
            split="newval",
            inference=True,
            smartfilter=self.smartfilter,
            image_transform=None,
            use_ids_from_file=self.cfg.new_valset_ids,
        )

    def get_comp_test_ds(self, comp_test_ds_class):
        return comp_test_ds_class(
            self.cfg,
            self.tokenizer,
            smartfilter=self.smartfilter,
            image_transform=None,
        )

    def get_val_ds(self, ds_class, val_capt_file):
        return ds_class(
            self.cfg,
            self.cfg.val_image_dir,
            val_capt_file,
            self.cfg.val_instances,
            self.tokenizer,
            inference=True,
            split="val",
            smartfilter=self.smartfilter,
            image_transform=None,
        )

    def get_train_ds(self, ds_class, train_capt_file):
        return ds_class(
            self.cfg,
            self.cfg.train_image_dir,
            train_capt_file,
            self.cfg.train_instances,
            self.tokenizer,
            split="train",
            smartfilter=self.smartfilter,
            image_transform=self.image_transform,
        )

    def get_absurd_ds(self, ds_class, capt_file):
        with open(self.cfg.uscoco_captions) as f:
            absurd_dict = json.load(f)
        return ds_class(
            self.cfg,
            self.cfg.val_image_dir,
            capt_file,
            self.cfg.uscoco_instances,
            self.tokenizer,
            inference=True,
            split="absurd",
            smartfilter=None,
            image_transform=None,
            use_ids_from_file=list(set([ann["image_id"] for ann in absurd_dict["annotations"]])),
        )

    def train_dataloader(self):
        return DataLoader(
            self.coco_train,
            batch_size=self.cfg.batch_size,
            pin_memory=self.cfg.cuda and self.cfg.pin_memory,
            drop_last=False,
            num_workers=self.cfg.num_workers if not self.cfg.debug else self.cfg.debug_num_workers,
            shuffle=not self.cfg.deterministic,
            collate_fn=self.ds_for_attributes.collate,
        )

    def val_dataloader(self):
        return self._get_inference_dl(self.val_set_for_val)

    def _get_inference_dl(self, ds):
        return DataLoader(
            ds,
            batch_size=self.cfg.val_batch_size,
            pin_memory=self.cfg.cuda and self.cfg.pin_memory,
            num_workers=self.cfg.num_workers if not self.cfg.debug else self.cfg.debug_num_workers,
            drop_last=False,
            shuffle=False,
            collate_fn=self.ds_for_attributes.collate,
        )

    def test_dataloader(self):
        dls = [
            self._get_inference_dl(self.absurd_test_set),
            self._get_inference_dl(self.coco_val),
            self._get_inference_dl(self.coco_new_val) if self.cfg.use_new_valset else None,
            (
                self._get_inference_dl(self.coco_train)
                if self.cfg.save_probe_embeddings_train
                else None
            ),
        ]
        return dls

    def test_dataloader_select(self, val=False, absurd=False, test=False):
        dls = []
        if absurd:
            dls += [self._get_inference_dl(self.absurd_test_set)]
        if test:
            dls += [self._get_inference_dl(self.coco_val)]
        if val:
            dls += [
                self._get_inference_dl(
                    self.coco_new_val if self.cfg.use_new_valset else self.coco_val
                )
            ]
        return dls
