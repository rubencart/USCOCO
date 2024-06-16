# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import json
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
import utils
from config import Config
from data.dictionary import CategoryDictionary, PositionDictionary
from torch import nn

import model.obj_gan as ObjGAN


def label_mask(labels, cat_dict):
    return labels.ne(cat_dict.pad()) & labels.ne(cat_dict.eos()) & labels.ne(cat_dict.bos())


def filter_labels(labels, cat_dict: "CategoryDictionary", map_to_list=False) -> List:
    labels = labels.clone()
    labels.masked_fill_(labels.eq(cat_dict.eos()) | labels.eq(cat_dict.bos()), cat_dict.pad())

    def fn(x):
        return x.tolist() if map_to_list else x

    return [fn(row[row.ne(cat_dict.pad())]) for row in labels]


class PostProcess(nn.Module):
    """This module converts the model's output into the format expected by the coco api"""

    def __init__(self, cfg: Config, cat_dict: CategoryDictionary, pos_dict: PositionDictionary):
        super().__init__()
        assert cat_dict.symbols[-1] == "<NOBJ>"
        self.cat_dict = cat_dict
        self.pos_dict = pos_dict
        self.cfg = cfg
        self.prob_boxes = cfg.detr.probabilistic_bbox_predictions

        if cfg.model.obj_gan:
            self.gaussian_dict = np.load(cfg.obj_gan.gaussian_dict_path, allow_pickle=True).item()
            with open(cfg.obj_gan.mean_std_path) as f:
                self.dimension_mean_stds = json.load(f)

    @torch.no_grad()
    def forward(self, outputs, batch, ground_truth=False, ones_as_scores=False):
        """Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of
            each images of the batch
                          For evaluation, this must be the original image size
                          (before any data augmentation)
                          For visualization, this should be the image size after
                          data augment, but before padding
        """
        if ground_truth:
            return self.forward_with_ground_truth(outputs, batch)
        else:
            return self.forward_with_predictions(outputs, batch, ones_as_scores)

    def forward_with_predictions(self, outputs, batch, ones_as_scores=False):
        if self.cfg.model.obj_gan:
            boxes = outputs["gen_boxes"]
            labels, label_logits = outputs["gen_labels"], outputs["label_logits"]
            scores = F.softmax(label_logits, -1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)

            boxes = ObjGAN.coord_converter(boxes, self.dimension_mean_stds)
            v_masks = ObjGAN.validity_indices(labels, boxes, scores)
            pp_masks = ObjGAN.post_process(labels, boxes, self.cat_dict, self.gaussian_dict).bool()

            sp_masks = (
                labels.ne(self.cat_dict.pad())
                & labels.ne(self.cat_dict.eos())
                & labels.ne(self.cat_dict.nobj_label_id)
                & labels.ne(self.cat_dict.bos())
                & labels.ne(self.cat_dict.unk())
                & labels.ne(self.cat_dict.mask())
            )
            masks = v_masks & pp_masks & sp_masks
        elif self.cfg.model.autoregressive:
            out_boxes = outputs["gen_boxes"]
            labels, scores = outputs["gen_labels"], outputs["gen_labels_probs"]

            # exclude bos: logits & scores start from the token after bos,
            # boxes & labels & masks include bos
            out_boxes, labels = out_boxes[:, 1:], labels[:, 1:]
            masks = (
                labels.eq(self.cat_dict.pad())
                | labels.eq(self.cat_dict.eos())
                | labels.eq(self.cat_dict.nobj_label_id)
            )
            masks = ~masks

            boxes = self.pos_dict.decode_tensor(out_boxes) if self.prob_boxes else out_boxes
        else:
            out_logits, boxes, masks = (
                outputs["label_logits"].clone(),
                outputs["bbox_preds"],
                outputs["tgt_padding_mask"],
            )

            out_logits[..., list(range(self.cat_dict.nspecial))] = -float("Inf")
            prob = F.softmax(out_logits, -1)
            # this excludes the last element, NOBJ
            #  is this what we want? --> lower precision and higher recall?
            scores, labels = prob.max(-1)
            masks = ~masks & labels.ne(self.cat_dict.nobj_label_id)

        selected_labels = [lab[m].tolist() for lab, m in zip(labels, masks)]
        coco_labels = self.cat_dict.convert_cat_labels_to_coco_ids(selected_labels)

        xywh_boxes = utils.box_xyxy_to_xywh(utils.box_cxcywh_to_xyxy(boxes))
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = batch["img_shapes"].unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        scaled_boxes = xywh_boxes * scale_fct[:, None, :]

        boxes_flipped = boxes * torch.as_tensor([-1, 1, 1, 1]).type_as(boxes) + torch.as_tensor([
            1, 0, 0, 0
        ]).type_as(boxes)
        xywh_boxes_flipped = utils.box_xyxy_to_xywh(utils.box_cxcywh_to_xyxy(boxes_flipped))
        scaled_boxes_flipped = xywh_boxes_flipped * scale_fct[:, None, :]

        results = [
            {
                "image_id": img_id,
                "caption_id": caption_id,
                "scores": torch.ones(len(l)) if ones_as_scores else s[m].cpu().float(),
                "labels": l,
                # 'scaled_xywh_boxes': sb[m].cpu(),
                # 'scaled_xywh_boxes_flipped': sbf[m].cpu(),
                "norm_cxcywh_boxes": nb[m].cpu().float(),
                "norm_cxcywh_boxes_flipped": bf[m].cpu().float(),
                "norm_xywh_boxes": xywhb[m].cpu().float(),
                "boxes": xywhb[m].cpu(),
                "norm_xywh_boxes_flipped": xywhbf[m].cpu().float(),
                "boxes_flipped": xywhbf[m].cpu().float(),
                # 'scaled_boxes_xywh_flipped': sbf[m].cpu(),
                # 'sim_scores': sim_scores[],
                "tokens": tkns,
                "object_mask": m,
                "unmasked_labels_own": unm_lab,
                "tree_node_pos": tree_node_pos,
                "tree_node_mask": tree_node_mask,
                "span_list": span_list,
                "span_list_tokens": span_list_tokens,
            }
            for (
                img_id,
                s,
                l,
                sb,
                m,
                nb,
                bf,
                sbf,
                caption_id,
                xywhb,
                xywhbf,
                tkns,
                unm_lab,
                tree_node_pos,
                tree_node_mask,
                span_list,
                span_list_tokens,
            ) in zip(
                batch["img_ids"],
                scores,
                coco_labels,
                scaled_boxes,
                masks,
                boxes,
                boxes_flipped,
                scaled_boxes_flipped,
                batch["caption_ids"],
                xywh_boxes,
                xywh_boxes_flipped,
                # outputs['sim_scores']
                # outputs['text_embed'], outputs['text_lens'],
                batch.get("tokens", scores.shape[0] * [None]),
                labels,
                batch.get("tree_node_pos", scores.shape[0] * [None]),
                batch.get("tree_node_mask", scores.shape[0] * [None]),
                batch.get("span_list", scores.shape[0] * [None]),
                batch.get("span_list_tokens", scores.shape[0] * [None]),
            )
        ]
        if self.cfg.save_probe_embeddings:
            results = [
                #                         (bs x) n_layers x n_tokens x e
                {
                    **d,
                    "probe_embeddings": pe.transpose(1, 0)[m.bool()].transpose(0, 1).cpu().float(),
                    "probe_embeddings_inp": pe_i[m.bool()].cpu().float(),
                    "probe_n_tokens": m.sum(-1).cpu(),
                }
                for d, pe, pe_i, m in zip(
                    results,
                    outputs["probe_embeddings"],
                    outputs["probe_embeddings_inp"],
                    batch["attention_mask"],
                )
            ]

        # if self.cfg.debug:
        results = self.add_extra_info(
            results, batch, coco_labels, selected_labels, scale_fct, masks
        )

        return results

    def add_extra_info(self, results, batch, coco_labels, labels, scale_fct, masks):
        label_names = self.cat_dict.convert_coco_ids_to_coco_names(coco_labels)

        gt_labels = filter_labels(batch["labels"], self.cat_dict, map_to_list=True)
        gt_label_mask = label_mask(batch["labels"], self.cat_dict)
        gt_coco_labels = self.cat_dict.convert_cat_labels_to_coco_ids(gt_labels)
        gt_names = self.cat_dict.convert_coco_ids_to_coco_names(gt_coco_labels)

        if self.cfg.model.obj_gan:
            gt_boxes = batch["bboxes_cont"]
            gt_norm_ctrd_boxes = ObjGAN.coord_converter(gt_boxes, self.dimension_mean_stds)
        else:
            gt_norm_ctrd_boxes = batch["bboxes_cont"]
        gt_norm_unctrd_boxes = utils.box_xyxy_to_xywh(utils.box_cxcywh_to_xyxy(gt_norm_ctrd_boxes))
        gt_scaled_xywh_boxes = gt_norm_unctrd_boxes * scale_fct[:, None, :]
        replaced = batch["replaced"]  # [1:-1]

        results = [
            {
                **dct,
                "caption": caption,
                "labels_own": l,
                "names": n,
                "gt_labels_own": gt_l,
                "gt_labels": gt_c_l,
                "gt_names": gt_n,
                "gt_norm_cxcywh_boxes": gt_n_ctr[m].cpu().float(),
                "gt_norm_xywh_boxes": gt_n_unctr[m].cpu().float(),
                "size": size[:2].cpu(),
                "replaced": repl[m].cpu(),
            }
            for (
                dct,
                l,
                n,
                caption,
                gt_l,
                gt_c_l,
                gt_n,
                gt_n_ctr,
                gt_n_unctr,
                gt_s_unctr,
                m,
                size,
                repl,
            ) in zip(
                results,
                labels,
                label_names,
                batch["captions"],
                gt_labels,
                gt_coco_labels,
                gt_names,
                gt_norm_ctrd_boxes,
                gt_norm_unctrd_boxes,
                gt_scaled_xywh_boxes,
                gt_label_mask,
                scale_fct,
                replaced,
            )
        ]
        return results

    def forward_with_ground_truth(self, outputs, batch):
        """
        Should get perfect AP and AR for maxDets=100, for all IoU's and area's
        """
        raise NotImplementedError
        assert batch["img_shapes"].shape[1] == 2

        labels = batch["labels"]
        label_masks = [
            lab.ne(self.cat_dict.bos()) & lab.ne(self.cat_dict.pad()) & lab.ne(self.cat_dict.eos())
            for lab in labels
        ]
        labels = [lab[m] for lab, m in zip(labels, label_masks)]
        labels = [
            torch.as_tensor(
                [self.cat_dict.cat_label_to_coco_ids[lab] for lab in row], dtype=torch.long
            ).type_as(row)
            for row in labels
        ]

        boxes = self.pos_dict.decode_tensor(batch["bboxes"])
        # print(boxes.shape)
        # convert to [x0, y0, w, h] format, this is exact
        boxes = utils.box_xyxy_to_xywh(utils.box_cxcywh_to_xyxy(boxes))

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = batch["img_shapes"].unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        # print(scale_fct.shape)
        boxes = boxes * scale_fct[:, None, :]

        results = [
            {"scores": torch.ones_like(lab).type_as(lab), "labels": lab, "boxes": b[m]}
            for lab, b, m in zip(labels, boxes, label_masks)
        ]

        return results
