import logging

import torch
import torch.nn.functional as F
from config import Config
from data.dictionary import PositionDictionary
from fairseq.data import Dictionary
from model.loss.structure import (
    MultiObjectContrastiveAttnLoss,
    MultiObjectContrastiveMaxLoss,
)
from model.loss.utils import generalized_box_iou
from torch.nn.modules.loss import _Loss

from utils import box_cxcywh_to_xyxy

logger = logging.getLogger("pytorch_lightning")


class AutoregressiveCriterion(_Loss):
    def __init__(self, cfg: Config, category_dict: Dictionary, pos_dict: PositionDictionary):
        super().__init__()
        self.cfg = cfg
        self.padding_idx = category_dict.pad()
        self.eos_idx = category_dict.eos()
        self.bos_idx = category_dict.bos()

        self.pos_padding_id = pos_dict.pad()
        self.pos_cont_pad_id = cfg.pos_cont_pad_id
        self.pos_dict = pos_dict
        self.prob_bboxes = cfg.detr.probabilistic_bbox_predictions

        self.eps = cfg.detr.label_smoothing
        self.eps_pos = cfg.detr.label_smoothing_pos
        self.pos_wt = cfg.detr.prob_bbox_loss_coef
        self.weight_prob_bbox_loss_by_dist = cfg.detr.weight_prob_bbox_loss_by_dist
        self.bbox_loss_wt = cfg.detr.bbox_loss_coef
        self.struct_loss_wt = cfg.detr.struct_loss_coef
        self.giou_loss_wt = cfg.detr.giou_loss_coef
        self.bbox_rel_loss_wt = cfg.detr.bbox_rel_loss_coef
        self.bbox_prop_loss_wt = cfg.detr.bbox_prop_loss_coef
        self.prob_bbox_loss_wt = cfg.detr.prob_bbox_loss_coef
        self.bbox_rel_loss_type = cfg.detr.bbox_rel_loss_type

        if self.struct_loss_wt > 0.0:
            struct_loss_class = {
                "max": MultiObjectContrastiveMaxLoss,
                "attn": MultiObjectContrastiveAttnLoss,
            }[cfg.detr.struct_loss_type]
            self.struct_loss_fn = struct_loss_class(cfg.detr)

    def forward(self, outputs, batch):
        # LABEL LOSS
        # bos token is never predicted
        labels = batch["labels"][:, 1:]  # BS x L
        target_bboxes = batch["bboxes"][:, 1:] if self.prob_bboxes else batch["bboxes_cont"][:, 1:]
        label_logits = outputs["label_logits"]  # BS x L x T
        bbox_logits = outputs["bbox_logits"]

        if labels.shape[1] > label_logits.shape[1]:  # this should only occur during inference
            # todo when does this occur?
            labels = labels[:, : label_logits.shape[1]]
            target_bboxes = target_bboxes[:, : label_logits.shape[1]]
            # target_bboxes_cont = target_bboxes_cont[:, :label_logits.shape[1]]

        if labels.shape[1] < label_logits.shape[1]:  # this should only occur during inference
            label_logits = label_logits[:, : labels.shape[1]]
            bbox_logits = bbox_logits[:, : labels.shape[1]]

        label_non_padding_mask = labels.ne(self.padding_idx)  # BS x L
        # label_non_special_mask = (
        #     labels.ne(self.padding_idx) & labels.ne(self.bos_idx) & labels.ne(self.eos_idx)
        # )
        ntokens = label_non_padding_mask.long().sum()  # 1

        label_logprobs = F.log_softmax(label_logits, dim=-1)  # BS x L x T
        label_nll_loss = -label_logprobs.gather(dim=-1, index=labels.unsqueeze(-1))  # BS x L x 1
        label_smoothed_loss = -label_logprobs.sum(dim=-1, keepdim=True)  # BS x L x 1

        label_nll_loss = label_nll_loss[label_non_padding_mask]
        label_smoothed_loss = label_smoothed_loss[label_non_padding_mask]

        label_nll_loss = label_nll_loss.sum() / ntokens
        label_smoothed_loss = label_smoothed_loss.sum() / ntokens
        label_loss = (1 - self.eps) * label_nll_loss + self.eps / label_logprobs.size(
            -1
        ) * label_smoothed_loss

        # BBOX LOSS
        struct_loss, loss_l1, loss_giou, loss_rel, loss_prop = 5 * [torch.tensor(0.0)]

        if self.prob_bboxes:
            bbox_loss, bbox_nll_loss = self.compute_prob_bbox_loss(
                bbox_logits, outputs["bbox_preds"], target_bboxes
            )
        else:
            bbox_loss, loss_l1, loss_giou, loss_rel, loss_prop = self.compute_regression_bbox_loss(
                bbox_logits, target_bboxes
            )
            bbox_nll_loss = torch.tensor(0.0)

        loss = label_loss + bbox_loss  # / ntokens

        if self.cfg.detr.struct_loss_coef > 0.0:
            #  always predict one box that is the entire image so its
            #  representation can be matched
            #   against the sentence embedding?
            # we switch img and text because AttnGAN wants similarity between
            # 1) text-weighted visual embs and 2) text embs,
            # and we want similarity between 1) visual obj embs and 2) vis-obj weighted struct embs
            txt_embs, txt_lens = outputs["text_embed"], outputs["text_lens"]
            if self.cfg.detr.struct_loss_input == "tree_pos_embs":
                txt_embs, txt_lens = outputs["span_tree_embed"], outputs["span_tree_embed_mask"]
                txt_lens = txt_lens.sum(-1)
            struct_loss, _ = self.struct_loss_fn(
                txt_embs=txt_embs,
                txt_lens=txt_lens,
                img_embs=outputs["obj_embed"],
                img_lens=outputs["obj_lens"],
            )
            loss += self.struct_loss_wt * struct_loss

        # this loss is only used during training so OK
        bs = label_logits.shape[0]
        all_objects = label_logits.argmax(-1)
        all_objects = all_objects[all_objects.ne(self.padding_idx) & all_objects.ne(self.eos_idx)]
        avg_num_objects = (all_objects != outputs["label_logits"].shape[-1] - 1).sum().div(bs)
        avg_num_nobj = (all_objects == outputs["label_logits"].shape[-1] - 1).sum().div(bs)

        losses = {
            "loss_ce_nll": label_nll_loss,
            "loss_bbox_prob_nll": bbox_nll_loss,
            "loss_ce_smooth": label_smoothed_loss,
            "avg_num_objects": avg_num_objects,
            "avg_num_nobj": avg_num_nobj,
            "loss_ce": label_loss,
            "loss_bbox_prob": bbox_loss,
            "struct_loss": struct_loss,
            "loss": loss,
            "loss_bbox": loss_l1,
            "loss_bbox_prop": loss_prop,
            "loss_bbox_rel": loss_rel,
            "loss_giou": loss_giou,
        }
        return losses

    def compute_regression_bbox_loss(self, bbox_logits, target_bboxes):
        padding_mask = target_bboxes.ne(self.pos_cont_pad_id)
        idx_1 = torch.repeat_interleave(
            torch.arange(0, padding_mask.size(0)), padding_mask.size(1)
        )[padding_mask.all(dim=-1).view(-1)]
        idx_2 = torch.arange(0, padding_mask.size(1)).repeat(padding_mask.size(0))[
            padding_mask.all(dim=-1).view(-1)
        ]
        idx = idx_1, idx_2
        preds_wo_pad = bbox_logits[padding_mask].view(-1, 4)
        targets_wo_pad = target_bboxes[padding_mask].view(-1, 4)

        num_boxes, num_coords = preds_wo_pad.shape

        bbox_loss_l1 = (
            F.l1_loss(preds_wo_pad, targets_wo_pad, reduction="none").sum() / num_boxes / num_coords
        )

        # src_bbox_prop = preds_wo_pad[:, 2] / preds_wo_pad[:, 3]
        # target_bbox_prop = targets_wo_pad[:, 2] / targets_wo_pad[:, 3]
        src_bbox_prop = (preds_wo_pad[:, 2] - preds_wo_pad[:, 3]) / (
            preds_wo_pad[:, 2] + preds_wo_pad[:, 3]
        )
        target_bbox_prop = (targets_wo_pad[:, 2] - targets_wo_pad[:, 3]) / (
            targets_wo_pad[:, 2] + targets_wo_pad[:, 3]
        )

        loss_bbox_prop = F.l1_loss(src_bbox_prop, target_bbox_prop, reduction="none")
        loss_bbox_prop = loss_bbox_prop.sum() / len(targets_wo_pad)

        loss_bbox_rel = self._get_rel_bbox_loss(preds_wo_pad, targets_wo_pad, idx, rescale=True)

        bbox_loss_giou = 1 - torch.diag(
            generalized_box_iou(
                box_cxcywh_to_xyxy(preds_wo_pad), box_cxcywh_to_xyxy(targets_wo_pad)
            )
        )
        bbox_loss_giou = bbox_loss_giou.sum() / num_boxes / num_coords

        tot_loss = (
            self.bbox_loss_wt * bbox_loss_l1
            + self.giou_loss_wt * bbox_loss_giou
            + self.bbox_rel_loss_wt * loss_bbox_rel
            + self.bbox_prop_loss_wt * loss_bbox_prop
        )
        return tot_loss, bbox_loss_l1, bbox_loss_giou, loss_bbox_rel, loss_bbox_prop

    def _get_rel_bbox_loss(self, src_boxes, target_boxes, idx, rescale=True):
        assert torch.equal(idx[0], torch.sort(idx[0]).values)

        if self.bbox_rel_loss_type == "abs":
            src_distances = torch.cdist(
                src_boxes[:, :2].contiguous(), src_boxes[:, :2].contiguous(), p=2.0
            )
            tgt_distances = torch.cdist(
                target_boxes[:, :2].contiguous(), target_boxes[:, :2].contiguous(), p=2.0
            )
            rel_losses = torch.abs(src_distances - tgt_distances)
        elif self.bbox_rel_loss_type == "diag_scaled":
            src_distances = torch.cdist(
                src_boxes[:, :2].contiguous(), src_boxes[:, :2].contiguous(), p=2.0
            ) / (torch.sqrt(src_boxes[:, 2].pow(2) + src_boxes[:, 3].pow(2)) + 1e-6)
            tgt_distances = torch.cdist(
                target_boxes[:, :2].contiguous(), target_boxes[:, :2].contiguous(), p=2.0
            ) / (torch.sqrt(target_boxes[:, 2].pow(2) + target_boxes[:, 3].pow(2)) + 1e-6)
            rel_losses = torch.abs(src_distances - tgt_distances)
        else:
            raise Exception("unknown bbox_rel_loss_type: '{}'".format(self.bbox_rel_loss_type))

        b_weights = torch.zeros((torch.max(idx[0]) + 1, idx[0].size(0))).type_as(rel_losses)
        b_weights[idx[0], torch.arange(0, idx[0].size(0)).long()] = 1
        b_counts = torch.bincount(idx[0]).type_as(b_weights).long()
        b_weights = torch.repeat_interleave(b_weights, b_counts, dim=0)

        rel_losses *= b_weights
        rel_loss = rel_losses.sum()
        if rescale:
            rel_loss /= b_weights.sum() - idx[0].size(0) + 1e-6
        return rel_loss

    def compute_prob_bbox_loss(self, bbox_logits, decoded_bbox_preds, target_bboxes):
        bbox_logprobs = F.log_softmax(bbox_logits, dim=-1)
        target_bboxes_cont = self.pos_dict.decode_tensor(target_bboxes)
        bbox_nll_losses, bbox_smooth_losses = zip(*[
            self.compute_prob_bbox_loss_for_dim(
                target_bboxes[:, :, i],
                bbox_logprobs[:, :, i],
                target_bboxes_cont[:, :, i],
                decoded_bbox_preds[:, :, i],
            )
            for i in range(target_bboxes.size(-1))
        ])
        # div by number of instances and number of coords (4)
        bbox_smooth_loss = torch.cat(bbox_smooth_losses).sum() / target_bboxes.size(-1)
        bbox_nll_loss = torch.cat(bbox_nll_losses).sum() / target_bboxes.size(-1)
        # div by number of classes (grid)
        bbox_loss = (1.0 - self.eps_pos) * bbox_nll_loss + self.eps_pos / bbox_logprobs.size(
            -1
        ) * bbox_smooth_loss
        bbox_loss *= self.pos_wt
        return bbox_loss, bbox_nll_loss

    def compute_prob_bbox_loss_for_dim(self, target, logprobs, cont_target=None, cont_output=None):
        """
        :param target:          BS x num_obj
        :param logprobs:        BS x num_obj x pos_tokens
        :param cont_target:     BS x num_obj
        :param cont_output:     BS x num_obj
        :return:
        """
        # BS * num_obj x pos_tokens
        logprobs = logprobs.view(-1, logprobs.size(-1))
        # BS * num_obj
        target = target.reshape(-1)
        cont_target = cont_target.reshape(-1)
        cont_output = cont_output.reshape(-1)

        pos_non_pad_mask = target.ne(self.pos_padding_id)
        # nb_non_pad x pos_tokens
        logprobs = logprobs[pos_non_pad_mask]
        # nb_non_pad
        target = target[pos_non_pad_mask]
        cont_target = cont_target[pos_non_pad_mask]
        cont_output = cont_output[pos_non_pad_mask]
        # nb_non_pad x 1
        target = target.unsqueeze(-1)

        # nb_non_pad
        pos_nll_loss = -logprobs.gather(dim=-1, index=target).squeeze(-1)
        pos_smooth_loss = -logprobs.sum(dim=-1, keepdim=True).squeeze(-1)

        if self.weight_prob_bbox_loss_by_dist:
            assert (cont_target, cont_output) != (None, None)
            cont_difference = cont_target.sub(cont_output).abs()
            pos_nll_loss *= cont_difference
            pos_smooth_loss *= cont_difference

        return pos_nll_loss, pos_smooth_loss
