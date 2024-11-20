# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
from typing import Any, Dict

import torch
import torch.nn.functional as F
from config import Config
from data.dictionary import CategoryDictionary, PositionDictionary
from model.loss.structure import (
    MultiObjectContrastiveAttnLoss,
    MultiObjectContrastiveMaxLoss,
)
from model.loss.utils import generalized_box_iou
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn

from utils import box_cxcywh_to_xyxy


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class SetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, cfg: Config, cat_dict: CategoryDictionary, matcher, weight_dict, losses):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories,
            omitting the special no-object category
            matcher: module able to compute a matching
            between targets and proposals
            weight_dict: dict containing as key the names of the
            losses and as values their relative weight.
            nobj_coef: relative classification weight applied to
            the no-object category
            losses: list of all the losses to be applied.
            See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = len(cat_dict)
        self.cat_dict = cat_dict
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.nobj_coef = cfg.detr.nobj_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes)
        empty_weight[-1] = self.nobj_coef
        self.register_buffer("empty_weight", empty_weight)
        self.prob_bboxes = cfg.detr.probabilistic_bbox_predictions
        self.weight_prob_bbox_loss_by_dist = cfg.detr.weight_prob_bbox_loss_by_dist
        self.eps = cfg.detr.label_smoothing
        self.eps_pos = cfg.detr.label_smoothing_pos
        self.autoregressive = cfg.model.autoregressive
        self.recompute_indices_for_aux_losses = cfg.detr.recompute_indices_for_aux_losses
        self.bbox_rel_loss_type = cfg.detr.bbox_rel_loss_type
        self.cfg = cfg

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "label_logits" in outputs
        src_logits = outputs["label_logits"]  # shape BS x L x C

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])

        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes - 1, dtype=torch.int64
        ).type_as(
            target_classes_o
        )  # , device=src_logits.device
        target_classes[idx] = target_classes_o
        # cross_entropy wants preds shape BS x C x L and target shape BS x L
        loss_ce_nll = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        loss_ce_smooth = (
            -F.log_softmax(src_logits, dim=-1).sum() / src_logits.shape[0] / src_logits.shape[1]
        )
        loss_ce = (1 - self.eps) * loss_ce_nll + self.eps / src_logits.shape[2] * loss_ce_smooth

        losses = {"loss_ce": loss_ce, "loss_ce_nll": loss_ce_nll, "loss_ce_smooth": loss_ce_smooth}

        if log:
            losses["class_error"] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """Compute the cardinality error, ie the absolute error
        in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only.
        It doesn't propagate gradients
        """
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets]).type_as(
            targets[0]["boxes_discr"]
        )

        if self.autoregressive and "gen_labels" in outputs:
            labels = outputs["gen_labels"][:, 1:]
        else:  # todo sample from detr
            labels = outputs["label_logits"].argmax(-1)

        # Count the number of predictions that are NOT "no-object" (which is the last class)
        obj_mask = labels.ne(self.num_classes - 1) & outputs["mask"]
        card_pred = obj_mask.sum(-1).float()
        avg_num_objects = card_pred.mean()
        avg_num_nobj = outputs["mask"].sum(-1).float().mean() - avg_num_objects

        card_err = F.l1_loss(card_pred, tgt_lengths.float())
        losses = {
            "cardinality_error": card_err,
            "avg_num_objects": avg_num_objects,
            "avg_num_nobj": avg_num_nobj,
        }
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes,
        the L1 regression loss and the GIoU loss
        targets dicts must contain the key "boxes" containing a
        tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, w, h),
        normalized by the image size.
        """
        assert "bbox_logits" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["bbox_logits"][idx]

        losses = {}
        if self.prob_bboxes:
            target_boxes = torch.cat(
                [t["boxes_discr"][i] for t, (_, i) in zip(targets, indices)], dim=0
            )

            num_bxs, num_coords, num_pos = src_boxes.shape
            src_lprobs = F.log_softmax(src_boxes, dim=-1)
            loss_bbox_nll = -src_lprobs.gather(dim=-1, index=target_boxes.unsqueeze(-1)).squeeze(-1)
            loss_bbox_smooth = -src_lprobs.sum(dim=-1, keepdim=True).squeeze(-1)

            if self.weight_prob_bbox_loss_by_dist:
                decoded_src_bboxes = outputs["bbox_preds"][idx]
                cont_tgt_boxes = torch.cat(
                    [t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0
                )
                cont_difference = cont_tgt_boxes.sub(decoded_src_bboxes).abs()
                # print(cont_difference)
                # print(pos_nll_loss.shape)
                loss_bbox_nll *= cont_difference

                # todo?
                loss_bbox_smooth *= cont_difference

            # div by number of instances and number of coords (4)
            loss_bbox_nll = loss_bbox_nll.sum() / num_boxes / num_coords
            loss_bbox_smooth = loss_bbox_smooth.sum() / num_boxes / num_coords

            losses.update({
                "loss_bbox_prob": (
                    1 - self.eps_pos
                ) * loss_bbox_nll + self.eps_pos / num_pos * loss_bbox_smooth,
                "loss_bbox_prob_nll": loss_bbox_nll,
                "loss_bbox_smooth": loss_bbox_smooth,
            })
        else:
            target_boxes = torch.cat(
                [t["boxes_cont"][i] for t, (_, i) in zip(targets, indices)], dim=0
            )
            num_bxs, num_coords = src_boxes.shape

            loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")
            losses["loss_bbox"] = loss_bbox.sum() / num_boxes / num_coords

            src_bbox_prop = (src_boxes[:, 2] - src_boxes[:, 3]) / (
                src_boxes[:, 2] + src_boxes[:, 3]
            )
            target_bbox_prop = (target_boxes[:, 2] - target_boxes[:, 3]) / (
                target_boxes[:, 2] + target_boxes[:, 3]
            )
            loss_bbox_prop = F.l1_loss(src_bbox_prop, target_bbox_prop, reduction="none")
            losses["loss_bbox_prop"] = (
                loss_bbox_prop.sum() / num_boxes
                if (self.cfg.old_prop_loss_nan or not loss_bbox_prop.isnan().sum() > 0)
                else 0.0
            )

            loss_bbox_rel = self._get_rel_bbox_loss(src_boxes, target_boxes, idx, rescale=True)
            losses["loss_bbox_rel"] = loss_bbox_rel if not loss_bbox_rel.isnan() else 0.0

            loss_giou = 1 - torch.diag(
                generalized_box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
            )
            losses["loss_giou"] = loss_giou.sum() / num_boxes / num_coords
        return losses

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

        if idx[0].numel() > 0:
            b_weights = torch.zeros((torch.max(idx[0]) + 1, idx[0].size(0))).type_as(rel_losses)
            b_weights[idx[0], torch.arange(0, idx[0].size(0)).long()] = 1
            b_counts = torch.bincount(idx[0]).type_as(b_weights).long()
            b_weights = torch.repeat_interleave(b_weights, b_counts, dim=0)
        else:
            b_weights = torch.tensor(1.0).type_as(rel_losses)

        rel_losses *= b_weights
        rel_loss = rel_losses.sum()
        if rescale:
            rel_loss /= b_weights.sum() - idx[0].size(0) + 1e-6
        if rel_loss > 100:
            rel_loss *= 1.0
        return rel_loss

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            "labels": self.loss_labels,
            "cardinality": self.loss_cardinality,
            "boxes": self.loss_boxes,
            # 'masks': self.loss_masks
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification
             of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied,
                       see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with
        # the output of each intermediate layer.
        if "aux_outputs" in outputs:
            losses["aux_losses"] = []
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = (
                    self.matcher(aux_outputs, targets)
                    if self.recompute_indices_for_aux_losses
                    else indices
                )
                l_dict = {}
                for loss in self.losses:
                    if loss == "masks":
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == "labels":
                        # Logging is enabled only for the last layer
                        kwargs = {"log": False}
                    l_dict.update(
                        self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    )
                    # l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                l_dict["layer_num"] = i
                losses["aux_losses"].append(l_dict)

        return losses


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and
    the predictions of the network

    For efficiency reasons, the targets don't include the no_object.
    Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1
    matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(
        self,
        prob_boxes: bool,
        cost_class: float = 1,
        cost_bbox: float = 1,
        cost_giou: float = 1,
        cost_prob_bbox: float = 1,
    ):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the
            classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the
            bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the
            bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.cost_prob_bbox = cost_prob_bbox
        assert (
            cost_class != 0
            or cost_bbox != 0
            or cost_giou != 0
            or (prob_boxes and cost_prob_bbox != 0)
        ), "all costs cant be 0"
        self.prob_boxes = prob_boxes

    @torch.no_grad()
    def forward(self, outputs, targets):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "label_logits": Tensor of dim [batch_size, num_queries, num_classes]
                 with the classification logits
                 "bbox_preds": Tensor of dim [batch_size, num_queries, 4]
                 with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size),
            where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes]
                 (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4]
                 containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["label_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = (
            outputs["label_logits"].flatten(0, 1).softmax(-1)
        )  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["bbox_preds"].flatten(0, 1)  # [batch_size * num_queries, 4]

        if self.prob_boxes:
            out_bbox_prob = (
                outputs["bbox_logits"].softmax(-1).flatten(0, 1)
            )  # [bs * nq, 4, num_pos]

            tgt_bbox_discr = torch.cat([v["boxes_discr"] for v in targets])  # [num_gt_boxes, 4]
            # [4, bs * nq, num_gt_boxes]
            cost_per_coord = -torch.stack([
                pred[:, tgt]
                for pred, tgt in zip(out_bbox_prob.transpose(0, 1), tgt_bbox_discr.transpose(0, 1))
            ]).type_as(out_bbox_prob)
            cost_bbox_prob = cost_per_coord.mean(dim=0)

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes" if self.prob_boxes else "boxes_cont"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost between boxes
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        if self.prob_boxes:
            C += self.cost_prob_bbox * cost_bbox_prob
        C = C.view(bs, num_queries, -1).cpu()

        # todo this doesn't work anymore if eos in labels has no box
        sizes = [len(v["boxes"]) for v in targets]
        # indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        indices = [
            linear_sum_assignment(c[i][outputs["mask"][i]] if outputs["mask"] is not None else c[i])
            # split last dim, corresponding to targets, in sizes
            for i, c in enumerate(C.split(sizes, -1))
        ]
        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]


class Txt2ImgSetCriterion(SetCriterion):
    def __init__(
        self, cfg: Config, category_dict: CategoryDictionary, pos_dict: PositionDictionary
    ):
        losses = ["labels", "boxes", "cardinality"]
        weight_dict = {"loss_ce": cfg.detr.label_loss_coef}
        if cfg.detr.probabilistic_bbox_predictions:
            weight_dict["loss_bbox_prob"] = cfg.detr.prob_bbox_loss_coef
        else:
            weight_dict.update({
                "loss_bbox": cfg.detr.bbox_loss_coef,
                "loss_giou": cfg.detr.giou_loss_coef,
                "loss_bbox_prop": cfg.detr.bbox_prop_loss_coef,
                "loss_bbox_rel": cfg.detr.bbox_rel_loss_coef,
            })

        matcher = HungarianMatcher(
            cfg.detr.probabilistic_bbox_predictions,
            cfg.detr.set_cost_class_coef,
            cfg.detr.set_cost_bbox_coef,
            cfg.detr.set_cost_giou_coef,
            cfg.detr.set_cost_prob_bbox_coef,
        )
        super().__init__(cfg, category_dict, matcher, weight_dict, losses)

        self.struct_loss_weight = cfg.detr.struct_loss_coef
        if self.struct_loss_weight > 0.0:
            struct_loss_class = {
                "max": MultiObjectContrastiveMaxLoss,
                "attn": MultiObjectContrastiveAttnLoss,
            }[cfg.detr.struct_loss_type]
            self.struct_loss_fn = struct_loss_class(cfg.detr)

        assert category_dict.symbols[-1] == "<NOBJ>"
        self.category_dict = category_dict
        self.pos_dict = pos_dict
        self.pos_cont_pad_id = cfg.pos_cont_pad_id
        self.pos_pad_id = pos_dict.pad()
        self.label_pad_id = category_dict.pad()
        self.label_bos_id = category_dict.bos()
        self.label_eos_id = category_dict.eos()
        self.cfg = cfg
        self.eps_length = cfg.model.length_label_smoothing
        self.max_length = cfg.model.max_target_positions

    def forward(self, outputs: Dict[str, Any], batch, log_aux_losses=True):
        # outputs = outputs.clone()

        assert self.cfg.model.ignore_eos_loss
        # replace eos token by pad
        if self.cfg.model.ignore_eos_loss:
            batch["labels"].masked_fill_(
                batch["labels"].eq(self.label_eos_id) | batch["labels"].eq(self.label_bos_id),
                self.label_pad_id,
            )

        # group targets per sample and select only non padded elements
        targets = [
            {
                "labels": labels[
                    labels.ne(self.label_pad_id)
                ],  # & labels.ne(self.category_dict.eos())
                "boxes": self.pos_dict.decode_tensor(
                    boxes_discr[boxes_discr.ne(self.pos_pad_id)].view(-1, 4)
                ),
                "boxes_discr": boxes_discr[boxes_discr.ne(self.pos_pad_id)].view(-1, 4),
                "boxes_cont": boxes_cont[boxes_discr.ne(self.pos_pad_id)].view(-1, 4),
            }
            for labels, boxes_discr, boxes_cont in zip(
                batch["labels"], batch["bboxes"], batch["bboxes_cont"]
            )
        ]

        if self.cfg.model.autoregressive:
            # tgt_padding_mask defines mask over input,
            # which starts from bos, but we want to use masks to select
            # logits, which start from the first word after bos
            labels = outputs["gen_labels"][:, 1:]
            # we don't care about the eos token for this loss
            outputs["mask"] = (
                labels.ne(self.label_pad_id)
                & labels.ne(self.label_eos_id)
                & labels.ne(self.label_bos_id)
            )
        else:
            outputs["mask"] = ~outputs["tgt_padding_mask"]

        # if 'aux_outputs' in outputs:
        if self.cfg.detr.aux_loss:
            outputs["aux_outputs"] = [
                {
                    "label_logits": a,
                    "bbox_logits": b,
                    "bbox_preds": (
                        self.pos_dict.decode_tensor(b.argmax(-1)).type_as(b)
                        if self.prob_bboxes
                        else b
                    ),
                    "mask": outputs["mask"],
                }
                for a, b in zip(
                    outputs["aux_outputs"]["label_logits"], outputs["aux_outputs"]["outputs_bbox"]
                )
            ]

        # compute set losses
        loss_dict = super().forward(outputs, targets)

        # compute weighted sum
        # 'cardinality_error' not included bc not in weight dict
        losses = sum(loss_dict[k] * self.weight_dict[k] for k in self.weight_dict)

        if self.struct_loss_weight > 0.0:
            # always predict one box that is the entire image
            #  so its representation can be matched
            #   against the sentence embedding?
            txt_embs, txt_lens = outputs["text_embed"], outputs["text_lens"]
            if self.cfg.detr.struct_loss_input == "tree_pos_embs":
                txt_embs, txt_lens = outputs["span_tree_embed"], outputs["span_tree_embed_mask"]
                txt_lens = txt_lens.sum(-1)
            struct_loss, sim_scores = self.struct_loss_fn(
                txt_embs=txt_embs,
                txt_lens=txt_lens,
                img_embs=outputs["obj_embed"],
                img_lens=outputs["obj_lens"],
            )
            loss_dict["struct_loss"] = struct_loss
            # loss_dict["sim_scores"] = sim_scores.detach().cpu()
            losses += self.struct_loss_weight * struct_loss

        # weighted sum of aux losses, reformat aux losses
        if self.cfg.detr.aux_loss:
            for aux_loss_dict in loss_dict["aux_losses"]:
                losses += sum(aux_loss_dict[k] * self.weight_dict[k] for k in self.weight_dict)

            # restructure for easier logging
            if log_aux_losses:
                loss_dict.update({
                    f'{k}_{dct["layer_num"]}': v
                    for dct in loss_dict["aux_losses"]
                    for k, v in dct.items()
                    if not k.startswith("layer_num")
                })
            loss_dict.pop("aux_losses", None)

        # compute and add length loss todo to SetCriterion?
        if self.cfg.model.predict_num_queries and not self.cfg.model.autoregressive:
            length_loss, length_loss_nll = self.loss_length(batch, outputs)

            losses += self.cfg.model.length_loss_coef * length_loss
            loss_dict.update({
                "length_loss": length_loss.detach(), "length_loss_nll": length_loss_nll.detach()
            })

        # this does not exclude NOBJ predictions
        loss_dict["avg_predicted_length"] = outputs["mask"].sum(-1).float().mean()
        loss_dict["loss"] = losses

        return loss_dict

    def loss_length(self, batch, outputs):
        length_lprobs: Tensor = F.log_softmax(outputs["predicted_lengths"], dim=-1)
        length_target = batch["labels"].ne(self.label_pad_id).sum(dim=-1)
        # - 1 because highest element = element 0 means length 1
        length_target = (length_target - 1).clamp(min=0)

        length_loss_nll = -length_lprobs.gather(dim=-1, index=length_target.unsqueeze(-1)).squeeze(
            -1
        )
        length_loss_smooth = -length_lprobs.sum(dim=-1)

        if self.weight_prob_bbox_loss_by_dist:
            predicted_lengths = length_lprobs.argmax(dim=-1).float()
            cont_difference = length_target.sub(predicted_lengths).abs() / self.max_length
            length_loss_nll *= cont_difference

            # todo?
            length_loss_smooth *= cont_difference

        length_loss_nll = length_loss_nll.mean()
        length_loss_smooth = length_loss_smooth.mean()
        length_loss = (
            1 - self.eps_length
        ) * length_loss_nll + self.eps_length / length_lprobs.shape[-1] * length_loss_smooth
        return length_loss, length_loss_nll
