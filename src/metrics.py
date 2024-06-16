import logging
import math
from collections import defaultdict
from functools import partial
from pprint import pprint
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import scipy
import torch
import torchvision
from iglovikov_helper_functions.utils.general_utils import group_by_key
from torch.nn import functional as F

from data.dictionary import CategoryDictionary

logger = logging.getLogger("pytorch_lightning")


def box_xywh_to_xyxy(boxes):
    xmin, ymin, w, h = boxes.unbind(-1)
    return torch.stack((xmin, ymin, xmin + w, ymin + h), dim=-1)


def compute_precision_recall(
    indexed_by_img_id,
    cat_dict: CategoryDictionary,
    # exclude_nobj=True,
    gt_anns=None,
    log=False,
    replaced_only=False,
):
    if gt_anns is not None:
        raise NotImplementedError
        gt_grouped_by_id = group_by_key(list(gt_anns.values()), "image_id")
        gt_indexed_by_img_id = {
            img_id: {
                "labels": (
                    [pred["category_id"] for pred in gt_grouped_by_id[img_id]]
                    # not all imgs have instance annotations, but all imgs have captions!
                    if img_id in gt_grouped_by_id
                    else []
                ),
            }
            for img_id in indexed_by_img_id.keys()
        }
    else:
        gt_indexed_by_img_id = {
            img_id: {
                "labels": (
                    dct["gt_labels"]
                    if not replaced_only
                    # else dct['gt_labels'][dct['replaced']]
                    else [l for (l, r) in zip(dct["gt_labels"], dct["replaced"]) if r]
                )
            }
            for img_id, dct in indexed_by_img_id.items()
        }

    num_classes = cat_dict.max_coco_category + 1
    # average over instances within sample, then over samples
    sample_avg_precision, sample_avg_recall = 0.0, 0.0
    num_samples_with_preds, num_samples_with_gt = 0, 0
    # only needed for first averaging over samples then over classes
    # class_precisions, class_recalls, divisor = 3 * [np.zeros(num_classes)]
    # sum over samples then average over classes
    all_FPs, all_TPs, all_FNs = 3 * [np.zeros(num_classes)]
    all_ps, all_gts = 2 * [np.zeros(num_classes)]

    keys = list(indexed_by_img_id.keys())
    assert len(keys) > 0
    for key in keys:
        lgt = gt_indexed_by_img_id[key]["labels"]
        lp = indexed_by_img_id[key]["labels"]
        # if exclude_nobj:
        lp = [lab for lab in lp if lab != cat_dict.nobj_coco_id]

        gt_unique, gt_counts = np.unique(lgt, return_counts=True)
        p_unique, p_counts = np.unique(lp, return_counts=True)
        gt = np.zeros(num_classes)
        p = np.zeros(num_classes)
        if len(lgt) > 0:
            np.put_along_axis(gt, indices=gt_unique, values=gt_counts, axis=0)
        if len(lp) > 0:
            np.put_along_axis(p, indices=p_unique, values=p_counts, axis=0)

        delta = p - gt
        num_preds = np.sum(p)
        num_gts = np.sum(gt)

        # over all instances, for one sample
        FP = np.sum(delta[delta > 0.0])
        TP = num_preds - FP
        FN = np.abs(np.sum(delta[delta < 0.0]))

        if num_preds > 0:
            precision = TP / (TP + FP)
            sample_avg_precision += precision
            num_samples_with_preds += 1
        if num_gts > 0:
            recall = TP / (TP + FN)
            sample_avg_recall += recall
            num_samples_with_gt += 1

        # per class, over all samples
        FPs = delta.clip(0.0)
        TPs = p - FPs
        FNs = (-delta).clip(0.0)

        all_ps, all_gts = all_ps + p, all_gts + gt
        all_FPs, all_TPs, all_FNs = all_FPs + FPs, all_TPs + TPs, all_FNs + FNs

    class_avg_precisions = np.divide(all_TPs, (all_FPs + all_TPs), where=(all_FPs + all_TPs) != 0)
    class_avg_recalls = np.divide(all_TPs, (all_FNs + all_TPs), where=(all_FNs + all_TPs) != 0)
    # average over classes
    class_avg_prec = np.mean(class_avg_precisions[all_ps.nonzero()[0]])
    class_avg_rec = np.mean(class_avg_recalls[all_gts.nonzero()[0]])

    # sample average
    sample_avg_precision = (
        sample_avg_precision / num_samples_with_preds if num_samples_with_preds > 0 else 0
    )
    sample_avg_recall = sample_avg_recall / num_samples_with_gt if num_samples_with_gt > 0 else 0

    # micro average (average over all instances)
    num_all_preds = np.sum(all_ps)
    num_all_gts = np.sum(all_gts)
    micro_FP = np.sum(all_FPs)
    micro_TP = np.sum(all_TPs)
    assert micro_TP == num_all_preds - micro_FP
    micro_FN = np.sum(all_FNs)
    micro_precision = micro_TP / (micro_TP + micro_FP) if num_all_preds > 0 else 0
    micro_recall = micro_TP / (micro_TP + micro_FN) if num_all_gts > 0 else 0

    if log:
        log_precision_recalls(
            cat_dict,
            all_gts,
            all_ps,
            class_avg_prec,
            class_avg_precisions,
            class_avg_rec,
            class_avg_recalls,
            micro_precision,
            micro_recall,
            sample_avg_precision,
            sample_avg_recall,
        )

    return (
        sample_avg_precision,
        sample_avg_recall,
        class_avg_prec,
        class_avg_rec,
        micro_precision,
        micro_recall,
        all_ps,
        class_avg_precisions,
        all_gts,
        class_avg_recalls,
    )


def log_precision_recalls(
    cat_dict,
    all_gts,
    all_ps,
    class_avg_prec,
    class_avg_precisions,
    class_avg_rec,
    class_avg_recalls,
    micro_precision,
    micro_recall,
    sample_avg_precision,
    sample_avg_recall,
):
    print("PRECISION - label, count, name, prec (per class, avg over all instances)")
    pprint(
        list(
            zip(
                all_ps.nonzero()[0],
                all_ps[all_ps.nonzero()[0]],
                cat_dict.convert_coco_ids_to_coco_names(all_ps.nonzero()[0]),
                class_avg_precisions[all_ps.nonzero()[0]],
            )
        )
    )
    print("RECALL - label, count, name, rec (per class, avg over all instances)")
    pprint(
        list(
            zip(
                all_gts.nonzero()[0],
                all_gts[all_gts.nonzero()[0]],
                cat_dict.convert_coco_ids_to_coco_names(all_gts.nonzero()[0]),
                class_avg_recalls[all_gts.nonzero()[0]],
            )
        )
    )
    print("PRECISION (avg 1st over instances within class, then over classes)")
    print(class_avg_prec)
    print("RECALL (avg 1st over instances within class, then over classes)")
    print(class_avg_rec)
    print("PRECISION (avg 1st over instances within sample, then over samples)")
    print(sample_avg_precision)
    print("RECALL (avg 1st over instances within sample, then over samples)")
    print(sample_avg_recall)
    print("PRECISION (avg over all instances)")
    print(micro_precision)
    print("RECALL (avg over all instances)")
    print(micro_recall)


def compute_precision_recall_iou(
    indexed_by_img_id,
    cat_dict: CategoryDictionary,
    iou: float,
    gt_anns=None,
    log=False,
    flip_boxes=True,
    replaced_only=False,
):
    """
        CAUTION: do NOT use this method for metrics without IoU threshold,
            since the max/argmax operators in
            iglovikov_helper_functions.metrics.map.recall_precision will
            return the same elements if all overlaps are 0.0, so incorrect!
    :param iou:
    :param log:
    :param gt_anns:
    :param indexed_by_img_id:
    :param cat_dict:
    :return:
    """
    from coco_eval import CocoEvaluator

    assert iou > 0.0
    prepare_for_coco_det = partial(CocoEvaluator.prepare_for_coco_detection, None)
    num_classes = cat_dict.max_coco_category + 1

    gt_anns_list = prepare_gt_annotations(
        gt_anns, indexed_by_img_id, prepare_for_coco_det, replaced_only
    )
    gt_by_cat = group_by_key(gt_anns_list, "category_id")

    # if len(indexed_by_img_id) > 1:
    gt_by_img_id = group_by_key(gt_anns_list, "image_id")
    running_p, running_r, tot_nonzero_p, tot_nonzero_gt = 0.0, 0.0, 0.0, 0.0

    which_preds_flipped = torch.zeros(len(indexed_by_img_id))
    list_preds = list(indexed_by_img_id.items())

    for i, (img_id, preds) in enumerate(list_preds):
        gt = group_by_key(gt_by_img_id[img_id], "category_id")

        preds_per_instance = prepare_for_coco_det({img_id: preds})
        ps, precs, gts, recs = compute_precision_recall_iou_per_category(
            preds_per_instance, gt, cat_dict.nobj_coco_id, num_classes, iou
        )
        flipped_preds = {
            **preds,
            "boxes": preds["boxes_flipped"] if flip_boxes else preds["boxes"],
            "boxes_flipped": preds["boxes"] if flip_boxes else preds["boxes_flipped"],
            "norm_xywh_boxes": preds["norm_xywh_boxes_flipped"],
            "norm_xywh_boxes_flipped": preds["norm_xywh_boxes"],
            "norm_cxcywh_boxes": preds["norm_cxcywh_boxes_flipped"],
            "norm_cxcywh_boxes_flipped": preds["norm_cxcywh_boxes"],
        }
        flipped_preds_per_instance = prepare_for_coco_det({img_id: flipped_preds})
        fps, fprecs, fgts, frecs = compute_precision_recall_iou_per_category(
            flipped_preds_per_instance, gt, cat_dict.nobj_coco_id, num_classes, iou
        )

        prec = np.sum(ps * precs) / np.sum(ps) if np.sum(ps) > 0 else 0
        rec = np.sum(gts * recs) / np.sum(gts) if np.sum(gts) > 0 else 0
        fprec = np.sum(fps * fprecs) / np.sum(fps) if np.sum(fps) > 0 else 0
        frec = np.sum(fgts * frecs) / np.sum(fgts) if np.sum(fgts) > 0 else 0

        if fprec > prec:  # and False:
            which_preds_flipped[i] = 1
            prec = fprec
            rec = frec
            indexed_by_img_id[img_id] = flipped_preds

        running_p, running_r = running_p + prec, running_r + rec
        tot_nonzero_p = tot_nonzero_p + (1 if len(preds["labels"]) > 0 else 0)
        tot_nonzero_gt = tot_nonzero_gt + (1 if len(gt_by_img_id[img_id]) > 0 else 0)

    # over instances within sample, then over samples
    sample_avg_prec = running_p / tot_nonzero_p if tot_nonzero_p > 0 else 0
    sample_avg_rec = running_r / tot_nonzero_gt if tot_nonzero_gt > 0 else 0

    # over instances within class, then over classes
    preds_per_instance = prepare_for_coco_det(indexed_by_img_id)
    all_ps, cl_a_precs, all_gts, cl_a_recs = compute_precision_recall_iou_per_category(
        preds_per_instance, gt_by_cat, cat_dict.nobj_coco_id, num_classes, iou
    )
    class_avg_prec = np.mean(cl_a_precs[all_ps.nonzero()[0]])
    class_avg_rec = np.mean(cl_a_recs[all_gts.nonzero()[0]])

    # over instances within class, then over classes weighted by nb instances per class = micro
    # print(all_ps, cl_a_precs, np.sum(all_ps * cl_a_precs), np.sum(all_ps))
    micro_prec = np.sum(all_ps * cl_a_precs) / np.sum(all_ps) if np.sum(all_ps) > 0 else 0
    micro_rec = np.sum(all_gts * cl_a_recs) / np.sum(all_gts) if np.sum(all_gts) > 0 else 0

    if log:
        log_precision_recalls(
            cat_dict,
            all_gts,
            all_ps,
            class_avg_prec,
            cl_a_precs,
            class_avg_rec,
            cl_a_recs,
            micro_prec,
            micro_rec,
            sample_avg_prec,
            sample_avg_rec,
        )

    return (
        sample_avg_prec,
        sample_avg_rec,
        class_avg_prec,
        class_avg_rec,
        micro_prec,
        micro_rec,
        all_ps,
        cl_a_precs,
        all_gts,
        cl_a_recs,
        indexed_by_img_id,
        float(which_preds_flipped.sum()),
    )


def prepare_gt_annotations(gt_anns, indexed_by_img_id, prepare_for_coco_det, replaced_only=False):
    if gt_anns is None:
        gt_anns_list = prepare_for_coco_det({
            img_id: {
                "boxes": (
                    pred["gt_norm_xywh_boxes"]
                    if not replaced_only
                    else pred["gt_norm_xywh_boxes"][pred["replaced"]]
                ),
                "labels": (
                    pred["gt_labels"]
                    if not replaced_only
                    else torch.tensor(pred["gt_labels"])[pred["replaced"]].tolist()
                ),
                "scores": (
                    torch.ones(len(pred["gt_labels"]))
                    if not replaced_only
                    else torch.ones(len(pred["gt_labels"]))[pred["replaced"]]
                ),
            }
            for img_id, pred in indexed_by_img_id.items()
        })
    else:
        raise NotImplementedError
    return gt_anns_list


def compute_precision_recall_iou_per_category(
    preds_per_instance,
    gt_by_cat,
    nobj_coco_id,
    num_classes,
    iou_threshold,
):
    preds_by_cat = group_by_key(preds_per_instance, "category_id")

    all_cats = list(
        set(preds_by_cat.keys()).union(set(gt_by_cat.keys())).difference({nobj_coco_id})
    )

    class_avg_recalls = np.zeros(num_classes)
    class_avg_precisions = np.zeros(num_classes)
    all_ps, all_gts = np.zeros(num_classes), np.zeros(num_classes)

    for cat_id in all_cats:
        gt_hit = cat_id in gt_by_cat
        p_hit = cat_id in preds_by_cat
        gt_by_cat_id = gt_by_cat[cat_id] if gt_hit else []
        preds_by_cat_id = preds_by_cat[cat_id] if p_hit else []

        if gt_hit and p_hit:
            rs, ps = recall_precision_own(
                gt_by_cat_id, preds_by_cat_id, iou_threshold, similarity_fn=get_iou_overlaps
            )
            class_avg_recalls[cat_id] = rs[-1]
            class_avg_precisions[cat_id] = ps[-1]

        all_gts[cat_id] = len(gt_by_cat[cat_id]) if gt_hit else 0.0
        all_ps[cat_id] = len(preds_by_cat[cat_id]) if p_hit else 0.0

    return all_ps, class_avg_precisions, all_gts, class_avg_recalls


def get_iou_overlaps(boxes1: np.ndarray, boxes2: np.ndarray):
    """
    :param boxes1:  x1, y1, w, h  format, not prop (but may be?)
    :param boxes2:
    :return:
    """
    # wants x1,y1,x2,y2 format
    return torchvision.ops.box_iou(
        box_xywh_to_xyxy(torch.from_numpy(boxes1)), box_xywh_to_xyxy(torch.from_numpy(boxes2))
    )


def recall_precision_own(
    gt: np.ndarray,
    predictions: np.ndarray,
    iou_threshold: float,
    similarity_fn: Callable = get_iou_overlaps,
) -> Tuple[np.array, np.array]:
    """
        Based on
    https://github.com/ternaus/iglovikov_helper_functions/blob/master/iglovikov_helper_functions/metrics/map.py
    https://github.com/rafaelpadilla/review_object_detection_metrics/blob/main/src/evaluators/coco_evaluator.py
    """
    num_gts = len(gt)
    image_gts = group_by_key(gt, "image_id")

    image_gt_boxes = {
        img_id: np.array([[float(z) for z in b["bbox"]] for b in boxes])
        for img_id, boxes in image_gts.items()
    }
    image_gt_checked = {img_id: np.zeros(len(boxes)) for img_id, boxes in image_gts.items()}

    predictions = sorted(predictions, key=lambda x: x["score"], reverse=True)

    # go down dets and mark TPs and FPs
    num_predictions = len(predictions)
    tp = np.zeros(num_predictions)
    fp = np.zeros(num_predictions)
    # predictions_to_gts = np.full(num_predictions, -1)

    for prediction_index, prediction in enumerate(predictions):
        box = np.array(prediction["bbox"])

        # max_overlap = -np.inf
        max_overlap = iou_threshold
        jmax = -1

        try:
            gt_boxes = image_gt_boxes[prediction["image_id"]]  # gt_boxes per image
            gt_checked = image_gt_checked[prediction["image_id"]]  # gt flags per image
        except KeyError:
            gt_boxes = []
            gt_checked = None

        if len(gt_boxes) > 0:
            similarities = similarity_fn(box[None], gt_boxes)[0]

            # max_overlap = np.max(overlaps)
            # jmax = np.argmax(overlaps)
            # max_overlap = iou_threshold
            # jmax = -1

            for j, overlap in enumerate(similarities):
                if gt_checked[j] == 0 and overlap > max_overlap:
                    max_overlap = overlap
                    jmax = j

        if jmax > -1:
            tp[prediction_index] = 1.0
            gt_checked[jmax] = 1
            # predictions_to_gts[prediction_index] = jmax
        else:
            fp[prediction_index] = 1.0
        # if max_overlap >= iou_threshold:
        #     if gt_checked[jmax] == 0:
        #         tp[prediction_index] = 1.0
        #         gt_checked[jmax] = 1
        #     else:
        #         fp[prediction_index] = 1.0
        # else:
        #     fp[prediction_index] = 1.0

    # compute precision recall
    fp = np.cumsum(fp, axis=0)
    tp = np.cumsum(tp, axis=0)

    recalls = tp / float(num_gts)

    # avoid divide by zero in case the first detection matches a difficult ground truth
    precisions = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)

    # ap = get_ap(recalls, precisions)

    return recalls, precisions  # , predictions_to_gts


def group_by_key_and_count(list_dicts: List[dict], key: Any) -> defaultdict:
    groups: defaultdict = defaultdict(list)
    for i, detection in enumerate(list_dicts):
        groups[detection[key]].append({**detection, "count": i})
    return groups


def get_center_distances(boxes: np.ndarray, gt_boxes: np.ndarray) -> np.ndarray:
    # boxes in cxcywh format!
    centers = np.stack((boxes[:, 0], boxes[:, 1]), axis=1)
    gt_centers = np.stack((gt_boxes[:, 0], gt_boxes[:, 1]), axis=1)

    distances = scipy.spatial.distance.cdist(centers, gt_centers)
    return distances


def relative_pairwise_distances(
    indexed_by_img_id: Union[List, Dict], return_all=False, gt_anns=None
):
    if isinstance(indexed_by_img_id, Dict):
        indexed_by_img_id = list(indexed_by_img_id.items())

    mdds = torch.full((len(indexed_by_img_id),), float("Inf"))
    msdds = torch.full((len(indexed_by_img_id),), float("Inf"))
    msdns = torch.full((len(indexed_by_img_id),), float("Inf"))
    mdxs = torch.full((len(indexed_by_img_id),), float("Inf"))
    mdys = torch.full((len(indexed_by_img_id),), float("Inf"))

    for i, (img_id, preds) in enumerate(indexed_by_img_id):
        ps_to_gts = compute_greedy_matching({img_id: preds}, gt_anns=gt_anns)
        matched = ps_to_gts > -1

        matched_p = preds["norm_cxcywh_boxes"][matched]
        matched_gt = preds["gt_norm_cxcywh_boxes"][ps_to_gts[matched]]
        num_objects = matched_p.shape[0]

        if num_objects < 2:
            continue

        # difference between relative distances between objects in preds and GT
        # centers_p = torch.stack((matched_p[:, 0], matched_p[:, 1]), dim=1)
        pairwise_p_dists = torch.cdist(matched_p[:, [0, 1]], matched_p[:, [0, 1]])
        pairwise_gt_dists = torch.cdist(matched_gt[:, [0, 1]], matched_gt[:, [0, 1]])
        delta_dist = pairwise_gt_dists.sub(pairwise_p_dists).abs()

        indices = torch.triu_indices(num_objects, num_objects, offset=1)
        nonzero_div = torch.full_like(pairwise_p_dists, 1e-6)

        mean_delta_dist = delta_dist[indices[0], indices[1]].mean()

        # difference between relative distances scaled by diagonal length
        # between objects in preds and GT
        # p_areas = matched_p[:, 2] * matched_p[:, 3]
        # gt_areas = matched_gt[:, 2] * matched_gt[:, 3]
        p_diags = F.pairwise_distance(matched_p[:, [2, 3]], torch.zeros_like(matched_p[:, [2, 3]]))
        gt_diags = F.pairwise_distance(
            matched_gt[:, [2, 3]], torch.zeros_like(matched_gt[:, [2, 3]])
        )
        p_divisor = (
            p_diags  # .repeat((num_objects, 1))
            + p_diags[:, None]  # .repeat((1, num_objects))
            + nonzero_div
        )
        gt_divisor = gt_diags + gt_diags[:, None] + nonzero_div
        pairwise_scaled_p_dists = pairwise_p_dists / p_divisor
        pairwise_scaled_gt_dists = pairwise_gt_dists / gt_divisor
        delta_scaled_dist = pairwise_scaled_gt_dists.sub(pairwise_scaled_p_dists).abs()
        mean_scaled_delta_dist = delta_scaled_dist[indices[0], indices[1]].mean()

        # difference between normalized differences in scale between P and GT objects
        # p=1 distance equals abs(w1 - w2) + abs(h1 - h2)

        def pw_norm_dim_diffs(boxes):
            w_sums = (boxes[:, 2] + boxes[:, 2, None]) + nonzero_div
            w_diffs = boxes[:, 2] - boxes[:, 2, None]
            h_sums = (boxes[:, 3] + boxes[:, 3, None]) + nonzero_div
            h_diffs = boxes[:, 3] - boxes[:, 3, None]
            return w_diffs.abs() / w_sums + h_diffs.abs() / h_sums

        pw_p_norm_dim_diffs = pw_norm_dim_diffs(matched_p)
        pw_gt_norm_dim_diffs = pw_norm_dim_diffs(matched_gt)
        delta_scaled_norm = pw_p_norm_dim_diffs.sub(pw_gt_norm_dim_diffs).abs()
        mean_scaled_delta_norm = delta_scaled_norm[indices[0], indices[1]].mean()

        # difference between pairwise X and Y differences scaled by sums of W / H
        # absolute value of X diff so invariant to horizontal flip
        def coord_diffs(boxes, coord, scale, absol):
            numerator = boxes[:, coord] - boxes[:, coord, None]
            denominator = (boxes[:, scale] + boxes[:, scale, None]) + nonzero_div
            return (numerator.abs() if absol else numerator) / denominator

        p_x_diffs = coord_diffs(matched_p, 0, 2, absol=True)
        p_y_diffs = coord_diffs(matched_p, 1, 3, absol=False)
        gt_x_diffs = coord_diffs(matched_gt, 0, 2, absol=True)
        gt_y_diffs = coord_diffs(matched_gt, 1, 3, absol=False)
        x_delta = p_x_diffs.sub(gt_x_diffs).abs()
        y_delta = p_y_diffs.sub(gt_y_diffs).abs()
        mean_delta_x = x_delta[indices[0], indices[1]].mean()
        mean_delta_y = y_delta[indices[0], indices[1]].mean()

        vals = (
            mean_delta_dist,
            mean_scaled_delta_dist,
            mean_scaled_delta_norm,
            mean_delta_x,
            mean_delta_y,
        )
        for tens, val in zip((mdds, msdds, msdns, mdxs, mdys), vals):
            tens[i] = val

    results = {} if not return_all else {"all": (mdds, msdds, msdns, mdxs, mdys)}
    return {
        **results,
        "delta_pairwise_distances": mdds[mdds.ne(float("Inf"))].mean(),
        "delta_pairwise_distances_scaled_diags": msdds[msdds.ne(float("Inf"))].mean(),
        "delta_pairwise_norm_width_height_diffs": msdns[msdns.ne(float("Inf"))].mean(),
        "delta_pairwise_x_distances_norm_w": mdxs[mdxs.ne(float("Inf"))].mean(),
        "delta_pairwise_y_distances_norm_h": mdys[mdys.ne(float("Inf"))].mean(),
    }


def _get_pw_distance_matrix(coordinates, distance="euclid", coord="x"):
    if distance == "L1":
        # dists = torch.cdist(coordinates, coordinates)
        c = 0 if coord == "x" else 1
        diff = coordinates[:, c].sub(coordinates[:, c].unsqueeze(1))
        return diff
    else:
        assert distance == "euclid"
        dists = torch.cdist(coordinates, coordinates) / math.sqrt(2)
    return dists


def _get_pw_diff_matrix(preds, gt, distance):
    if gt.dim() < 2 or preds.dim() < 2:
        return torch.tensor([[]])
    if distance == "L1":
        pw_p_x_dists = (preds[:, 0] - preds[:, 0].unsqueeze(1)).abs()
        pw_p_y_dists = (preds[:, 1] - preds[:, 1].unsqueeze(1)) / 2
        pw_gt_x_dists = (gt[:, 0] - gt[:, 0].unsqueeze(1)).abs()
        pw_gt_y_dists = (gt[:, 1] - gt[:, 1].unsqueeze(1)) / 2
        x_delta = (pw_gt_x_dists - pw_p_x_dists).abs()
        y_delta = (pw_gt_y_dists - pw_p_y_dists).abs()
        return (x_delta + y_delta) / 2
    else:
        assert distance == "euclid"
        pw_p_dists = torch.cdist(preds[:, [0, 1]], preds[:, [0, 1]]) / math.sqrt(2)
        pw_gt_dists = torch.cdist(gt[:, [0, 1]], gt[:, [0, 1]]) / math.sqrt(2)
        return pw_gt_dists.sub(pw_p_dists).abs()


def relative_pairwise_similarity_pr(
    indexed_by_img_id: Union[List, Dict], return_all=False, gt_anns=None, distance="euclid"
):
    if isinstance(indexed_by_img_id, Dict):
        indexed_by_img_id = list(indexed_by_img_id.items())

    # return {
    #     'all': torch.tensor([1 for img_id, _ in indexed_by_img_id]),
    #     'pairwise_similarity_prec_rec': 1,
    # }

    all_sims = torch.full((len(indexed_by_img_id),), float("Inf"))
    all_rec_sims = torch.full((len(indexed_by_img_id),), float("Inf"))
    all_pre_sims = torch.full((len(indexed_by_img_id),), float("Inf"))
    all_f1_sims = torch.full((len(indexed_by_img_id),), float("Inf"))

    undef_rec, undef_prec = 0, 0
    for i, (img_id, preds) in enumerate(indexed_by_img_id):
        if preds["gt_norm_cxcywh_boxes"].shape[0] < 2 and preds["norm_cxcywh_boxes"].shape[0] < 2:
            # no gt objects and preds, both undefined, skip
            undef_rec, undef_prec = undef_rec + 1, undef_prec + 1
            continue
        if preds["gt_norm_cxcywh_boxes"].shape[0] < 2:
            # no gt objects, recall undefined (skip, inf), precision 0
            all_pre_sims[i] = 0
            undef_rec += 1
            continue
        if preds["norm_cxcywh_boxes"].shape[0] < 2:
            # no preds, precision undefined skip, recall 0
            all_rec_sims[i] = 0
            undef_prec += 1
            continue

        ps_to_gts = compute_greedy_matching({img_id: preds}, gt_anns=gt_anns)
        matched = ps_to_gts > -1

        # if matched.sum() < 2:
        #     continue

        matched_p = preds["norm_cxcywh_boxes"][matched]
        matched_gt = preds["gt_norm_cxcywh_boxes"][ps_to_gts[matched]]
        if matched_gt.dim() < 2 or matched_p.dim() < 2:
            logger.error(
                "Less then one preds or GTs: %s, %s, GT: %s, preds: %s"
                % (preds["caption_id"], preds["caption"], preds["names"], preds["gt_names"])
            )
            all_pre_sims[i] = 0
            all_rec_sims[i] = 0
            continue

        num_mtchd_obj = matched_p.shape[0]
        num_p_obj = preds["norm_cxcywh_boxes"].shape[0]
        num_gt_obj = preds["gt_norm_cxcywh_boxes"].shape[0]
        num_obj = num_p_obj + num_gt_obj - num_mtchd_obj

        # if num_mtchd_obj < 2:
        #     continue
        # gts_to_ps = -1 * np.ones(len(preds['gt_labels']), dtype=np.int)
        # for p, gt in enumerate(ps_to_gts):
        #     gts_to_ps[gt] = p

        sims = torch.zeros((num_obj, num_obj))
        pre_sims = torch.zeros(
            (preds["norm_cxcywh_boxes"].shape[0], preds["norm_cxcywh_boxes"].shape[0])
        )
        rec_sims = torch.zeros(
            (preds["gt_norm_cxcywh_boxes"].shape[0], preds["gt_norm_cxcywh_boxes"].shape[0])
        )

        delta_dist = _get_pw_diff_matrix(matched_p, matched_gt, distance)

        matched_pw_sims = 1 - delta_dist
        sims[: delta_dist.shape[0], : delta_dist.shape[1]] = matched_pw_sims
        rec_sims[: delta_dist.shape[0], : delta_dist.shape[1]] = matched_pw_sims
        pre_sims[: delta_dist.shape[0], : delta_dist.shape[1]] = matched_pw_sims

        s = sims[(*torch.triu_indices(num_obj, num_obj, offset=1),)].mean()
        all_sims[i] = s
        r = rec_sims[(*torch.triu_indices(num_gt_obj, num_gt_obj, offset=1),)].mean()
        all_rec_sims[i] = r
        p = pre_sims[(*torch.triu_indices(num_p_obj, num_p_obj, offset=1),)].mean()
        all_pre_sims[i] = p
        all_f1_sims[i] = fbeta(r, p)

    results = {} if not return_all else {"all": (all_sims, all_rec_sims, all_pre_sims, all_f1_sims)}
    rec = all_rec_sims[all_rec_sims.ne(float("Inf"))].nanmean()
    pre = all_pre_sims[all_pre_sims.ne(float("Inf"))].nanmean()
    return {
        **results,
        "pw_sim_prec_rec": all_sims[all_sims.ne(float("Inf"))].nanmean(),
        "pw_sim_rec": rec,
        "pw_sim_prec": pre,
        "pw_sim_f1": all_f1_sims[all_pre_sims.ne(float("Inf"))].nanmean(),
        "pw_sim_f1_after": fbeta(rec, pre),
        "undefined_precision": undef_prec,
        "undefined_recall": undef_rec,
    }


def compute_greedy_matching(
    indexed_by_img_id: Dict,
    gt_anns: Optional[List] = None,
    dist_fn: Callable = get_center_distances,
) -> np.array:

    # meant for 1 sample
    # gt_by_img_id = group_by_key(gt_anns_list, 'image_id')
    # gt = group_by_key(gt_by_img_id[img_id], 'category_id')

    assert len(indexed_by_img_id) == 1
    from coco_eval import CocoEvaluator

    prepare_for_coco_det = partial(CocoEvaluator.prepare_for_coco_detection, None)
    preds_per_instance = prepare_for_coco_det(indexed_by_img_id)

    gt_anns_list = prepare_gt_annotations(gt_anns, indexed_by_img_id, prepare_for_coco_det)
    gt_by_cat = group_by_key_and_count(gt_anns_list, "category_id")

    preds_by_cat = group_by_key_and_count(preds_per_instance, "category_id")
    all_cats = list(set(preds_by_cat.keys()).union(set(gt_by_cat.keys())))

    all_ps_to_gts = np.full(len(preds_per_instance), -1)

    for cat_id in all_cats:
        gt_hit = cat_id in gt_by_cat
        p_hit = cat_id in preds_by_cat
        gt_by_cat_id = gt_by_cat[cat_id] if gt_hit else []
        preds_by_cat_id = preds_by_cat[cat_id] if p_hit else []

        if gt_hit and p_hit:
            pred_idxs = np.array([p["count"] for p in preds_by_cat_id])
            gts_idxs = np.array([p["count"] for p in gt_by_cat_id])

            p_to_gts = _compute_greedy_matching_one_category(gt_by_cat_id, preds_by_cat_id, dist_fn)
            all_ps_to_gts[pred_idxs[p_to_gts > -1]] = gts_idxs[p_to_gts[p_to_gts > -1]]

    return all_ps_to_gts


def _compute_greedy_matching_one_category(gt, predictions, dist_fn):
    image_gts = group_by_key(gt, "image_id")
    # print(image_gts)
    image_gt_boxes = {
        img_id: np.array([[float(z) for z in b["bbox"]] for b in boxes])
        for img_id, boxes in image_gts.items()
    }
    image_gt_checked = {img_id: np.zeros(len(boxes)) for img_id, boxes in image_gts.items()}
    sorted_predictions = sorted(
        zip(range(len(predictions)), predictions), key=lambda x: x[1]["score"], reverse=True
    )
    # predictions = sorted(predictions, key=lambda x: x["score"], reverse=True)

    num_predictions = len(predictions)
    predictions_to_gts = np.full(num_predictions, -1)
    for prediction_index, prediction in sorted_predictions:
        box = np.array(prediction["bbox"])

        min_distance = np.inf
        jmax = -1

        try:
            gt_boxes = image_gt_boxes[prediction["image_id"]]  # gt_boxes per image
            gt_checked = image_gt_checked[prediction["image_id"]]  # gt flags per image
        except KeyError:
            gt_boxes = []
            gt_checked = None

        if len(gt_boxes) > 0:
            distances = dist_fn(box[None, :], gt_boxes)[0]

            for j, distance in enumerate(distances):
                if gt_checked[j] == 0 and distance < min_distance:
                    min_distance = distance
                    jmax = j

        if jmax > -1:
            predictions_to_gts[prediction_index] = jmax
            gt_checked[jmax] = 1
    return predictions_to_gts


def fbeta(recall, precision, beta=1):
    return (1 + beta**2) * recall * precision / (recall + (beta**2) * precision + 1e-8)


def combined_fbeta_multipl(r0, p0, r5, p5, norm=False, beta=1, beta_in=1):
    return fbeta(
        fbeta(r0, r5, beta_in) if norm else r0 * r5,
        fbeta(p0, p5, beta_in) if norm else p0 * p5,
        beta=beta,
    )


def combined_fbeta_subtr(r0, p0, r5, p5, norm=False, beta=1):
    div, term = (1, 0) if not norm else (2, 0.5)
    return fbeta((r5 - (1 - r0)) / div + term, (p5 - (1 - p0)) / div + term, beta=beta)
