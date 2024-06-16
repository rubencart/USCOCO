import copy
import errno
import json
import logging
import os
import socket
import sys
import textwrap
from datetime import datetime
from pathlib import Path
from shutil import rmtree
from textwrap import wrap
from typing import List, Optional

import dateutil.tz
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision
import wandb
from torch import Tensor
from tqdm import tqdm

from data.dictionary import PositionDictionary

logger = logging.getLogger("pytorch_lightning")


def get_home_dir():
    host = socket.gethostname()
    homedir = "home1" if host in ["frodo", "rose"] else "home2"
    return homedir


def generate_output_dir_name():
    now = datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime("%Y_%m_%d_%H_%M_%S_%f")

    output_dir = Path("./output") / timestamp
    output_dir.mkdir(parents=True, exist_ok=False)
    return str(output_dir)


def initialize_logging(output_dir, local_rank=0, to_file=True):
    logger = logging.getLogger("pytorch_lightning")
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(process)d - %(levelname)s - %(message)s", datefmt="%d-%m-%Y %H:%M:%S"
    )
    if len(logger.handlers) > 0:
        logger.handlers[0].setFormatter(formatter)
    else:
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    logger.setLevel(logging.INFO)

    if to_file:
        path = os.path.join(output_dir, "console-output_rank-{}.log".format(local_rank))
        fh = logging.FileHandler(path)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        logger.info("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        logger.info("Initialized logging to %s" % path)
        logger.info("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    # else:
    #     logger.setLevel(logging.ERROR)
    return logger


def get_df_from_wandb(
    project_name: str,
    num: int = 2000,
    convert_dists_type: bool = True,
    later_than: Optional[str] = None,
) -> pd.DataFrame:
    api = wandb.Api(timeout=30)
    # Project is specified by <entity/project-name>
    runs = api.runs(
        f"liir-kuleuven/{project_name}",
        per_page=num,
        filters={"createdAt": {"$gt": later_than}} if later_than else None,
    )
    summary_list, config_list, name_list = [], [], []
    last_epoch, last_step = [], []
    for run in tqdm(runs):
        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files
        summary_list.append(run.summary._json_dict)
        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        config_list.append({k: v for k, v in run.config.items() if not k.startswith("_")})
        # .name is the human-readable name of the run.
        name_list.append(run.name)
        last_epoch.append(run.history()["epoch"].max() if len(run.history()) > 0 else np.NaN)
        last_step.append(
            run.history()["trainer/global_step"].max() if len(run.history()) > 0 else np.NaN
        )
    runs_df = pd.DataFrame({
        "summary": summary_list,
        "config": config_list,
        "name": name_list,
        "last_step": last_step,
        "last_epoch": last_epoch,
    })
    df = pd.concat(
        [
            runs_df.drop(["config", "summary"], axis=1),
            runs_df["config"].apply(pd.Series),
            runs_df["summary"].apply(pd.Series),
        ],
        axis=1,
    )
    if convert_dists_type:
        dists = [
            "test_comp_delta_pairwise_distances_scaled_diags",
            "test_indom_delta_pairwise_distances_scaled_diags",
            "test_abs_delta_pairwise_distances_scaled_diags",
        ]
        df[dists] = df[dists].astype("float64")
    return df


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def rmdir_p(path):
    if os.path.isdir(path):
        rmtree(path)
    elif not os.path.exists(path):
        pass
    elif os.path.isfile(path):
        raise FileExistsError("Path is file: %s" % path)
    else:
        raise Exception


def print_predicted_bboxes(
    output,
    batch,
    category_dict,
    pos_dict,
    filename,
    dirname,
    width=512,
    height=512,
    max_objects=-1,
    save=False,
    show=True,
):
    max_vals, max_preds = F.softmax(output["label_logits"], dim=-1).max(-1)
    sorted_max_vals, arg_sorted_vals = max_vals.sort(dim=-1, descending=True)

    no_obj_label = len(category_dict) - 1
    mkdir_p(dirname)
    from matplotlib import pyplot as plt

    plt.close("all")

    for i, caption in enumerate(batch["captions"]):
        height, width = 500, 500
        img_id = str(batch["img_ids"][i])
        if "caption_ids" in batch:
            img_id += "-" + str(batch["caption_ids"][i])
        img = torch.zeros((3, height, width), dtype=torch.uint8)

        obj_indices = max_preds[i][arg_sorted_vals[i]].ne(no_obj_label)
        obj_to_plot = max_preds[i][arg_sorted_vals[i]][obj_indices]
        obj_to_plot = obj_to_plot[:max_objects] if max_objects > 0 else obj_to_plot

        txt_labels = [
            f"{category_dict.symbols[lab]} {sorted_max_vals[i][idx]:.2f}"
            for idx, lab in enumerate(obj_to_plot)
        ]
        # print(txt_labels)
        img_path = os.path.join(dirname, img_id)

        bboxes = output["bbox_preds"][i][arg_sorted_vals[i]][obj_indices].clone()
        bboxes = bboxes[:max_objects] if max_objects > 0 else bboxes

        if bboxes.shape[-1] != 4:
            raise ValueError
            bboxes = pos_dict.decode_tensor(bboxes.argmax(-1))

        bboxes_int = copy.deepcopy(bboxes)
        bboxes_int[:, 0] *= width  # proportional center x
        bboxes_int[:, 1] *= height  # prop center y
        bboxes_int[:, 2] *= width  # prop width
        bboxes_int[:, 3] *= height  # prop height

        bboxes_int[:, 0] = bboxes_int[:, 0] - (bboxes_int[:, 2] / 2)  # center to left upper corner
        bboxes_int[:, 1] = bboxes_int[:, 1] - (bboxes_int[:, 3] / 2)
        bboxes_int[:, 2] = bboxes_int[:, 0] + bboxes_int[:, 2]
        bboxes_int[:, 3] = bboxes_int[:, 1] + bboxes_int[:, 3]

        colors = 10 * ["red", "green", "yellow", "purple", "pink", "orange", "blue"]
        img_with_boxes = torchvision.utils.draw_bounding_boxes(
            img.byte(),
            bboxes_int,
            txt_labels,
            colors[: len(bboxes_int)],
            font="/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
            font_size=18,
            width=2,
        )

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(img_with_boxes.permute(1, 2, 0))

        title = ax.set_title("\n".join(wrap(caption, 60)))

        fig.tight_layout()
        title.set_y(1.05)
        fig.subplots_adjust(top=0.9)

        if show:
            fig.show()
        if save:
            fig.savefig(img_path)
        plt.close("all")


def print_bboxes_for_sample(
    caption,
    img_id,
    dirname,
    labels,
    boxes=None,
    prop_boxes=None,
    img=None,
    caption_id=None,
    show=True,
    save=False,
    centered=True,
    outline=True,
):
    from matplotlib import pyplot as plt

    plt.close("all")

    filename = str(img_id)
    if caption_id:
        filename += "-" + str(caption_id)
    filename += ".png"

    if img is None:
        height, width = 500, 500
        img = 255 * torch.ones((3, height, width), dtype=torch.uint8)
    else:
        height, width = img.shape[1:]

    if boxes is None:
        boxes_to_use = torch.Tensor(copy.deepcopy(prop_boxes))
        if len(boxes_to_use) > 0:
            boxes_to_use[:, 0] *= width  # proportional center x
            boxes_to_use[:, 1] *= height  # prop center y
            boxes_to_use[:, 2] *= width  # prop width
            boxes_to_use[:, 3] *= height  # prop height
    else:
        boxes_to_use = torch.Tensor(copy.deepcopy(boxes))

    mkdir_p(dirname)
    img_path = os.path.join(dirname, filename)

    bboxes_int = torch.Tensor(boxes_to_use)

    if len(boxes_to_use) > 0:
        if centered:
            bboxes_int[:, 0] = bboxes_int[:, 0] - (
                bboxes_int[:, 2] / 2
            )  # center to left upper corner
            bboxes_int[:, 1] = bboxes_int[:, 1] - (bboxes_int[:, 3] / 2)
        bboxes_int[:, 2] = bboxes_int[:, 0] + bboxes_int[:, 2]  # w/h to right bottom corner
        bboxes_int[:, 3] = bboxes_int[:, 1] + bboxes_int[:, 3]

    # colors = 10 * ['firebrick', 'mediumblue', 'forestgreen', 'purple', 'black', 'orange']
    colors = 10 * ["darkred", "darkblue", "darkgreen", "Black", "olive", "purple"]
    # colors = 20 * ['black',]
    img_with_boxes = torchvision.utils.draw_bounding_boxes(
        img,
        bboxes_int,
        labels,
        colors[: len(bboxes_int)],
        # font='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        font="/cw/liir_code/NoCsBack/rubenc/Arial.ttf",
        font_size=24,
        width=3,
    )

    fig = plt.figure(
        # constrained_layout=True, # figsize=(width / dpi, (height + 1) / dpi), dpi=dpi
        figsize=(5, 5)
    )

    ax = fig.add_subplot(111)

    ax.set_aspect(1)
    ax.imshow(img_with_boxes.permute(1, 2, 0))
    ax.set_xticks([])
    ax.set_yticks([])

    if caption:
        caption = (caption[0].upper() + caption[1:]).strip()
        if caption[-1] != ".":
            caption += "."

        ax.set_title(textwrap.dedent("\n".join(textwrap.wrap(caption, 28))), fontsize=18, y=-0.16)

    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(1.5 if outline else 0.0)

    ax.set_aspect(1)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)

    if save:
        plt.savefig(img_path)
    if show:
        plt.show()


def filter_boxes_discr(boxes: Tensor, pos_dict: "PositionDictionary", map_to_list=False) -> List:
    def fn(x):
        return x.tolist() if map_to_list else x

    return [fn(sample[sample.ne(pos_dict.pad())].view(-1, 4)) for sample in boxes]


def filter_boxes_cont(boxes: Tensor, pad_id: int, map_to_list=False) -> List:
    def fn(x):
        return x.tolist() if map_to_list else x

    return [fn(sample[sample.ne(pad_id)].view(-1, 4)) for sample in boxes]


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(-1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=-1)


def convert_to_quantized_annotations(
    ann_file: str, new_ann_file: str, pos_dict: PositionDictionary
):
    raise NotImplementedError

    if os.path.isfile(new_ann_file):
        logger.info("Reusing quantized validation bboxes for COCO evaluator: %s" % new_ann_file)
        return

    logger.info("Quantizing validation bboxes for COCO evaluator: %s" % new_ann_file)
    with open(ann_file, "r") as f:
        annotation_dict = json.load(f)

    imgs = {}
    for img in annotation_dict["images"]:
        imgs[img["id"]] = img

    result = copy.deepcopy(annotation_dict)
    for ann in tqdm(result["annotations"]):
        img = imgs[ann["image_id"]]
        ih, iw = img["height"], img["width"]
        # original image sizes, xywh
        bx, by, bw, bh = ann["bbox"]
        # proportional
        px, py, pw, ph = bx / iw, by / ih, bw / iw, bh / ih
        # centered
        cx, cy = px + pw / 2, py + ph / 2
        # encode - decode
        decoded = pos_dict.decode_tensor(torch.as_tensor(pos_dict.encode([cx, cy, pw, ph])))
        # to proportional xywh
        ox, oy, ow, oh = box_xyxy_to_xywh(box_cxcywh_to_xyxy(decoded)).tolist()
        # to original image sizes
        ann["bbox"] = [ox * iw, oy * ih, ow * iw, oh * ih]

    with open(new_ann_file, "w") as f:
        json.dump(result, f)
