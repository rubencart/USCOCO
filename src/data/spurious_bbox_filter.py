import logging
import math
import pickle
from itertools import repeat

import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

from data.singleton import S

logger = logging.getLogger("pytorch_lightning")


class SpuriousFilter:
    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class SingleSpuriousFilter(SpuriousFilter):
    def __init__(
        self,
        discrimination_type="rel",
        min_relative_size=0.5,
        max_deviation=2.0,
        img_dir="",
        annFile="",
        normalize_dist=True,
        distr_type="avgmax",
        dist_type="reltomax",
        pretrained_spurious_filter_file=None,
        save_file=None,
        filter_crowds=False,
        multiproc=True,
        instance_ds=None,
    ):
        self.discrimination_type = discrimination_type
        self.min_relative_size = min_relative_size
        self.max_deviation = max_deviation
        self.filter_crowds = filter_crowds
        self.multiproc = multiproc
        self.instance_ds = instance_ds
        if instance_ds is None:
            self.instance_ds = S.TrainDetection(img_dir, annFile, transform=transforms.ToTensor())

        if pretrained_spurious_filter_file is not None:
            with open(pretrained_spurious_filter_file, "rb") as file:
                sf = pickle.load(file)
            self.img_dir = sf.img_dir
            self.annFile = sf.annFile
            self.normalize_dist = sf.normalize_dist
            self.distr_type = sf.distr_type
            self.dist_type = sf.dist_type
            self.distributions = sf.distributions
        else:
            self.img_dir = img_dir
            self.annFile = annFile
            self.normalize_dist = normalize_dist
            self.distr_type = distr_type
            self.dist_type = dist_type
            self.distributions = self._build_distributions()
            if save_file is not None:
                self.save_spurious_filter(save_file)

    def _build_distributions(self):
        logger.info("calculating spuriousfilter distributions...")
        distributions = {}

        if self.multiproc:
            total, chunksize = len(self.instance_ds), 40000
            processes = 3
            with torch.multiprocessing.Pool(processes) as pool:
                args = (
                    repeat(self.instance_ds),
                    repeat(self.filter_crowds),
                    repeat(self.distr_type),
                    repeat(self.normalize_dist),
                    repeat(self.dist_type),
                )
                results = pool.starmap(
                    self._process_instances_for_img,
                    tqdm(zip(self.instance_ds.ids, *args), total=total),
                    chunksize=chunksize,
                )
        else:
            results = [
                self._process_instances_for_img(
                    img_id,
                    self.instance_ds,
                    self.filter_crowds,
                    self.distr_type,
                    self.normalize_dist,
                    self.dist_type,
                )
                for img_id in self.instance_ds.ids
            ]

        for img_distribs in results:
            for cid, areas in img_distribs.items():
                distributions.setdefault(cid, []).extend(areas)

        for category_id in distributions:
            areas = np.array(distributions[category_id])
            distributions[category_id] = (np.average(areas), np.sqrt(np.var(areas)))

        logger.info("calculating spuriousfilter distributions done.")
        return distributions

    @staticmethod
    def _process_instances_for_img(
        image_id, instance_ds, filter_crowds, distr_type, normalize_dist, dist_type
    ):
        distributions = {}
        instances = instance_ds._load_target(image_id)
        if filter_crowds:
            instances = [i for i in instances if i["iscrowd"] == 0]
        img_ann = instance_ds.coco.loadImgs(image_id)[0]
        img_height, img_width = img_ann["height"], img_ann["width"]
        areas = SingleSpuriousFilter._calc_inst_areas(
            instances, img_height, img_width, normalize_dist, dist_type
        )
        if distr_type == "avg":
            for area, inst in zip(areas, instances):
                distributions.setdefault(inst["category_id"], []).append(area)
        elif distr_type == "avgmax":
            img_distribution = {}
            for area, inst in zip(areas, instances):
                img_distribution.setdefault(inst["category_id"], []).append(area)
            for category_id, category_areas in img_distribution.items():
                distributions.setdefault(category_id, []).append(max(category_areas))
        else:
            raise Exception("unknown spurious filter type: '{}'".format(distr_type))
        return distributions

    @staticmethod
    def _calc_inst_areas(instances, img_height, img_width, normalize_dist, dist_type):
        if len(instances) == 0:
            return []

        diags = np.zeros(len(instances))

        if normalize_dist:
            n_height, n_width = img_height, img_width
        else:
            n_width, n_height = 1, 1

        for i, inst in enumerate(instances):
            diag = math.sqrt((inst["bbox"][2] / n_width) ** 2 + (inst["bbox"][3] / n_height) ** 2)
            diags[i] = diag

        if dist_type == "abs":
            diags = diags
        elif dist_type == "reltomax":
            max_diag = np.max(diags)
            diags = diags / max_diag
        else:
            raise Exception("unknown spurious_filter_dist_type: '{}'".format(dist_type))

        return diags

    def __call__(self, instances, img_height, img_width):
        areas = self._calc_inst_areas(
            instances, img_height, img_width, self.normalize_dist, self.dist_type
        )
        new_instances = []
        for area, inst in zip(areas, instances):
            category_avg, category_sd = self.distributions[inst["category_id"]]
            if self.discrimination_type == "rel":
                if area > self.min_relative_size * category_avg:
                    new_instances.append(inst)
            elif self.discrimination_type == "sd":
                if area > category_avg - (category_sd * self.max_deviation):
                    new_instances.append(inst)
            else:
                raise Exception(
                    "unknown spurious_filter_discrimination_type: '{}'".format(
                        self.discrimination_type
                    )
                )

        return new_instances

    def save_spurious_filter(self, save_file):
        with open(save_file, "wb") as file:
            pickle.dump(self, file)


class CompoundSpuriousFilter(SpuriousFilter):
    def __init__(self, spurious_filters, combination_logic):
        assert len(spurious_filters) > 0
        self.spurious_filters = spurious_filters
        self.combination_logic = combination_logic

    def __call__(self, instances, img_height, img_width):
        r_ids = {segm["id"] for segm in self.spurious_filters[0](instances, img_height, img_width)}
        for sf in self.spurious_filters[1:]:
            f_ids = {inst["id"] for inst in sf(instances, img_height, img_width)}
            if self.combination_logic == "or":
                r_ids.update(f_ids)
            elif self.combination_logic == "and":
                r_ids = set(filter(lambda r_id: r_id in f_ids, r_ids))
            else:
                raise Exception(
                    "unknown compound spurious filter combination_logic: '{}'".format(
                        self.combination_logic
                    )
                )

        r_instances = list(filter(lambda inst: inst["id"] in r_ids, instances))
        return r_instances
