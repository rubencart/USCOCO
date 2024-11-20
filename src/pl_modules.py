import contextlib
import json
import logging
import os
import pickle
import pprint
from abc import ABC
from collections import OrderedDict
from typing import Any, Dict, List

import numpy as np
import pytorch_lightning as pl
import torch
import torch.distributed
from torch import Tensor, optim
from torch.optim.lr_scheduler import LinearLR, SequentialLR

import metrics
from config import Config, TextEncoderConfig
from data.dictionary import CategoryDictionary, PositionDictionary
from data.tokenization import Tokenizer
from model.loss.autoregressive import AutoregressiveCriterion
from model.loss.detr import Txt2ImgSetCriterion
from model.loss.obj_gan import ObjGANCriterion
from model.modules import (
    AutoregressiveGenerationModel,
    DETRGenerationModel,
    ObjGANGenerationModel,
)
from model.postprocess import PostProcess

logger = logging.getLogger("pytorch_lightning")


def unpack_tensor(inp):
    # print(type(inp))
    if isinstance(inp, Tensor):
        try:
            return inp.tolist()  # if inp.numel() != 1 else inp.item()
        except ValueError as e:
            # logger.error(str(e))
            logger.error(inp)
            raise
    else:
        return inp


class GenerationModule(pl.LightningModule, ABC):
    def __init__(
        self,
        cfg: Config,
        category_dict: CategoryDictionary,
        pos_dict: PositionDictionary,
        tokenizer: Tokenizer,
        automatic_optimization: bool = True,
    ):
        super().__init__()
        if not isinstance(cfg.text_encoder, TextEncoderConfig):
            print(type(cfg.text_encoder))
            print(cfg.text_encoder)
            cfg.process_args(process=False)
            # cfg.process_args()
            cfg._parsed = True

        self.cfg = cfg
        self.category_dict = category_dict
        self.pos_dict = pos_dict
        self.save_hyperparameters(
            cfg.to_dict(),
            ignore=["category_dict", "pos_dict", "tokenizer", "automatic_optimization"],
        )
        self.automatic_optimization = automatic_optimization
        self.tokenizer = tokenizer

        self.post_process = PostProcess(cfg, category_dict, pos_dict)
        self.predictions = {}
        self.statistics = {}

        self.save_predictions_to_file = False
        self.save_predictions_location = os.path.join(self.cfg.run_output_dir, "preds.json")
        self.save_val_predictions_location = os.path.join(self.cfg.run_output_dir, "val_preds.json")

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]):
        if self.cfg.text_encoder.use_llama and not self.cfg.text_encoder.txt_enc_finetune:
            # exclude huge llama weights from checkpoint to save disk space
            state_dict = OrderedDict()
            for name, param in checkpoint["state_dict"].items():
                if "model.text_encoder.text_encoder.model" not in name:
                    state_dict[name] = param
            checkpoint["state_dict"] = state_dict

    def configure_optimizers(self):
        params = list(self.named_parameters())

        def is_text_enc(n):
            return "text_encoder" in n

        params_and_configs = [([p for n, p in params if not is_text_enc(n)], self.cfg.detr)]
        if self.cfg.model.train_text_encoder:
            params_and_configs += [
                ([p for n, p in params if is_text_enc(n)], self.cfg.text_encoder)
            ]

        optimizer_class = optim.AdamW if self.cfg.optimizer == "adamw" else optim.SGD
        optimizer = optimizer_class(
            [
                {
                    "params": params,
                    "lr": (
                        cfg.lr * self.trainer.num_processes
                        if self.cfg.scale_lr_by_ddp_nodes
                        else cfg.lr
                    ),
                    **(
                        {
                            "betas": (cfg.adam_beta_1, cfg.adam_beta_2),
                            "eps": cfg.adam_eps,
                            "weight_decay": cfg.weight_decay,
                        }
                        if self.cfg.optimizer == "adamw"
                        else {
                            "weight_decay": 0.0,
                            "momentum": 0.9,
                            "nesterov": True,
                        }
                    ),
                }
                for (params, cfg) in params_and_configs
            ],
        )
        logger.info(optimizer)

        schedulers = []
        if self.cfg.lr_schedule is not None:
            logger.info("Using LR scheduler: %s" % self.cfg.lr_schedule)
            if self.cfg.lr_schedule == "linear_with_warmup":
                scheduler1 = LinearLR(
                    optimizer,
                    start_factor=1 / self.cfg.lr_schedule_warmup_epochs,
                    end_factor=1.0,
                    total_iters=self.cfg.lr_schedule_warmup_epochs - 1,
                )
                scheduler2 = LinearLR(
                    optimizer,
                    start_factor=1.0,
                    end_factor=1e-8,
                    total_iters=self.cfg.train.max_epochs - self.cfg.lr_schedule_warmup_epochs,
                )
                scheduler = SequentialLR(
                    optimizer,
                    [scheduler1, scheduler2],
                    milestones=[self.cfg.lr_schedule_warmup_epochs - 1],
                )
                schedulers.append(scheduler)

        return [optimizer], schedulers

    def forward(self, batch):
        output = self.model.generate(batch)
        # output = self.model(batch)
        return {
            **output,
            **batch,
        }

    def on_train_epoch_start(self) -> None:
        if self.cfg.debug and self.cfg.lr_schedule is not None:
            logger.info(f"Current LR: {self.lr_schedulers().get_last_lr()}")

    def training_step(self, batch, batch_idx):
        output = self.model(batch)
        loss_output = self.train_criterion(output, batch)

        # https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#training-step
        assert "loss" in loss_output
        loss_output = {
            k: v.detach().clone() if (isinstance(v, Tensor) and not k == "loss") else v
            for k, v in loss_output.items()
        }
        bs = len(batch["captions"])

        # exclude aux loss logs
        excludes = ["layer_num", "avg_num_objects_", "avg_predicted_length_", "cardinality_error_"]
        self.log_dict(
            {
                k: v.detach().clone() if isinstance(v, Tensor) else v
                for k, v in loss_output.items()
                if not any([k.startswith(excl) for excl in excludes])
            },
            on_step=True,
            on_epoch=True,
            logger=True,
            prog_bar=False,
            batch_size=bs,
        )
        return loss_output

    def on_test_epoch_start(self) -> None:
        return self.on_validation_epoch_start()

    def test_step(self, batch, batch_idx, dataloader_idx):
        return {"dataloader_idx": dataloader_idx, "preds": self.validation_step(batch, batch_idx)}

    def test_epoch_end(self, outputs):

        if len(outputs) == 1:
            dsets = (("val", "test_indom"),)
        else:
            dsets = (
                ("absurd", "test_abs"),
                ("val", "test_indom"),
                ("newval", "new_val"),
            )[: len(outputs) + 1]
        if self.cfg.save_probe_embeddings_train:
            dsets = dsets + (("train", "train"),)

        self.predictions = {}
        for i, (split, prefix) in enumerate(dsets):
            # rdict = self.compute_metrics(outputs[i])
            # results = {f'test_{prefix}_{k}': v for k, v in rdict['result_dict'].items()}
            # self.statistics.update(results)
            # self.log_dict(results)
            # all_results[prefix] = rdict
            preds = [d["preds"] for d in outputs[i]]
            metrics, updated_preds = self._evaluate_outputs(preds, stage="test", split=split)
            log_dict = {f"{prefix}_{k}": v for k, v in metrics.items()}
            self.log_dict(log_dict, sync_dist=True, prog_bar=False)
            self.statistics.update(log_dict)
            self.predictions[prefix] = updated_preds

        if self.save_predictions_to_file:
            p = {
                split: {
                    strat: {
                        s_id: {
                            metr: unpack_tensor(val)
                            for (metr, val) in self.predictions[split][strat][s_id].items()
                        }
                        for s_id in self.predictions[split][strat]
                    }
                    for strat in self.predictions[split]
                }
                for split in self.predictions
            }
            pred_path = self.save_predictions_location
            logger.info("Saving predictions to %s" % pred_path)
            with open(pred_path, "w") as f:
                json.dump(p, f)

    def validation_step(self, batch, batch_idx):
        result = {}
        for strategy in self.coco_evaluators.keys():
            result[strategy] = self.validation_step_with_strategy(batch, strategy)
        return result

    def validation_step_with_strategy(self, batch, strategy):
        output = self.model.generate(batch, strategy=strategy)
        processed_output = self.post_process(
            output, batch, ground_truth=False, ones_as_scores=False
        )
        indexed_by_img_id = OrderedDict([
            (img_id, output) for img_id, output in zip(batch["img_ids"], processed_output)
        ])
        # indexed_by_img_id = self.add_inspection_info(processed_output, batch)
        # self.coco_evaluators[strategy].update(indexed_by_img_id)

        result = {"predictions": indexed_by_img_id}

        if strategy == self.strategy_for_eval_criterion:
            loss_output = self.eval_criterion(output, batch)
            result.update({
                k: v.detach().clone() if isinstance(v, Tensor) else v
                for k, v in loss_output.items()
            })
        return result

    def validation_epoch_end(self, list_of_step_outputs: List[Dict]):
        log_dict, updated_preds = self._evaluate_outputs(list_of_step_outputs)

        if self.save_predictions_to_file:
            pred_path = self.save_val_predictions_location
            logger.info("Saving predictions to %s" % pred_path)
            with open(pred_path, "wb") as f:
                pickle.dump(updated_preds, f)

        self.log_dict(log_dict, sync_dist=True, prog_bar=False)
        assert self.cfg.model_checkpoint_monitor in log_dict

        if self.cfg.debug or self.cfg.wandb_offline:
            logger.info("log_dict: %s" % pprint.pformat(log_dict, indent=2))

        self.statistics = log_dict
        self.predictions = {"new_val": updated_preds}

    def _evaluate_outputs(self, list_of_step_outputs, stage="val", split="val"):
        preds_by_strategy_by_img_id = {strat: {} for strat in self.coco_evaluators.keys()}

        for step_output in list_of_step_outputs:
            for strategy, loss_and_preds in step_output.items():
                for s_i, (img_id, v) in enumerate(loss_and_preds["predictions"].items()):
                    if "sim_scores" in loss_and_preds:
                        v["sim_scores"] = loss_and_preds["sim_scores"][s_i]
                    preds_by_strategy_by_img_id[strategy][img_id] = v

        # exclude aux loss logs
        excludes = [
            "layer_num",
            "avg_num_objects_",
            "avg_predicted_length_",
            "cardinality_error_",
            "predictions",
            "sim_scores",
        ]

        log_dict = {
            f"{stage}_{k}": (
                torch.stack([d[strategy][k] for d in list_of_step_outputs]).mean().detach().clone()
            )
            for strategy in self.coco_evaluators.keys()
            for k in list_of_step_outputs[0][strategy]
            if not any([excl in k for excl in excludes])
        }

        updated_preds = {strat: {} for strat in self.coco_evaluators.keys()}
        for strategy, evaluator in self.coco_evaluators.items():
            stats_dict, predictions = self._compute_metrics(
                preds_by_strategy_by_img_id[strategy],
                evaluator,
                # official_coco_metrics=stage != 'test'
                official_coco_metrics=False,
                split=split,
            )
            updated_preds[strategy] = predictions

            log_dict.update({f"{strategy}_{k}": v for k, v in stats_dict.items()})
            if strategy == self.strategy_for_eval_criterion:
                # log_dict.update(prec_rec_dict)
                log_dict.update(stats_dict)

        # self.predictions = updated_preds
        return log_dict, updated_preds

    def _compute_metrics(self, preds_by_img_id, evaluator, official_coco_metrics=True, split="val"):
        stats_dict = {}

        (
            sam_prec,
            sam_rec,
            cls_prec,
            cls_rec,
            mic_prec,
            mic_rec,
            all_ps,
            class_avg_precisions,
            all_gts,
            class_avg_recalls,
        ) = metrics.compute_precision_recall(preds_by_img_id, self.category_dict, log=False)

        (
            sam_prec_05,
            sam_rec_05,
            cls_prec_05,
            cls_rec_05,
            mic_prec_05,
            mic_rec_05,
            all_ps_iou,
            class_avg_precisions_iou,
            all_gts_iou,
            class_avg_recalls_iou,
            flipped_preds_by_img_id,
            nb_flipped,
        ) = metrics.compute_precision_recall_iou(
            preds_by_img_id, self.category_dict, iou=0.5, log=False, flip_boxes=self.cfg.flip_boxes
        )

        stats_dict.update({
            **self._metrics_from_precision_recall(
                sam_rec,
                sam_prec,
                cls_rec,
                cls_prec,
                mic_prec,
                mic_rec,
                sam_rec_05,
                sam_prec_05,
                cls_rec_05,
                cls_prec_05,
                mic_prec_05,
                mic_rec_05,
            ),
            "nb_preds_horizontally_flipped": nb_flipped,
        })
        if split == "absurd":
            stats_dict.update(self._metrics_replaced_only(preds_by_img_id))

        def compute_coco_eval_metrics(evaluator):
            evaluator.synchronize_between_processes()  # this actually does not synchronize anything
            evaluator.accumulate()
            evaluator.summarize()

        if official_coco_metrics:
            evaluator.update(flipped_preds_by_img_id)

            # compute_coco_eval_metrics(evaluator)
            with open(os.devnull, "w") as devnull:
                with contextlib.redirect_stdout(devnull):
                    compute_coco_eval_metrics(evaluator)

            coco_stats = evaluator.coco_eval["bbox"].stats
            stats_dict.update({
                "Avg_Precision_IoU=0.50:0.95_area=all_maxDets=100": coco_stats[0],
                "coco_ap_iou_05-095": coco_stats[0],
                "Avg_Precision_IoU=0.50_area=all_maxDets=100": coco_stats[1],
                "coco_ap_iou_05": coco_stats[1],
                "Avg_Precision_IoU=0.75_area=all_maxDets=100": coco_stats[2],
                "Avg_Precision_IoU=0.50:0.95_area=small_maxDets=100": coco_stats[3],
                "Avg_Precision_IoU=0.50:0.95_area=medium_maxDets=100": coco_stats[4],
                "Avg_Precision_IoU=0.50:0.95_area=large_maxDets=100": coco_stats[5],
                "Avg_Recall_IoU=0.50:0.95_area=all_maxDets=1": coco_stats[6],
                "Avg_Recall_IoU=0.50:0.95_area=all_maxDets=10": coco_stats[7],
                "Avg_Recall_IoU=0.50:0.95_area=all_maxDets=100": coco_stats[8],
                "coco_ar_iou_05-095": coco_stats[8],
                "Avg_Recall_IoU=0.50:0.95_area=small_maxDets=100": coco_stats[9],
                "Avg_Recall_IoU=0.50:0.95_area=medium_maxDets=100": coco_stats[10],
                "Avg_Recall_IoU=0.50:0.95_area=large_maxDets=100": coco_stats[11],
                "Avg_Precision_IoU=0.00_area=all_maxDets=100": coco_stats[12],
                "coco_ap_iou_00": coco_stats[12],
                "Avg_Recall_IoU=0.00_area=all_maxDets=100": coco_stats[13],
                "coco_ar_iou_00": coco_stats[13],
                # 'type_AP': type(stats[12]),
            })

        list_predictions = list(flipped_preds_by_img_id.items())
        pairwise_metrics = metrics.relative_pairwise_distances(
            list_predictions, return_all=True or self.cfg.debug
        )
        pw_sims = metrics.relative_pairwise_similarity_pr(
            list_predictions, return_all=True, distance="euclid"
        )
        pw_y_sims = metrics.relative_pairwise_similarity_pr(
            list_predictions, return_all=True, distance="L1"
        )

        mdds, msdds, msdns, mdxs, mdys = pairwise_metrics.pop("all")
        sims, rec_sims, pre_sims, f1_sims = pw_sims.pop("all")
        sims_y, rec_sims_y, pre_sims_y, f1_sims_y = pw_y_sims.pop("all")
        combined_metrics = [
            "f1_comb_mult_raw_iou_05",
            "f1_comb_mult_f1_iou_05",
            "f1_comb_subtr_norm_iou_05",
        ]
        per_sample_avg_metrics = ["f1_iou_00", "f1_iou_05", "min_iou_05", "min_iou_00"]
        combined = {k: [] for k in combined_metrics + per_sample_avg_metrics}
        for i, (img_id, v) in enumerate(list_predictions):
            v = self._update_predictions_with_per_sample_metrics(img_id, v, split)
            v["delta_pairwise_distances"] = float(mdds[i])
            v["delta_pairwise_distances_scaled_diags"] = float(msdds[i])
            v["delta_pairwise_norm_width_height_diffs"] = float(msdns[i])
            v["delta_pairwise_x_distances_norm_w"] = float(mdxs[i])
            v["delta_pairwise_y_distances_norm_h"] = float(mdys[i])
            v["pw_sim_prec_rec"] = float(sims[i])
            v["pw_sim_rec"] = float(rec_sims[i])
            v["pw_sim_prec"] = float(pre_sims[i])
            v["pw_sim_f1"] = float(f1_sims[i])
            v["y_pw_sim_prec_rec"] = float(sims_y[i])
            v["y_pw_sim_rec"] = float(rec_sims_y[i])
            v["y_pw_sim_prec"] = float(pre_sims_y[i])
            v["y_pw_sim_f1"] = float(f1_sims_y[i])
            for k in combined_metrics + per_sample_avg_metrics:
                combined[k].append(v[k])

        stats_dict.update(pairwise_metrics)
        stats_dict.update(pw_sims)
        stats_dict.update({f"y_{k}": v for k, v in pw_y_sims.items()})
        for m in combined_metrics:
            stats_dict[m] = np.mean(combined[m])
        for m in per_sample_avg_metrics:
            stats_dict[f"{m}_sam_lvl"] = np.mean(combined[m])

        stats_dict = {k: float(v) for (k, v) in stats_dict.items()}
        return stats_dict, flipped_preds_by_img_id

    @staticmethod
    def _metrics_from_precision_recall(
        sam_rec,
        sam_prec,
        cls_rec,
        cls_prec,
        mic_prec,
        mic_rec,
        sam_rec_05,
        sam_prec_05,
        cls_rec_05,
        cls_prec_05,
        mic_prec_05,
        mic_rec_05,
    ):
        return {
            "f1_iou_00": metrics.fbeta(sam_rec, sam_prec),
            "precision_iou_00": sam_prec,
            "recall_iou_00": sam_rec,
            "precision_class_iou_00": cls_prec,
            "recall_class_iou_00": cls_rec,
            "f1_class_iou_00": metrics.fbeta(cls_rec, cls_prec),
            "precision_micro_iou_00": mic_prec,
            "recall_micro_iou_00": mic_rec,
            "f1_micro_iou_00": metrics.fbeta(mic_rec, mic_prec),
            "precision_iou_05": sam_prec_05,
            "recall_iou_05": sam_rec_05,
            "f1_iou_05": metrics.fbeta(sam_rec_05, sam_prec_05),
            "precision_class_iou_05": cls_prec_05,
            "recall_class_iou_05": cls_rec_05,
            "f1_class_iou_05": metrics.fbeta(cls_rec_05, cls_prec_05),
            "precision_micro_iou_05": mic_prec_05,
            "recall_micro_iou_05": mic_rec_05,
            "f1_micro_iou_05": metrics.fbeta(mic_rec_05, mic_prec_05),
            # 'class_avg_precisions': torch.as_tensor(class_avg_precisions),
            # 'class_avg_recalls': torch.as_tensor(class_avg_recalls),
            "f2_iou_00": metrics.fbeta(sam_rec, sam_prec, beta=2),
            "f2_iou_05": metrics.fbeta(sam_rec_05, sam_prec_05, beta=2),
            "f1_f2_iou_05_f2_iou_00": metrics.fbeta(
                metrics.fbeta(sam_rec, sam_prec, beta=2),
                metrics.fbeta(sam_rec_05, sam_prec_05, beta=2),
                beta=1,
            ),
        }

    @staticmethod
    def _metrics_combined(sam_rec, sam_prec, sam_rec_05, sam_prec_05):
        # on sample level!!! So that first combined, then avg'd over samples
        return {
            "f1_comb_mult_raw_iou_05": metrics.combined_fbeta_multipl(
                sam_rec, sam_prec, sam_rec_05, sam_prec_05
            ),
            "f1_comb_mult_f1_iou_05": metrics.combined_fbeta_multipl(
                sam_rec, sam_prec, sam_rec_05, sam_prec_05, norm=True
            ),
            "f1_comb_subtr_norm_iou_05": metrics.combined_fbeta_subtr(
                sam_rec, sam_prec, sam_rec_05, sam_prec_05, norm=True
            ),
        }

    def _metrics_replaced_only(self, predictions):
        (p_repl_00, r_repl_00, p_class_repl_00, r_class_repl_00, _, _, _, _, _, _) = (
            metrics.compute_precision_recall(predictions, self.category_dict, replaced_only=True)
        )

        (p_repl_05, r_repl_05, p_class_repl_05, r_class_repl_05, _, _, _, _, _, _, _, _) = (
            metrics.compute_precision_recall_iou(
                predictions,
                self.category_dict,
                iou=0.5,
                flip_boxes=self.cfg.flip_boxes,
                replaced_only=True,
            )
        )
        return {
            "f1_repl_iou_00": metrics.fbeta(r_repl_00, p_repl_00),
            "precision_repl_iou_00": p_repl_00,
            "recall_repl_iou_00": r_repl_00,
            "precision_repl_iou_05": p_repl_05,
            "recall_repl_iou_05": r_repl_05,
            "f1_repl_iou_05": metrics.fbeta(r_repl_05, p_repl_05),
        }

    def _update_predictions_with_per_sample_metrics(self, img_id, predictions, split="val"):
        spr, src, cpr, crc, mpr, mrc, _, _, _, _ = metrics.compute_precision_recall(
            {img_id: predictions}, self.category_dict
        )
        spr5, src5, cpr5, crc5, mpr5, mrc5, _, _, _, _, _, flipped = (
            metrics.compute_precision_recall_iou(
                {img_id: predictions},
                self.category_dict,
                iou=0.5,
                flip_boxes=self.cfg.flip_boxes,
            )
        )
        predictions.update({
            **self._metrics_from_precision_recall(
                src, spr, crc, cpr, mpr, mrc, src5, spr5, crc5, cpr5, mpr5, mrc5
            ),
            **self._metrics_combined(src, spr, src5, spr5),
            "min_iou_05": min(spr5, src5),
            "min_iou_00": min(spr, src),
            "flipped": flipped,
        })
        if split == "absurd":
            predictions.update(self._metrics_replaced_only({img_id: predictions}))
        return predictions


class DETRGenerationModule(GenerationModule):
    def __init__(
        self,
        cfg: Config,
        category_dict: CategoryDictionary,
        pos_dict: PositionDictionary,
        tokenizer: Tokenizer,
        # coco_api: COCO = None,
        automatic_optimization: bool = True,
    ):
        super().__init__(
            cfg,
            category_dict,
            pos_dict,
            tokenizer,
            automatic_optimization,
        )

        logger.info("Initializing model and criterion")
        model_args = (self.cfg, self.tokenizer, self.category_dict, self.pos_dict)
        criterion_args = (self.cfg, self.category_dict, self.pos_dict)
        self.model = DETRGenerationModel(*model_args)
        self.train_criterion = Txt2ImgSetCriterion(*criterion_args)
        self.eval_criterion = self.train_criterion
        self.strategy_for_eval_criterion = "detr"

    def on_validation_epoch_start(self) -> None:
        self.coco_evaluators = {"detr": None}  # CocoEvaluator(self.coco_api, ('bbox',))


class AutoregressiveGenerationModule(GenerationModule):
    def __init__(
        self,
        cfg: Config,
        category_dict: CategoryDictionary,
        pos_dict: PositionDictionary,
        tokenizer: Tokenizer,
        # coco_api: COCO = None,
        automatic_optimization: bool = True,
    ):
        super().__init__(
            cfg,
            category_dict,
            pos_dict,
            tokenizer,
            automatic_optimization,
        )

        logger.info("Initializing model and criterion")
        model_args = (self.cfg, self.tokenizer, self.category_dict, self.pos_dict)
        criterion_args = (self.cfg, self.category_dict, self.pos_dict)
        self.model = AutoregressiveGenerationModel(*model_args)
        self.train_criterion = AutoregressiveCriterion(*criterion_args)
        self.eval_criterion = AutoregressiveCriterion(*criterion_args)
        # self.eval_criterion = Txt2ImgSetCriterion(*criterion_args)
        self.strategy_for_eval_criterion = (  # cfg.detr.generation_strategy_for_eval_criterion
            "greedy"
        )

    def on_validation_epoch_start(self) -> None:
        self.coco_evaluators = {
            strategy: None  # CocoEvaluator(self.coco_api, ('bbox',))
            for strategy in self.cfg.detr.generation_strategies
        }


class ObjGANGenerationModule(GenerationModule):
    def __init__(
        self,
        cfg: Config,
        category_dict: CategoryDictionary,
        pos_dict: PositionDictionary,
        tokenizer: Tokenizer,
        # coco_api: COCO = None,
        automatic_optimization: bool = True,
    ):
        super().__init__(
            cfg,
            category_dict,
            pos_dict,
            tokenizer,
            automatic_optimization,
        )

        logger.info("Initializing ObjGAN module and criterion")
        model_args = (self.cfg, self.tokenizer, self.category_dict, self.pos_dict)
        criterion_args = (self.cfg, self.category_dict, self.pos_dict)
        self.model = ObjGANGenerationModel(*model_args)
        self.train_criterion = ObjGANCriterion(*criterion_args)
        self.eval_criterion = ObjGANCriterion(*criterion_args)
        # self.eval_criterion = Txt2ImgSetCriterion(*criterion_args)
        self.strategy_for_eval_criterion = (  # cfg.detr.generation_strategy_for_eval_criterion
            "greedy"
        )

    def on_validation_epoch_start(self) -> None:
        self.coco_evaluators = {
            strategy: None  # CocoEvaluator(self.coco_api, ('bbox',))
            for strategy in self.cfg.detr.generation_strategies
        }
