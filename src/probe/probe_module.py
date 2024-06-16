import logging
import os
from typing import Any, Dict

import numpy as np
import pytorch_lightning as pl
import torch
from config import Config
from torch import Tensor, optim
from torch.optim.lr_scheduler import LinearLR, ReduceLROnPlateau, SequentialLR
from torchmetrics.functional.classification import multiclass_f1_score

from probe.probe_model import ProbeLoss, ProbeModel
from probe.probe_utils import ConstituentDictionary

logger = logging.getLogger("pytorch_lightning")


class ProbeModule(pl.LightningModule):
    """
    https://pytorch-lightning.readthedocs.io/en/latest/advanced/transfer_learning.html
    """

    def __init__(self, cfg: Config, tag_dict: ConstituentDictionary, embedding_dim: int):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(cfg.to_dict(), ignore=["tag_dict"])

        self.predictions: Dict[str, Any] = {}
        self.statistics: Dict[str, Any] = {}
        self.save_predictions_to_file = False
        self.save_predictions_location = os.path.join(self.cfg.run_output_dir, "preds.pkl")

        logger.info("Initializing model and criterion")
        self.probe_model = ProbeModel(cfg, tag_dict, embedding_dim)
        self.criterion = ProbeLoss()
        self.tag_dict = tag_dict

    def configure_optimizers(self):
        cfg = self.cfg.probe
        optimizer = optim.Adam(
            self.parameters(),
            cfg.lr,
            betas=(cfg.adam_beta_1, cfg.adam_beta_2),
            eps=cfg.adam_eps,
            weight_decay=cfg.weight_decay,
        )
        logger.info(optimizer)

        schedulers = []
        if self.cfg.probe.lr_schedule is not None:
            logger.info("Using LR scheduler: %s" % self.cfg.probe.lr_schedule)
            if self.cfg.probe.lr_schedule == "linear_with_warmup":
                raise NotImplementedError
                scheduler1 = LinearLR(
                    optimizer,
                    start_factor=1 / self.cfg.probe.lr_schedule_warmup_epochs,
                    end_factor=1.0,
                    total_iters=self.cfg.probe.lr_schedule_warmup_epochs - 1,
                )
                scheduler2 = LinearLR(
                    optimizer,
                    start_factor=1.0,
                    end_factor=1e-8,
                    total_iters=self.cfg.train.max_epochs
                    - self.cfg.probe.lr_schedule_warmup_epochs,
                )
                scheduler = SequentialLR(
                    optimizer,
                    [scheduler1, scheduler2],
                    milestones=[self.cfg.probe.lr_schedule_warmup_epochs - 1],
                )
                schedulers.append(scheduler)
            elif self.cfg.probe.lr_schedule == "reduce_on_plateau":
                # https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.core.LightningModule.html#pytorch_lightning.core.LightningModule.configure_optimizers
                val_every_n_steps = self.cfg.probe.val_every_n_steps
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": ReduceLROnPlateau(
                            optimizer,
                            mode=self.cfg.probe.lr_schedule_mode,
                            factor=self.cfg.probe.lr_schedule_factor,
                            patience=self.cfg.probe.lr_schedule_patience,
                            verbose=True,
                        ),
                        "monitor": self.cfg.probe.lr_schedule_monitor,
                        "interval": "step" if val_every_n_steps > 1 else "epoch",
                        "frequency": val_every_n_steps if val_every_n_steps > 1 else 1,
                    },
                }

        return [optimizer], schedulers

    def forward(self, batch):
        scores = self.probe_model(batch)
        return {"scores": scores}

    def on_train_epoch_start(self) -> None:
        if self.cfg.probe.lr_schedule not in (None, "reduce_on_plateau"):
            logger.info(f"Current LR: {self.lr_schedulers().get_last_lr()}")

    def training_step(self, batch, batch_idx: int):
        scores = self.probe_model(batch)
        loss_output = self.criterion(batch, scores)
        log_dict = loss_output

        # exclude aux loss logs
        excludes = ["logits", "tags"]
        self.log_dict(
            {
                k: v
                for k, v in log_dict.items()
                if not any([k.startswith(excl) for excl in excludes])
            },
            # on_step=True, on_epoch=True, logger=True, prog_bar=False,
            batch_size=batch["embeddings"].shape[0],
        )
        return log_dict

    def test_step(self, batch, batch_idx, dataloader_idx):
        return self.valtest_step(batch, "test")

    def validation_step(self, batch, batch_idx: int):
        return self.valtest_step(batch, "val")

    def valtest_step(self, batch, stage: str):
        scores = self.probe_model(batch)
        loss_output = self.criterion(batch, scores)

        excludes = ["logits", "tags"]
        self.log_dict(
            {
                f"{stage}_{k}": v
                for k, v in loss_output.items()
                if not any([k.startswith(excl) for excl in excludes])
            },
            # on_step=True, on_epoch=True, logger=True, prog_bar=False,
            batch_size=batch["embeddings"].shape[0],
        )
        if torch.isnan(loss_output["loss"]):
            logger.error(f"{stage}_loss is NaN!")
        return {
            "scores": scores.detach(),
            "loss": loss_output["loss"].detach(),
            # 'batch': copy.deepcopy(batch),
            "batch": batch,
        }

    def test_epoch_end(self, outputs):
        all_results = {}
        dsets = (
            ("absurd", "test_abs"),
            ("val", "test_indom"),
            ("newval", "new_val"),
        )
        for i, (split, prefix) in enumerate(dsets):
            rdict = self.compute_metrics(outputs[i])
            results = {f"{prefix}_{k}": v for k, v in rdict["result_dict"].items()}
            self.statistics.update(results)
            self.log_dict(results)
            all_results[prefix] = rdict

        if self.save_predictions_to_file:
            logger.info("Saving test preds to %s" % self.save_predictions_location)
            with open(self.save_predictions_location, "wb") as f:
                torch.save(all_results, f)

        # return super().test_epoch_end(outputs)

    def validation_epoch_end(self, list_of_step_outputs):
        ps = [p.view(-1) for p in self.parameters()]
        self.log("param_norm", torch.cat(ps).detach().norm())

        rdict = self.compute_metrics(list_of_step_outputs)
        self.log_dict({f"val_{k}": v for k, v in rdict["result_dict"].items()})

        return super().validation_epoch_end(list_of_step_outputs)

    def compute_metrics(self, output_list) -> Dict[str, Tensor]:
        caption_ids = []
        span_scores, span_tags_per_sent = [], []
        for d in output_list:
            batch, scores = d["batch"], d["scores"]
            caption_ids.extend(batch["caption_ids"])
            span_scores.extend(scores)
            span_tags_per_sent.extend([
                torch.tensor(tgs).type_as(scores).long() for tgs in batch["tags"]
            ])

        span_scores_t = torch.stack(span_scores)
        span_tags_t = torch.cat(span_tags_per_sent)
        span_scores_per_sent = []
        i = 0
        for tags in span_tags_per_sent:
            span_scores_per_sent.append(span_scores_t[i : i + tags.shape[0]])
            i += tags.shape[0]

        span_preds = span_scores_t.argmax(dim=-1)
        span_preds_per_sent = [s.argmax(dim=-1) for s in span_scores_per_sent]

        result_dict = {}

        # count tag frequencies
        pred_tag_bins = (
            torch.bincount(span_preds, minlength=len(self.tag_dict)).cpu() / span_preds.shape[0]
        )
        gt_tag_bins = (
            torch.bincount(span_tags_t, minlength=len(self.tag_dict)).cpu() / span_tags_t.shape[0]
        )
        for i, b in enumerate(pred_tag_bins):
            result_dict[f"freq_pred_{self.tag_dict.tokens[i]}"] = b
        for i, b in enumerate(gt_tag_bins):
            result_dict[f"freq_gt_{self.tag_dict.tokens[i]}"] = b

        # filter out tags that do not occur in this dataset (are not in GT and not in preds)
        cum_tags = (gt_tag_bins + pred_tag_bins).gt(0).long().cumsum(-1) - 1
        # cum_gt_tags = gt_tag_bins.gt(0).long().cumsum(-1) - 1
        mapped_span_preds = torch.tensor([cum_tags[v] for v in span_preds]).type_as(span_tags_t)
        mapped_span_tags = torch.tensor([cum_tags[v] for v in span_tags_t]).type_as(span_tags_t)

        # todo add majority baselines
        # todo add f1 per tag category to see if e.g. VPs are easier than NPs

        sent_result_dict: Dict = {}
        for avg in ("micro", "macro"):
            num_classes = int(cum_tags.max()) + 1
            result_dict.update({
                f"sent_micro_tag_{avg}_f1": multiclass_f1_score(
                    mapped_span_preds, mapped_span_tags, num_classes=num_classes, average=avg
                ).cpu(),
            })

            for capt_id, preds, tags in zip(caption_ids, span_preds_per_sent, span_tags_per_sent):
                sent_pred_tag_bins = (
                    torch.bincount(preds, minlength=len(self.tag_dict)).cpu() / preds.shape[0]
                )
                sent_gt_tag_bins = (
                    torch.bincount(tags, minlength=len(self.tag_dict)).cpu() / tags.shape[0]
                )
                sent_cum_tags = (sent_gt_tag_bins + sent_pred_tag_bins).gt(0).long().cumsum(-1) - 1
                sent_num_classes = int(sent_cum_tags.max()) + 1

                mapped_preds = torch.tensor([sent_cum_tags[v] for v in preds]).type_as(span_tags_t)
                mapped_tags = torch.tensor([sent_cum_tags[v] for v in tags]).type_as(span_tags_t)

                sent_res = {
                    "num_classes": sent_num_classes,
                }
                if sent_num_classes > 1:
                    sent_res[f"tag_{avg}_f1"] = multiclass_f1_score(
                        mapped_preds, mapped_tags, num_classes=sent_num_classes, average=avg
                    ).cpu()
                sent_result_dict.setdefault(capt_id, dict()).update(sent_res)

            result_dict.update({
                f"sent_macro_tag_{avg}_f1": np.mean([
                    v[f"tag_{avg}_f1"]
                    for (cid, v) in sent_result_dict.items()
                    if f"tag_{avg}_f1" in v
                ]),
            })

        predictions = {
            "result_dict": result_dict,
            "sent_result_dict": sent_result_dict,
            "span_tags_per_sent": span_tags_per_sent,
            "span_scores": span_scores,
            "span_tags": span_tags_t,
            "span_scores_per_sent": span_scores_per_sent,
        }
        return predictions
