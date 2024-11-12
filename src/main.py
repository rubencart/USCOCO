import json
import logging
import os
import pprint
from argparse import ArgumentParser
from typing import Dict, Tuple, Type

import cv2
import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
)
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.profiler import AdvancedProfiler, PyTorchProfiler

from config import Config, ConfigNs
from data.data_module import COCODataModule
from data.tokenization import Tokenizer, Vocabulary
from pl_modules import (
    AutoregressiveGenerationModule,
    DETRGenerationModule,
    ObjGANGenerationModule,
)
from utils import generate_output_dir_name, initialize_logging, mkdir_p

os.environ["TORCH_HOME"] = "../.torch"
logger = logging.getLogger("pytorch_lightning")


def main(cfg, train_ns_cfg):
    # Make sure to set the random seed before the instantiation of a Trainer() so
    # that each model initializes with the same weights.
    # https://pytorch-lightning.readthedocs.io/en/stable/multi_gpu.html#distributed-data-parallel
    seed_everything(cfg.seed)

    cv2.setNumThreads(0)

    local_rank = os.environ.get("LOCAL_RANK", 0)

    if local_rank == 0:
        run_output_dir = generate_output_dir_name()
        mkdir_p(run_output_dir)
        os.environ["RUN_OUTPUT_DIR"] = run_output_dir
    else:
        run_output_dir = os.environ["RUN_OUTPUT_DIR"]

    initialize_logging(run_output_dir, local_rank=local_rank, to_file=True)
    cfg.run_output_dir = run_output_dir

    logger.info(pprint.pformat(cfg.to_dict(), indent=2))
    # cfg.save(os.path.join(cfg.run_output_dir, 'cfg.json'), skip_unpicklable=True)
    with open(os.path.join(cfg.run_output_dir, "cfg.json"), "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)

    text_tokenizer = Tokenizer.build_tokenizer(cfg.text_encoder)
    data_module = COCODataModule(cfg, text_tokenizer)
    data_module.prepare_data()  # doesn't do anything for now

    data_module.setup("test" if cfg.do_test else "fit")
    model, model_class = make_model(data_module, cfg, text_tokenizer)
    set_number_of_params(cfg, model)

    trainer_kwargs, model_checkpoint = build_trainer_kwargs(cfg, run_output_dir)
    trainer = Trainer.from_argparse_args(
        train_ns_cfg,
        **trainer_kwargs,
    )

    if cfg.do_test:
        model.save_predictions_to_file = True
        trainer.test(model, dataloaders=data_module.test_dataloader())
    else:
        assert cfg.do_train
        trainer.fit(
            model,
            train_dataloaders=data_module.train_dataloader(),
            val_dataloaders=data_module.val_dataloader(),
        )
        model.save_predictions_to_file = True
        logger.info("loading best checkpoint %s" % model_checkpoint.best_model_path)
        model = model_class.load_from_checkpoint(
            model_checkpoint.best_model_path,
            cfg=cfg,
            category_dict=data_module.category_dict,
            pos_dict=data_module.pos_dict,
            tokenizer=text_tokenizer,
            strict=not cfg.text_encoder.use_llama,
        )
        trainer.test(model, dataloaders=data_module.test_dataloader())

        if cfg.do_validate_during_training:
            logger.info(
                "Best model: %s, %s"
                % (model_checkpoint.best_model_score, model_checkpoint.best_model_path)
            )


def set_number_of_params(cfg, model):
    # todo doesn't work as supposed
    summary = pl.utilities.model_summary.summarize(model)
    cfg.total_parameters = summary.total_parameters
    cfg.trainable_parameters = summary.trainable_parameters


def build_trainer_kwargs(cfg: Config, run_output_dir: str) -> Tuple[Dict, ModelCheckpoint]:
    tags = ["ar" if cfg.model.autoregressive else "detr"]
    tags += ["14" if cfg.dataset == "coco14" else "17"]
    tags += [str(cfg.seed)]
    # todo resume wandb if cfg.continue_training_from_checkpoint...
    wandb_logger = WandbLogger(
        name=f"{os.path.split(run_output_dir)[1]}_{cfg.cfg_name}",
        project=cfg.wandb_project_name,
        entity=cfg.wandb_org_name,
        save_dir=run_output_dir,
        offline=cfg.wandb_offline,
        save_code=True,
        tags=tags,
        config=cfg.to_dict(),
    )
    for metric, summary in (
        ("precision_iou_00", "max"),
        ("precision_iou_05", "max"),
        ("recall_iou_00", "max"),
        ("recall_iou_05", "max"),
        ("coco_ap_iou_05", "max"),
        ("coco_ap_iou_00", "max"),
        ("coco_ap_iou_05-095", "max"),
        ("coco_ar_iou_00", "max"),
        ("coco_ar_iou_05-095", "max"),
        ("delta_pairwise_distances", "min"),
        ("delta_pairwise_distances_scaled_diags", "min"),
        ("delta_pairwise_norm_width_height_diffs", "min"),
        ("delta_pairwise_x_distances_norm_w", "min"),
        ("delta_pairwise_y_distances_norm_h", "min"),
        ("f1_iou_05", "max"),
        ("f1_iou_00", "max"),
    ):
        logger.info("defining metric as %s: %s" % (summary, metric))
        wandb_logger.experiment.define_metric(metric, summary=summary)
        wandb_logger.experiment.define_metric(metric, summary="last")

    trainer_kwargs = {
        "logger": wandb_logger,
        # 'weights_summary': 'full',
        "deterministic": cfg.deterministic,
    }
    if cfg.profiler is not None:
        if cfg.profiler == "advanced":
            profiler = AdvancedProfiler(output_filename=os.path.join(run_output_dir, "profile.txt"))
        else:
            assert cfg.profiler == "pytorch"
            profiler = PyTorchProfiler(output_filename=os.path.join(run_output_dir, "profile.txt"))
        trainer_kwargs["profiler"] = profiler

    callbacks = []
    model_checkpoint = None
    if cfg.do_validate_during_training:
        model_checkpoint = ModelCheckpoint(
            monitor=cfg.model_checkpoint_monitor,
            # monitor='mean_training_loss',
            dirpath=os.path.join(run_output_dir, "checkpoints"),
            filename="{epoch}-{step}",
            save_top_k=1,
            mode=cfg.model_checkpoint_monitor_min_or_max,
            save_last=True,
        )
        callbacks.append(model_checkpoint)
    if cfg.early_stop not in [None, ""]:  # todo handle properly
        logger.info("Using early stopping on %s" % cfg.early_stop)
        early_stopping = EarlyStopping(
            cfg.early_stop,
            patience=cfg.early_stop_patience,
            mode=cfg.early_stop_min_or_max,
            min_delta=0.0001,
            strict=False,  # so monitoring only when epochs > E todo does this work?
            verbose=True,
        )
        callbacks.append(early_stopping)
    callbacks.append(ModelSummary(max_depth=10))

    trainer_kwargs.update({
        "callbacks": callbacks,
        "check_val_every_n_epoch": 1,
        "sync_batchnorm": True,
        "num_sanity_val_steps": 2,
    })
    if cfg.debug:
        torch.autograd.set_detect_anomaly(True)
        trainer_kwargs.update({
            "num_sanity_val_steps": 2,
            "min_epochs": 1,
            "max_epochs": cfg.debug_max_epochs,
            # 'limit_train_batches': 30,
            # 'overfit_batches': 100,
            # 'limit_val_batches': 10,
            "log_every_n_steps": 10,
        })
        lr_monitor = LearningRateMonitor(logging_interval=None)
        callbacks.append(lr_monitor)
        if cfg.overfit_on_val_samples > 0:
            trainer_kwargs.update({
                "check_val_every_n_epoch": 5, "min_epochs": 1000, "max_epochs": 1000
            })
        else:
            trainer_kwargs.update({
                "limit_train_batches": 20, "limit_val_batches": 5, "limit_test_batches": 10
            })
    if not cfg.do_validate_during_training:
        trainer_kwargs.update({"limit_val_batches": 0.0})
    if cfg.continue_training_from_checkpoint:
        trainer_kwargs.update({
            "resume_from_checkpoint": cfg.checkpoint,
        })

    return trainer_kwargs, model_checkpoint


def make_model(
    dm: COCODataModule, cfg: Config, tokenizer: Tokenizer
) -> Tuple[pl.LightningModule, Type[pl.LightningModule]]:
    if cfg.model.obj_gan:
        module_class = ObjGANGenerationModule
    elif cfg.model.autoregressive:
        module_class = AutoregressiveGenerationModule
    else:
        module_class = DETRGenerationModule

    if cfg.load_weights_from_checkpoint or cfg.continue_training_from_checkpoint:
        logger.info("loading from checkpoint %s" % cfg.checkpoint)
        try:
            model = module_class.load_from_checkpoint(
                cfg.checkpoint,
                cfg=cfg,
                category_dict=dm.category_dict,
                pos_dict=dm.pos_dict,
                tokenizer=tokenizer,
                strict=not cfg.text_encoder.use_llama,
            )
        except RuntimeError as e:
            logger.exception(e)
            # raise e
            model = module_class.load_from_checkpoint(
                cfg.checkpoint,
                cfg=cfg,
                category_dict=dm.category_dict,
                pos_dict=dm.pos_dict,
                tokenizer=tokenizer,
                strict=False,
            )

    else:
        logger.info("starting new model")
        model = module_class(
            cfg,
            category_dict=dm.category_dict,
            pos_dict=dm.pos_dict,
            tokenizer=tokenizer,
        )

    return model, module_class


def build_configs() -> Tuple[Config, ConfigNs]:
    parser = ArgumentParser()
    parser.add_argument("--cfg", type=str)
    parser.add_argument("--seed", type=int, default=-1)

    args = parser.parse_args()

    with open(args.cfg, "r") as f:
        dict_cfg = yaml.load(f, Loader=yaml.FullLoader)

    cfg = Config()
    cfg.from_dict(dict_cfg)
    cfg.process_args()

    print(cfg.train.as_dict())
    train_ns_cfg = ConfigNs(cfg.train.as_dict())

    if args.seed > 0:
        cfg.seed = args.seed
    return cfg, train_ns_cfg


if __name__ == "__main__":
    cfg, train_ns_cfg = build_configs()
    main(cfg, train_ns_cfg)
