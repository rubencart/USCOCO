import json
import logging
import os
import pprint
import time
from argparse import ArgumentParser
from typing import Dict, Optional, Tuple

import cv2
import pytorch_lightning as pl
import torch
import wandb
import yaml
from config import Config, ConfigNs
from data.tokenization import Tokenizer
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
)
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.profiler import AdvancedProfiler, PyTorchProfiler
from utils import generate_output_dir_name, initialize_logging, mkdir_p

from probe import dataset
from probe.probe_module import ProbeModule

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
    with open(os.path.join(cfg.run_output_dir, "cfg.json"), "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)

    if cfg.probe.loop_over_layers:
        logger.info("Looping over layers...")
        for layer in list(range(cfg.detr.encoder_layers + 2))[
            cfg.probe.start_at_layer :
        ]:  # +1 for before first layer +1 for before projection
            logger.info("Running for layer %s" % layer)
            cfg, run_name = configure_for_layer(cfg, layer, run_output_dir)
            run_trainer(cfg, train_ns_cfg, run_name)
    else:
        logger.info("Running for single layer: %s" % cfg.probe.current_layer)
        run_trainer(cfg, train_ns_cfg)


def configure_for_layer(cfg, layer, run_output_dir):
    if wandb.run is not None:
        wandb.finish()
        time.sleep(10)

    cfg.probe.current_layer = layer
    cfg.run_output_dir = os.path.join(run_output_dir, f"layer_{layer}")
    mkdir_p(cfg.run_output_dir)

    run_name = f"{os.path.split(run_output_dir)[1]}_layer_{layer}"

    return cfg, run_name


def run_trainer(cfg, train_ns_cfg, run_name=None):
    text_tokenizer = Tokenizer.build_tokenizer(cfg.text_encoder, use_fast=False)
    dls = dataset.build_dataloaders(cfg, text_tokenizer)

    trainer_kwargs, model_checkpoint = build_trainer_kwargs(cfg, cfg.run_output_dir, run_name)
    trainer = Trainer.from_argparse_args(
        train_ns_cfg,
        **trainer_kwargs,
    )

    logger.info("Starting new module")
    model = ProbeModule(cfg, dls["tag_dict"], dls["emb_dim"])

    ckpt_path = None
    if cfg.continue_training_from_checkpoint or cfg.load_weights_from_checkpoint:
        ckpt_path = cfg.checkpoint

    if cfg.do_test:
        # model.save_predictions_to_file = True
        trainer.test(
            model,
            dataloaders=[dls["absurd"], dls["val"], dls["newval"] if "newval" in dls else None],
            ckpt_path=ckpt_path,
        )
    else:
        assert cfg.do_train
        trainer.fit(
            model, train_dataloaders=dls["train"], val_dataloaders=dls["val"], ckpt_path=ckpt_path
        )
        # model.save_predictions_to_file = True
        trainer.test(dataloaders=[dls["absurd"], dls["val"], dls["newval"]], ckpt_path="best")

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


def build_trainer_kwargs(
    cfg: Config, run_output_dir: str, run_name: Optional[str] = None
) -> Tuple[Dict, ModelCheckpoint]:
    tags = [str(cfg.seed)]
    if run_name is None:
        run_name = os.path.split(run_output_dir)[1]
    wandb_logger = WandbLogger(
        name=run_name,
        project=cfg.wandb_project_name,
        save_dir=run_output_dir,
        offline=cfg.wandb_offline,
        save_code=True,
        tags=tags,
        config=cfg.to_dict(),
    )

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

    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)

    trainer_kwargs.update({
        "callbacks": callbacks,
        "check_val_every_n_epoch": 1,
        "sync_batchnorm": True,
        # 'num_sanity_val_steps': -1,
    })
    if cfg.debug:
        torch.autograd.set_detect_anomaly(True)
        trainer_kwargs.update({
            "num_sanity_val_steps": 0,
            "min_epochs": 1,
            "max_epochs": cfg.debug_max_epochs,
            # 'limit_train_batches': 30,
            # 'overfit_batches': 100,
            # 'limit_val_batches': 10,
            "log_every_n_steps": 10,
        })
        if cfg.overfit_on_val_samples > 0:
            trainer_kwargs.update({
                "check_val_every_n_epoch": 5, "min_epochs": 1000, "max_epochs": 1000
            })
        else:
            trainer_kwargs.update({
                "limit_train_batches": 100, "limit_val_batches": 5, "limit_test_batches": 10
            })
    if not cfg.do_validate_during_training:
        trainer_kwargs.update({"limit_val_batches": 0.0})

    if cfg.running_on_vsc_server and not cfg.debug:
        trainer_kwargs.update({
            "enable_progress_bar": False,
        })

    return trainer_kwargs, model_checkpoint


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
    _cfg, _train_ns_cfg = build_configs()
    main(_cfg, _train_ns_cfg)
