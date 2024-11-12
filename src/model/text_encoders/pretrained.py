import logging
from abc import ABC

from config import TextEncoderConfig
from huggingface_hub import PyTorchModelHubMixin
from torch import nn

logger = logging.getLogger("pytorch_lightning")


class PretrainedTextEncoder(nn.Module, ABC, PyTorchModelHubMixin):
    hidden_size: int

    def __init__(self, cfg: TextEncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.finetune = cfg.txt_enc_finetune
        # self.eval()

    def train(self, mode: bool = True):
        # This is here to make sure this model can never be put in training mode,
        # otherwise pytorch lightning will do so at the beginning of the training loop!
        # This changes batchnorm and dropout behavior!
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/2824
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/2551
        continue_training = self.finetune
        return super().train(mode and continue_training)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def get_text_enc_hidden_dim(self):
        return self.hidden_size
