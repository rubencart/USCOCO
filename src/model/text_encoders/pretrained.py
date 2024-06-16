import logging
from abc import ABC

from config import TextEncoderConfig
from torch import nn

logger = logging.getLogger("pytorch_lightning")


class PretrainedTextEncoder(nn.Module, ABC):
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

    @classmethod
    def build_text_encoder(cls, cfg, tokenizer):
        if cfg.use_tg:
            logger.info("Initializing TG Syntax text encoder")
            from model.text_encoders.text_encoders import TGEncoder

            text_encoder = TGEncoder(cfg.text_encoder, tokenizer)
        elif cfg.use_plm:
            logger.info("Initializing PLM Syntax text encoder")
            from model.text_encoders.text_encoders import PLMEncoder

            text_encoder = PLMEncoder(cfg.text_encoder, tokenizer)
        else:
            if cfg.text_encoder.text_encoder in ("huggingface", "visualbert", "vokenization"):
                logger.info("Initializing Huggingface text encoder")
                from model.text_encoders.huggingface import (
                    HuggingFaceTextEncoder,
                )

                text_encoder = HuggingFaceTextEncoder(cfg.text_encoder, tokenizer)
            elif cfg.text_encoder.text_encoder == "qian_base_lm":
                logger.info("Initializing Qian baseline LM text encoder")
                from model.text_encoders.text_encoders import LMEncoder

                text_encoder = LMEncoder(cfg.text_encoder)
            elif cfg.text_encoder.text_encoder == "clip":
                logger.info("Initializing CLIP text encoder")
                from model.text_encoders.text_encoders import (
                    TokenCLIPTextEncoder,
                )

                text_encoder = TokenCLIPTextEncoder(cfg.text_encoder, tokenizer)
            elif cfg.text_encoder.text_encoder == "sent_clip":
                logger.info("Initializing CLIP text encoder")
                from model.text_encoders.text_encoders import (
                    SentenceCLIPTextEncoder,
                )

                text_encoder = SentenceCLIPTextEncoder(cfg.text_encoder, tokenizer)
            elif cfg.text_encoder.text_encoder in ("attn_gan", "rp_transformer", "rp_huggingface"):
                logger.info("Initializing AttnGAN text encoder")
                from model.text_encoders.text_encoders import (
                    AttnGANTextEncoder,
                )

                text_encoder = AttnGANTextEncoder(cfg.text_encoder, tokenizer)
            else:
                raise NotImplementedError
            from model.text_encoders.text_encoders import SequenceEncoder

            text_encoder = SequenceEncoder(cfg, text_encoder)
        return text_encoder
