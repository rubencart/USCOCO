import logging

from config import Config
from model.text_encoders.text_encoders import (
    AttnGANTextEncoder,
    LMEncoder,
    PLMEncoder,
    SentenceCLIPTextEncoder,
    SequenceEncoder,
    TGEncoder,
)

logger = logging.getLogger("pytorch_lightning")


def build_text_encoder(cfg: Config, tokenizer):
    if cfg.use_tg:
        logger.info("Initializing TG Syntax text encoder")

        text_encoder_class = TGEncoder
    elif cfg.use_plm:
        logger.info("Initializing PLM Syntax text encoder")

        text_encoder_class = PLMEncoder
    else:
        if cfg.text_encoder.text_encoder in ("huggingface", "visualbert", "vokenization"):
            logger.info("Initializing Huggingface text encoder")
            from model.text_encoders.huggingface import HuggingFaceTextEncoder

            text_encoder_class = HuggingFaceTextEncoder

        elif cfg.text_encoder.text_encoder == "gpt2_bllip":
            logger.info("Initializing Qian baseline LM text encoder")

            text_encoder_class = LMEncoder
        elif cfg.text_encoder.text_encoder == "sent_clip":
            logger.info("Initializing CLIP text encoder")

            text_encoder_class = SentenceCLIPTextEncoder
        elif cfg.text_encoder.text_encoder == "attn_gan":
            logger.info("Initializing AttnGAN text encoder")

            text_encoder_class = AttnGANTextEncoder
        else:
            raise NotImplementedError

    if (
        cfg.text_encoder.text_encoder
        not in ("huggingface", "visualbert", "vokenization", "sent_clip")
        and cfg.text_encoder.download_from_hub
        and not cfg.model.download_from_hub
    ):
        logger.info(
            "Loading pretrained text encoder from huggingface hub: %s" % cfg.text_encoder.hub_path
        )
        text_encoder = text_encoder_class.from_pretrained(
            cfg.text_encoder.hub_path,
            cfg=cfg.text_encoder,
            tokenizer=tokenizer,
            cache_dir=cfg.text_encoder.cache_dir,
        )
    else:
        logger.info("Initializing text encoder with random weights")
        text_encoder = text_encoder_class(cfg.text_encoder, tokenizer)

    if not cfg.use_plm and not cfg.use_tg:
        text_encoder = SequenceEncoder(cfg, text_encoder)
    return text_encoder
