import logging

import torch
from config import TextEncoderConfig
from model.text_encoders.pretrained import PretrainedTextEncoder
from torch import nn as nn
from transformers import (
    AutoConfig,
    AutoModel,
    PreTrainedTokenizer,
    VisualBertModel,
)

logger = logging.getLogger("pytorch_lightning")


def get_layer_number(param_name, num_hidden):
    if param_name.startswith("embeddings."):
        return -1
    else:
        for i in range(num_hidden):
            if f"layer.{i}." in param_name:
                return i
        if "pooler." in param_name:
            return num_hidden
        else:
            raise ValueError
            return -1


def freeze_first_layers(named_parameters, num_layers_to_freeze, total_layers):
    frozen, not_frozen = [], []
    for n, p in named_parameters:
        layer_num = get_layer_number(n, total_layers)
        if layer_num < num_layers_to_freeze:
            frozen.append(n)
            p.requires_grad = False
        else:
            not_frozen.append(n)
    return frozen, not_frozen


class HuggingFaceTextEncoder(PretrainedTextEncoder):
    def __init__(
        self, cfg: TextEncoderConfig, tokenizer: PreTrainedTokenizer
    ):  # , model_name_or_path: str = None):
        super().__init__(cfg)

        self.model_config = AutoConfig.from_pretrained(
            cfg.hf_model_name_or_path,
            cache_dir=cfg.cache_dir if cfg.cache_dir else None,
            local_files_only=cfg.huggingface_offline,
            torch_dtype="float16" if cfg.use_llama else None,
        )
        if cfg.txt_enc_pretrained:
            logger.info("Loading text encoder from: %s" % cfg.hf_model_name_or_path)
            self.model = AutoModel.from_pretrained(
                cfg.hf_model_name_or_path,
                cache_dir=cfg.cache_dir if cfg.cache_dir else None,
                local_files_only=cfg.huggingface_offline,
                torch_dtype=torch.float16 if cfg.use_llama else None,
            )
            self.hidden_size = self.model_config.hidden_size
        else:
            logger.info("Initializing new text encoder: %s" % cfg.hf_model_name_or_path)
            self.model = AutoModel.from_config(self.model_config)

        # https://huggingface.co/transformers/internal/tokenization_utils.html#transformers.tokenization_utils_base.SpecialTokensMixin.add_special_tokens
        if cfg.add_len_token and self.model.embeddings.word_embeddings.num_embeddings != len(
            tokenizer
        ):
            logger.info(
                "resizing embedding layer from %s..."
                % str(self.model.embeddings.word_embeddings.num_embeddings)
            )
            self.model.resize_token_embeddings(len(tokenizer))
            logger.info(
                "resized embedding layer to %s"
                % str(self.model.embeddings.word_embeddings.num_embeddings)
            )

        self.hidden_size = self.model_config.hidden_size
        self.visual_bert = isinstance(self.model, VisualBertModel)
        if self.visual_bert:
            self.visual_embs = nn.Embedding(
                num_embeddings=1, embedding_dim=self.model_config.visual_embedding_dim
            )

        if not cfg.txt_enc_finetune:
            for p in self.parameters():
                p.requires_grad = False

        if cfg.hf_finetune_last_n_layers > 0:
            frozen, not_frozen = freeze_first_layers(
                self.model.named_parameters(),
                num_layers_to_freeze=self.model_config.num_hidden_layers
                - cfg.hf_finetune_last_n_layers,
                total_layers=self.model_config.num_hidden_layers,
            )
            logger.info(f"Freezing layers {frozen} in Huggingface transformer")
            logger.info(f"Not freezing layers {not_frozen} in Huggingface transformer")

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        """
        :param input_ids:
        :param attention_mask:
        :return:    BS x L x D
        """
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            # 'token_type_ids': token_type_ids,
        }
        if token_type_ids is not None and "llama" not in self.cfg.hf_tokenizer_model_name_or_path:
            inputs["token_type_ids"] = token_type_ids
        if self.visual_bert:
            bs = attention_mask.shape[0]
            inputs["visual_embeds"] = self.visual_embs.weight.data.repeat(bs, 1, 1)
        encoder_out = self.model(**inputs)
        last_hidden_state = encoder_out.last_hidden_state
        if self.visual_bert:
            last_hidden_state = last_hidden_state[:, :-1]
        return {"last_hidden_state": last_hidden_state}
