import logging
import pickle

import clip
import torch
from config import Config, TextEncoderConfig
from model.attn_gan import RNNTextEncoder
from model.detr_transformer import TransformerEncoder, TransformerEncoderLayer
from model.plm.base_model import LM
from model.plm.model import PLM
from model.plm.tg_model import TG
from model.position_enc import PositionalSequenceEncoder
from model.text_encoders.pretrained import PretrainedTextEncoder
from torch import Tensor
from torch import nn as nn

logger = logging.getLogger("pytorch_lightning")


class SequenceEncoder(nn.Module):
    def __init__(self, cfg: Config, text_encoder: PretrainedTextEncoder):
        super().__init__()
        self.cfg = cfg
        self.hidden_size = cfg.model.encoder_embed_dim
        self.nopos = cfg.model.no_pos_encoding
        self.only_firstpos = cfg.model.pos_first_layer_only
        self.text_encoder = text_encoder

        if not self.nopos:
            self.pos_encoder = PositionalSequenceEncoder(
                cfg.lt.node_feature_size, self.hidden_size, dropout=cfg.detr.dropout
            )

        if not self.only_firstpos and not self.cfg.model.no_fc_after_text_encoder:
            self.fc = nn.Linear(
                self.text_encoder.hidden_size,
                self.hidden_size,
                bias=cfg.model.sequence_encoder_bias,
            )

    def get_text_enc_hidden_dim(self):
        return self.text_encoder.hidden_size

    def forward(self, input_ids, attention_mask, token_type_ids):
        encoder_out = self.text_encoder(
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )
        sequence_states: Tensor = encoder_out["last_hidden_state"]

        if self.only_firstpos:
            sequence_states = self.pos_encoder(sequence_states)
            pos_embed = None
        else:
            if not self.cfg.model.no_fc_after_text_encoder:
                sequence_states = self.fc(sequence_states)
            if self.nopos:
                pos_embed = None
            else:
                pos_embed = self.pos_encoder.get_pos_emb(sequence_states)

        return {
            **encoder_out,
            "sequence_states": sequence_states,
            "pos_embed": pos_embed,
        }


class PLMEncoder(PretrainedTextEncoder):
    def __init__(self, cfg: TextEncoderConfig, tokenizer):
        super().__init__(cfg)
        self.PLM = PLM(
            is_random_init=True,
            tokenizer=tokenizer,
            huggingface_offline=cfg.huggingface_offline,
            model_name=cfg.architecture,
        )
        self.hidden_size = self.PLM.config.n_embd

        if not cfg.download_from_hub and cfg.plm_checkpoint is not None:
            logger.info("Load parameters from file {}".format(cfg.plm_checkpoint))
            checkpoint = torch.load(cfg.plm_checkpoint)
            self.PLM.model.load_state_dict(checkpoint["model_state_dict"])

        if not self.finetune:
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, input_ids, attention_mask, tokens):
        seq_emb = self.PLM.get_hidden_states(
            input_ids,
            attention_mask,
            tokens,
            add_structured_mask=self.cfg.plm_add_structured_mask,
        )
        return {
            "sequence_states": seq_emb,
            "attention_mask": attention_mask,
        }


class LMEncoder(PretrainedTextEncoder):
    def __init__(self, cfg: TextEncoderConfig, tokenizer):
        super().__init__(cfg)
        self.LM = LM(huggingface_offline=cfg.huggingface_offline, model_name=cfg.architecture)
        self.hidden_size = self.LM.config.n_embd

        if not cfg.download_from_hub and cfg.lm_checkpoint is not None:
            logger.info("Load parameters from file {}".format(cfg.lm_checkpoint))
            checkpoint = torch.load(cfg.lm_checkpoint)
            self.LM.model.load_state_dict(checkpoint["model_state_dict"])

        if not self.finetune:
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, input_ids, attention_mask, *args, **kwargs):
        return {
            "last_hidden_state": self.LM.get_hidden_states(input_ids, attention_mask),
        }


class TGEncoder(PretrainedTextEncoder):
    def __init__(self, cfg: TextEncoderConfig, tokenizer):
        super().__init__(cfg)
        self.tg = TG(
            is_random_init=True,
            tokenizer=tokenizer,
            huggingface_offline=cfg.huggingface_offline,
            model_name=cfg.architecture,
        )
        self.hidden_size = self.tg.config.n_embd

        if not cfg.download_from_hub and cfg.tg_checkpoint is not None:
            logger.info("Load parameters from file {}".format(cfg.tg_checkpoint))
            checkpoint = torch.load(cfg.tg_checkpoint)
            self.tg.model.load_state_dict(checkpoint["model_state_dict"])

        if not self.finetune:
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, input_ids, attention_mask, tokens):
        seq_emb = self.tg.get_hidden_states(input_ids, attention_mask, tokens)
        return {
            "sequence_states": seq_emb,
            "attention_mask": attention_mask,
        }


class TokenCLIPTextEncoder(PretrainedTextEncoder):
    def __init__(self, cfg: TextEncoderConfig, tokenizer):
        super().__init__(cfg)
        logger.info("Loading pretrained clip model %s" % cfg.clip_model_name)
        self.clip_model, _ = clip.load(
            cfg.clip_model_name, device="cpu", jit=False, download_root="cache/clip/"
        )
        self.hidden_size = self.clip_model.transformer.width

        if not cfg.txt_enc_finetune:
            # for p in self.TEXT_ENCODER.parameters():
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        """
        :param input_ids:
        :param attention_mask:
        :return: BS x L x D
        """
        x = self.clip_model.token_embedding(input_ids).type(
            self.clip_model.dtype
        )  # [batch_size, n_ctx, d_model]

        x = x + self.clip_model.positional_embedding.type(self.clip_model.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND

        x = self.clip_model.transformer(x)

        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.clip_model.ln_final(x).type(self.clip_model.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return {"last_hidden_state": x}


class SentenceCLIPTextEncoder(TokenCLIPTextEncoder):
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        """
        :param input_ids:
        :param attention_mask:
        :return: BS x L x D
        """
        x = self.clip_model.encode_text(input_ids).unsqueeze(1)
        return {
            "last_hidden_state": x,
        }


class AttnGANTextEncoder(PretrainedTextEncoder):
    def __init__(self, cfg: TextEncoderConfig, tokenizer):
        super().__init__(cfg)

        self.attn_gan_model = RNNTextEncoder(cfg, tokenizer)
        if not cfg.download_from_hub and cfg.txt_enc_pretrained:
            logger.info(
                "Initializing AttnGAN text encoder from pretrained weights at file %s"
                % cfg.attn_gan_text_encoder_path
            )
            checkpoint = torch.load(
                cfg.attn_gan_text_encoder_path, map_location=lambda storage, loc: storage
            )
            # assert "pytorch-lightning_version" not in checkpoint and "callbacks" not in checkpoint
            state_dict = {
                k.replace("text_encoder.transformer.", ""): v
                for (k, v) in checkpoint["state_dict"].items()
            }
            try:
                self.attn_gan_model.load_state_dict(state_dict)
            except RuntimeError as e:
                print(repr(e))
                if "Missing key(s) in state_dict:" in repr(e):
                    print("Loading with strict=False")
                    self.attn_gan_model.load_state_dict(state_dict, strict=False)

        self.hidden_size = self.attn_gan_model.hidden_dim

        if not cfg.txt_enc_finetune:
            # for p in self.TEXT_ENCODER.parameters():
            for p in self.parameters():
                p.requires_grad = False

        self.N = cfg.attn_gan_extra_encoder_nb_att_layers
        if self.N > 0:
            self.pos_encoder = PositionalSequenceEncoder(
                self.hidden_size, self.hidden_size, dropout=cfg.attn_gan_extra_encoder_dropout
            )
            encoder_layer = TransformerEncoderLayer(
                self.hidden_size,
                cfg.attn_gan_extra_encoder_att_heads,
                cfg.attn_gan_extra_encoder_att_dff,
                cfg.attn_gan_extra_encoder_dropout,
                cfg.attn_gan_extra_encoder_activation,
                cfg.attn_gan_extra_encoder_normalize_before,
            )
            encoder_norm = (
                nn.LayerNorm(self.hidden_size)
                if cfg.attn_gan_extra_encoder_normalize_before
                else None
            )
            self.extra_encoder = TransformerEncoder(encoder_layer, self.N, encoder_norm)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        """
        :param input_ids:
        :param attention_mask:
        :return: BS x L x D
        """
        sequence_states, hidden = self.attn_gan_model(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            },
            return_hidden=True,
        )
        if self.N > 0:
            pos_embed = self.pos_encoder.get_pos_emb(sequence_states)
            sequence_states = self.encoder(
                sequence_states.transpose(0, 1),
                src_key_padding_mask=~attention_mask,
                pos=pos_embed.transpose(0, 1),
            ).transpose(0, 1)
        # words_emb = output.transpose(1, 2)
        # sent_emb = output[:, -1, :]
        nl, ndir = self.cfg.attn_gan_text_encoder_nlayers, (
            2 if self.cfg.attn_gan_text_encoder_bidirectional else 1
        )
        return {
            "last_hidden_state": sequence_states,  # batch x seq_len x num_directions * hidden_size
            # num_layers * num_directions x batch x hidden_size
            "lstm_hidden": hidden,  # .view(nl, ndir, hidden.shape[-2], hidden.shape[-1]),
            "layers_directions": (nl, ndir),
        }
