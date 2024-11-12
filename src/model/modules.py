import copy
import itertools
import json
import logging
from typing import Any, Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config, DETRConfig
from data.dictionary import CategoryDictionary, PositionDictionary
from data.tokenization import Tokenizer
from fairseq.data.dictionary import Dictionary
from torch import Tensor
from transformers import TopPLogitsWarper

from model.detr_transformer import (
    MLP,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
)
from model.obj_gan import DecoderRNN
from model.position_enc import PositionalTreeEncoder
from model.text_encoders.pretrained import PretrainedTextEncoder
from model.text_encoders.text_encoders import SequenceEncoder

logger = logging.getLogger("pytorch_lightning")


class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root)."""
        super(BertLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


def init_bert_weights(module):
    """Initialize the weights.
    From FairSeq (bert_seq2seq.py)
    """
    if isinstance(module, (nn.Linear, nn.Embedding)):
        # Slightly different from the TF version which uses truncated_normal for initialization
        # cf https://github.com/pytorch/pytorch/pull/5617
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, BertLayerNorm):
        module.beta.data.zero_()
        module.gamma.data.fill_(1.0)
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


class _SuperModel(nn.Module):
    def __init__(
        self,
        cfg: Config,
        tokenizer: Tokenizer,
        category_dict: Dictionary,
        pos_dict: PositionDictionary,
    ):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.category_dict = category_dict
        self.pos_dict = pos_dict

        self.text_encoder = PretrainedTextEncoder.build_text_encoder(cfg, tokenizer)

        self.detr_encoder = cfg.model.use_additional_detr_encoder
        self.text_enc_hdim = self.text_encoder.hidden_size
        self.layout_dec_hdim = cfg.detr.decoder_embed_dim
        self.layout_enc_hdim = (
            cfg.model.encoder_embed_dim if self.detr_encoder else self.text_enc_hdim
        )
        self.project_text_2_enc = self.text_encoder.hidden_size != cfg.model.encoder_embed_dim

        if self.detr_encoder:
            self.detr_encoder = DETRBasedEncoder(
                cfg.detr, self.layout_enc_hdim, return_intermediate=cfg.save_probe_embeddings
            )
            if self.project_text_2_enc:
                self.text_enc_detr_enc_project = nn.Linear(
                    self.text_enc_hdim, self.layout_enc_hdim, bias=False
                )

        if self.cfg.detr.load_span_tree_pos_embs:
            self.span_tree_embedder = PositionalTreeEncoder(
                cfg.lt.node_feature_size,
                self.layout_dec_hdim,
                n=cfg.lt.n,
                k=cfg.lt.k,
                p_repeats=cfg.lt.p_repeats,
                dropout=cfg.detr.dropout,
            )

        # if self.cfg.model.predict_num_queries:
        self.length_projection = nn.Linear(
            in_features=self.layout_enc_hdim,
            out_features=cfg.model.max_target_positions,
        )
        # already included in huggingface roberta output
        # self.length_dropout = nn.Dropout(p=cfg.model.length_dropout)
        self.length_projection.apply(init_bert_weights)
        self.seq_states_dropout = nn.Dropout(p=cfg.detr.dropout)

    def text_encoder_forward(
        self,
        batch,
        # input_ids, attention_mask, token_type_ids=None,
        seq_len_first=False,
    ):
        out = {}

        # TEXT ENCODE
        input_ids, attention_mask = batch["input_ids"], batch["attention_mask"]
        token_type_ids = batch.get("token_type_ids", None)
        predicted_lengths, pos_embed = None, None
        if self.cfg.use_plm or self.cfg.use_tg:
            encoder_out = self.text_encoder(input_ids, attention_mask, batch["tokens"])
        elif self.cfg.text_encoder.text_encoder == "sent_clip":
            encoder_out = self.text_encoder(input_ids, attention_mask, token_type_ids)
            batch["attention_mask"] = attention_mask = torch.ones(input_ids.shape[0], 1).type_as(
                attention_mask
            )
        else:
            if self.cfg.model.predict_num_queries and self.cfg.text_encoder.add_len_token:
                bs = input_ids.shape[0]
                input_ids = torch.cat(
                    (
                        input_ids[:, 0].unsqueeze(-1),
                        torch.LongTensor([self.tokenizer.len_token_id])
                        .repeat((bs, 1))
                        .type_as(input_ids),
                        input_ids[:, 1:],
                    ),
                    dim=1,
                )
                attention_mask = torch.cat(
                    (
                        attention_mask[:, 0].unsqueeze(-1),
                        torch.LongTensor([1]).repeat((bs, 1)).type_as(attention_mask),
                        attention_mask[:, 1:],
                    ),
                    dim=1,
                )
                if token_type_ids is not None:
                    token_type_ids = torch.cat(
                        (
                            token_type_ids[:, 0].unsqueeze(-1),
                            token_type_ids[:, 0].unsqueeze(-1),
                            token_type_ids[:, 1:],
                        ),
                        dim=1,
                    )

            encoder_out = self.text_encoder(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
            )
        sequence_states, pos_embed = encoder_out["sequence_states"], encoder_out.get(
            "pos_embed", None
        )

        # DETR ENCODE
        if self.cfg.save_probe_embeddings:
            # sequence_states is already projected to encoder_dim if text encoder is
            # wrapped by SequenceEncoder
            if (
                isinstance(self.text_encoder, SequenceEncoder)
                and "last_hidden_state" in encoder_out
            ):
                out["probe_embeddings_inp"] = encoder_out["last_hidden_state"]
            else:
                out["probe_embeddings_inp"] = sequence_states

        sequence_states = self.seq_states_dropout(sequence_states)
        states_for_length = sequence_states
        if self.detr_encoder:
            if self.project_text_2_enc:
                sequence_states = self.text_enc_detr_enc_project(sequence_states)

            detr_enc_out = self.detr_encoder(sequence_states, attention_mask, pos_embed=pos_embed)
            if self.cfg.save_probe_embeddings:
                states_for_length = detr_enc_out[-1]
                # n_layers x bs x n_tokens x dim --> bs x n_layers x ...
                out["probe_embeddings"] = detr_enc_out.transpose(1, 0)
            else:
                states_for_length = detr_enc_out.squeeze(0)

            states_for_length = self.seq_states_dropout(states_for_length)
            if not self.cfg.model.detr_encoder_for_length_only:
                sequence_states = states_for_length

        # LENGTH PREDICTION
        if (
            self.cfg.use_plm
            or self.cfg.use_tg
            or self.cfg.text_encoder.text_encoder
            in (
                "huggingface",
                "vokenization",
                "attn_gan",
                "gpt2_bllip",
            )
        ):
            length_states = states_for_length[:, 0, :]
            sent_embs, rest_embs = sequence_states[:, 0, :], sequence_states[:, 1:, :]
        elif self.cfg.text_encoder.text_encoder == "sent_clip":
            length_states = states_for_length[:, 0]
            sent_embs = sequence_states
            rest_embs = sequence_states
        # elif self.cfg.model.predict_num_queries and self.cfg.text_encoder.add_len_token:
        else:
            assert self.cfg.model.predict_num_queries and self.cfg.text_encoder.add_len_token
            # 2nd token is <len> token, like in DisCo
            # BS x h
            length_states = states_for_length[:, 1, :]
            sequence_states = torch.cat(
                (sequence_states[:, :1, :], sequence_states[:, 2:, :]), dim=1
            )
            sent_embs, rest_embs = sequence_states[:, 0, :], sequence_states[:, 1:, :]
            attention_mask = torch.cat((attention_mask[:, :1], attention_mask[:, 2:]), dim=1)
            pos_embed = (
                torch.cat((pos_embed[:, :1], pos_embed[:, 2:]), dim=1)
                if pos_embed is not None
                else pos_embed
            )

        if self.cfg.model.predict_num_queries:
            # BS x h --> BS x max_len
            predicted_lengths = self.length_projection(length_states)

        span_tree_embs = None
        if self.cfg.detr.load_span_tree_pos_embs:
            # todo something with mask?
            span_tree_embs = self.span_tree_embedder.get_pos_emb(batch["tree_node_pos"])

        out.update({
            **encoder_out,
            "length_states": length_states,
            "predicted_lengths": predicted_lengths,
            "encoder_out": sequence_states,
            #  if seq_len_first else attention_mask,
            "encoder_padding_mask": ~attention_mask.bool(),
            "pos_embed": pos_embed,
            "sent_embed": sent_embs,
            "rest_embed": rest_embs,
            "span_tree_embed": span_tree_embs,
            "span_tree_embed_mask": batch["tree_node_mask"],
        })

        return out


class DETRGenerationModel(_SuperModel):
    get_queries: Callable[[Any], Tensor]

    def __init__(
        self,
        cfg: Config,
        tokenizer: Tokenizer,
        cat_dict: CategoryDictionary,
        pos_dict: PositionDictionary,
    ):
        super().__init__(cfg, tokenizer, cat_dict, pos_dict)

        self.num_queries = cfg.detr.num_queries
        self.dec_dim = cfg.detr.decoder_embed_dim
        self.query_noise_agg = cfg.detr.query_noise_aggregate
        self.learnt_query_embeds = cfg.detr.learnt_query_embeds
        self.noise_as_queries = cfg.detr.noise_as_queries
        self.query_pos_first_layer_only = cfg.detr.query_pos_first_layer_only

        assert self.query_noise_agg != "concat" or (
            self.learnt_query_embeds and self.query_pos_first_layer_only
        )
        self.query_noise_dim = (
            self.dec_dim // 2 if self.query_noise_agg == "concat" else self.dec_dim
        )

        assert self.learnt_query_embeds or self.noise_as_queries
        if self.learnt_query_embeds:
            self.learnt_query_embed = nn.Embedding(cfg.detr.num_queries, self.query_noise_dim)
            self.learnt_query_embed_dropout = nn.Dropout(p=cfg.detr.dropout)
            nn.init.normal_(self.learnt_query_embed.weight, mean=0.0, std=0.02)
            self.learnt_query_embed_layernorm = nn.LayerNorm(self.query_noise_dim)
        if self.noise_as_queries:
            # self.queries = torch.randn(cfg.detr.num_queries, cfg.detr.decoder_embed_dim)
            self.register_buffer(
                "noise_queries", torch.randn(cfg.detr.num_queries, self.query_noise_dim)
            )
            self.noise_queries_dropout = nn.Dropout(p=cfg.detr.dropout)
            self.noise_queries_layernorm = nn.LayerNorm(self.query_noise_dim)

        # not an object already included in category_dict
        if cfg.detr.class_project_mlp:
            self.class_project = MLP(
                self.dec_dim,
                cfg.detr.label_pos_head_ffn_dim,  # cfg.detr.decoder_ffn_embed_dim,
                len(cat_dict),
                cfg.detr.bbox_regression_layers,
                cfg.detr.dropout,
            )
        else:
            self.class_project = nn.Linear(self.dec_dim, len(cat_dict), bias=False)

        self.bbox_project = BBoxDecoder(cfg, pos_dict, cat_dict)

        self.img_decoder = DETRBasedDecoder(cfg.detr, encoder_embed_dim=self.layout_enc_hdim)

        # DETR weight init: only Transformer class? (for us: decoder below)
        # for child in (self.query_embed, self.class_embed, self.bbox_embed):
        #     child.apply(init_bert_weights)

    def _get_query_pos_embeddings(self, batch_size):
        if self.learnt_query_embeds:
            return self.learnt_query_embed_dropout(
                self.learnt_query_embed_layernorm(
                    self.learnt_query_embed.weight.unsqueeze(0).repeat(batch_size, 1, 1)
                )
            )
        return None  # torch.zeros((self.num_queries, batch_size, self.query_noise_dim))

    def _get_init_queries(self, batch_size):
        if self.noise_as_queries:
            return self.noise_queries_dropout(
                self.noise_queries_layernorm(
                    self.noise_queries.unsqueeze(0).repeat(batch_size, 1, 1).normal_()
                )
            )
        return torch.zeros((batch_size, self.num_queries, self.query_noise_dim))

    def _input_embeddings(self, bs, type_as_t):
        # num_queries x embed_dim --> num_queries x BS x embed_dim
        query_embed = self._get_query_pos_embeddings(bs).type_as(type_as_t)
        # num_queries x BS x embed_dim
        tgt = self._get_init_queries(bs).type_as(type_as_t)
        if self.query_noise_agg == "sum":
            if self.query_pos_first_layer_only and self.learnt_query_embeds:
                return None, query_embed + tgt
            return query_embed, tgt
        else:
            # asserts already in init
            return None, torch.cat((tgt, query_embed), dim=-1)

    def forward(self, batch, *args, **kwargs):
        """
        def forward(self, src, mask, query_embed, pos_embed):
            # flatten NxCxHxW to HWxNxC
            bs, c, h, w = src.shape
            src = src.flatten(2).permute(2, 0, 1)
            pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
            query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
            mask = mask.flatten(1)

            tgt = torch.zeros_like(query_embed)
            memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
            hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                              pos=pos_embed, query_pos=query_embed)
            return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)
        """
        encoder_out = super().text_encoder_forward(batch)
        attention_mask = encoder_out["encoder_padding_mask"]
        bs = attention_mask.shape[0]

        query_embed, tgt = self._input_embeddings(bs, type_as_t=encoder_out["encoder_out"])

        tgt_key_padding_mask = torch.zeros(bs, self.num_queries).type_as(attention_mask).bool()
        predicted_lengths = encoder_out["predicted_lengths"]
        pos_embed = encoder_out["pos_embed"]
        if self.cfg.model.predict_num_queries:
            # shape BS x max_len --> BS
            # todo sample?
            max_pred_lengths = predicted_lengths.argmax(dim=-1)
            # += 1 because length 0 cannot be predicted
            max_pred_lengths = (max_pred_lengths + 1).clamp(
                max=self.cfg.model.max_target_positions - 1.0
            )
            # BS x num_queries
            # todo >= if possible to have 0 length, > if min length is 1
            tgt_key_padding_mask = torch.arange(self.cfg.detr.num_queries).type_as(
                max_pred_lengths
            ).repeat((bs, 1)) >= max_pred_lengths.unsqueeze(1)

            max_pred_length = max_pred_lengths.max().long()
            query_embed = query_embed[:, :max_pred_length] if query_embed is not None else None
            tgt = tgt[:, :max_pred_length]

            # True will be ignored
            tgt_key_padding_mask = tgt_key_padding_mask[:, :max_pred_length].bool()

        decoder_out = self.img_decoder(
            tgt=tgt,
            # tgt_mask=tgt_mask,
            memory=encoder_out["encoder_out"],
            memory_key_padding_mask=encoder_out["encoder_padding_mask"],
            tgt_key_padding_mask=tgt_key_padding_mask,
            pos=pos_embed,
            query_pos=query_embed,
        )
        # layer_num x L x BS x e --> layer_num x BS x num_obj x e
        # hs = decoder_out.transpose(1, 2)
        hs = decoder_out

        # layer_num x BS x num_obj x num_class
        outputs_class = self.class_project(hs)
        outputs_bbox = self.bbox_project(hs, label_logits=outputs_class)

        if self.cfg.detr.probabilistic_bbox_predictions:
            bbox_logits = outputs_bbox[-1]
            bbox_preds = self.pos_dict.decode_tensor(bbox_logits.argmax(-1))
        else:
            bbox_preds = bbox_logits = outputs_bbox[-1]

        out = {
            **encoder_out,
            "label_logits": outputs_class[-1],
            "bbox_logits": bbox_logits,
            "bbox_preds": bbox_preds,
            "tgt_padding_mask": tgt_key_padding_mask,
            # BS x max_len
            "predicted_lengths": predicted_lengths,
            # 'sent_embed': encoder_out['sent_embed'],
            # 'rest_embed': encoder_out['rest_embed'],
            "text_embed": encoder_out["encoder_out"],
            "text_lens": (~encoder_out["encoder_padding_mask"]).sum(-1),
            "obj_embed": hs[-1],
            "obj_lens": (~tgt_key_padding_mask).sum(-1),
        }
        if self.cfg.detr.aux_loss:
            out["aux_outputs"] = {
                "label_logits": outputs_class[:-1],
                "outputs_bbox": outputs_bbox[:-1],
            }
        return out

    def generate(self, *args, **kwargs):
        return self(*args, **kwargs)


class AutoregressiveGenerationModel(_SuperModel):
    def __init__(
        self,
        cfg: Config,
        tokenizer: Tokenizer,
        cat_dict: CategoryDictionary,
        pos_dict: PositionDictionary,
    ):
        super().__init__(cfg, tokenizer, cat_dict, pos_dict)
        self.input_agg = cfg.detr.label_bbox_embed_aggregate
        self.dec_dim = cfg.detr.decoder_embed_dim
        self.emb_dim = self.dec_dim // 2 if self.input_agg == "concat_half" else self.dec_dim
        self.bbox_gen_strategy = self.cfg.detr.bbox_generation_strategy
        self.bbox_embed_agg = cfg.detr.bbox_embed_aggregate
        self.bbox_padding = (
            self.pos_dict.pad() if cfg.detr.probabilistic_bbox_predictions else cfg.pos_cont_pad_id
        )
        self.bbox_key = "bboxes" if cfg.detr.probabilistic_bbox_predictions else "bboxes_cont"

        self.tgt_token_embed = nn.Embedding(len(cat_dict), self.emb_dim, cat_dict.pad())
        self.token_embed_layernorm = nn.LayerNorm(self.emb_dim)
        self.token_embed_dropout = nn.Dropout(p=cfg.detr.dropout_token_embed)
        nn.init.normal_(self.tgt_token_embed.weight, mean=0.0, std=0.02)

        bbox_embed_weights = None
        if cfg.detr.probabilistic_bbox_predictions:
            self.bbox_embed_dim = (
                self.emb_dim if self.bbox_embed_agg == "sum" else self.emb_dim // 4
            )
            self.embed_bboxes = nn.ModuleList([
                # + 1 for padding
                nn.Embedding(
                    len(pos_dict) + 1, embedding_dim=self.bbox_embed_dim, padding_idx=pos_dict.pad()
                )
                for _ in range(4)
            ])
            for emb in self.embed_bboxes:
                nn.init.normal_(emb.weight, mean=0, std=0.02)

            if self.bbox_embed_agg == "sum":
                bbox_embed_weights = nn.Parameter(
                    # exclude padding index
                    torch.cat([e.weight[:-1] for e in self.embed_bboxes], dim=0)
                )
            else:
                weights = torch.zeros((len(pos_dict) * 4, self.emb_dim)).type_as(
                    self.tgt_token_embed.weight.data
                )
                for i in range(4):
                    weights[
                        len(pos_dict) * i : len(pos_dict) * (i + 1),
                        self.bbox_embed_dim * i : self.bbox_embed_dim * (i + 1),
                    ] = self.embed_bboxes[i].weight[:-1]
                bbox_embed_weights = nn.Parameter(weights)
        else:
            self.bbox_embed_dim = self.emb_dim
            self.embed_bboxes = nn.Linear(4, self.bbox_embed_dim, bias=False)
            nn.init.normal_(self.embed_bboxes.weight, mean=0, std=0.02)

        self.bbox_embed_dropout = nn.Dropout(p=cfg.detr.dropout_bbox_embed)
        self.bbox_embed_layernorm = nn.LayerNorm(self.emb_dim)

        if self.input_agg == "concat_proj":
            self.input_proj = SingleLayerFC(
                2 * self.emb_dim, self.emb_dim, cfg.detr.dropout, fairseq_layernorm=False
            )

        if cfg.detr.class_project_mlp:
            self.class_project = MLP(
                self.dec_dim,
                cfg.detr.label_pos_head_ffn_dim,  # cfg.detr.decoder_ffn_embed_dim,
                len(cat_dict),
                cfg.detr.bbox_regression_layers,
                cfg.detr.dropout,
            )
        else:
            self.class_project = nn.Linear(self.dec_dim, len(cat_dict), bias=False)

        if (
            cfg.detr.tie_label_embed_to_label_output_proj
            and not self.input_agg == "concat_half"
            and not cfg.detr.class_project_mlp
        ):
            # not an object already included in category_dict
            self.class_project.weight = self.tgt_token_embed.weight

        tie_bbox_embed = (
            cfg.detr.tie_bbox_embed_to_bbox_output_proj
        )  # and self.input_agg != 'concat_half'
        self.bbox_project = BBoxDecoder(
            cfg,
            pos_dict,
            cat_dict,
            label_embed_dim=self.emb_dim,
            tie_label_embed=cfg.detr.tie_label_embed_to_bbox_label_embed,
            label_embed_weights=self.tgt_token_embed.weight,
            tie_bbox_embed=tie_bbox_embed,
            bbox_embed_weights=bbox_embed_weights,
        )

        self.img_decoder = DETRBasedDecoder(cfg.detr, self.layout_enc_hdim)

        # DETR weight init: only Transformer class? (for us: decoder below) todo
        # for child in (self.query_embed, self.class_embed, self.bbox_embed):
        #     child.apply(init_bert_weights)

    def forward(self, batch):
        bboxes = batch[self.bbox_key]
        tgt_input_ids = batch["labels"]

        if self.training:
            # last token is never input during training,
            # only trained to predict it from the 2nd but last token
            tgt_input_ids = tgt_input_ids[:, :-1]
            bboxes = bboxes[:, :-1]

        bs, seq_len = tgt_input_ids.shape
        encoder_out = super().text_encoder_forward(batch)

        positions, tgt_input_embs = self._input_embeddings(bboxes, tgt_input_ids)

        # True will be ignored
        tgt_key_padding_mask = (
            batch["tgt_attention_mask"]
            if "tgt_attention_mask" in batch and batch["tgt_attention_mask"] is not None
            else tgt_input_ids.eq(self.category_dict.pad())
        )

        # should be (tgt_len, src_len) but src is tgt here because
        # self-attention so (tgt_len, tgt_len)
        # True will be ignored
        tgt_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).type_as(
            encoder_out["encoder_padding_mask"]
        )

        decoder_out = self.img_decoder(
            tgt=tgt_input_embs,
            tgt_mask=tgt_mask,
            memory=encoder_out["encoder_out"],
            memory_key_padding_mask=encoder_out["encoder_padding_mask"],
            tgt_key_padding_mask=tgt_key_padding_mask,
            query_pos=positions,
        )

        # layer_num x L x BS x e --> layer_num x BS x L x e
        # hs = decoder_out.transpose(1, 2)
        hs = decoder_out

        # layer_num x BS x num_obj x num_class
        outputs_class = self.class_project(hs)
        # outputs_bbox = self.bbox_project(hs, label_logits=outputs_class)

        if self.training:
            # we do the projection in self.generate(...) during non-training,
            # because then the bbox projection
            #   might need the predicted labels
            outputs_bbox = self.bbox_project(hs, label_logits=outputs_class)
            if self.cfg.detr.probabilistic_bbox_predictions:
                bbox_logits = outputs_bbox[-1]
                bbox_preds = self.pos_dict.decode_tensor(bbox_logits.argmax(-1))
            else:
                bbox_preds = bbox_logits = outputs_bbox[-1]
        else:
            bbox_logits, bbox_preds = [None, None], None

        out = {
            **encoder_out,
            "label_logits": outputs_class[-1],
            "bbox_logits": bbox_logits,
            "bbox_preds": bbox_preds,
            "tgt_padding_mask": tgt_key_padding_mask,
            "text_embed": encoder_out["encoder_out"],
            "text_lens": (~encoder_out["encoder_padding_mask"]).sum(-1),
            "obj_embed": hs[-1],  # BS x L x e
            "obj_lens": (~tgt_key_padding_mask).sum(-1),
            # 'sent_embed': encoder_out['sent_embed'],
            # 'rest_embed': encoder_out['rest_embed'],
        }
        if self.cfg.detr.aux_loss:
            # out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
            out["aux_outputs"] = {
                "label_logits": outputs_class[:-1],
                "outputs_bbox": outputs_bbox[:-1],
            }
        return out

    def _input_embeddings(self, bboxes, tgt_input_ids):
        """
        First LN then dropout based on Gorjan and Roberta in huggingface,
            (https://arxiv.org/pdf/1801.05134.pdf but about BN,
            https://arxiv.org/pdf/2002.04745.pdf but about LN before attn in transformers)
        """
        # BS x L --> BS x L x embed --> L x BS x embed_dim
        tgt_input_embs = self.token_embed_dropout(
            self.token_embed_layernorm(
                self.tgt_token_embed(tgt_input_ids.clone())  # .transpose(0, 1)
            )
        )
        # BS x L x embed_dim --> L x BS x embed_dim
        positions = self.bbox_embed_dropout(
            self.bbox_embed_layernorm(
                self._bbox_embeddings(bboxes, strategy=self.bbox_embed_agg)  # .transpose(0, 1)
            )
        )
        if "concat_" in self.input_agg:
            # positions first! because of way weights are set in BBoxMLMHead if tied
            tgt_input_embs = torch.cat((positions, tgt_input_embs), dim=-1)
            positions = None
            if self.input_agg == "concat_proj":
                tgt_input_embs = self.input_proj(tgt_input_embs)
        elif self.input_agg == "sum" and self.cfg.detr.query_pos_first_layer_only:
            tgt_input_embs = tgt_input_embs + positions
            positions = None
        return positions, tgt_input_embs

    def _bbox_embeddings(self, bboxes, strategy="sum"):
        bs, seq_len, num_coord = bboxes.shape
        if self.cfg.detr.probabilistic_bbox_predictions:
            # BS x L x 4 --> 4 * [BS x L x e]
            embs = [emb(bboxes[:, :, i]) for i, emb in zip(range(num_coord), self.embed_bboxes)]
            if strategy == "sum":
                # 4 * [BS x L x e] --> 4 x BS x L x e
                positions = torch.stack(embs, dim=0)
                # ... --> BS x L x e
                positions = positions.sum(dim=0)
            else:
                assert strategy == "concat", strategy
                # 4 * [BS x L x e] --> BS x L x 4 * e
                positions = torch.cat(embs, dim=-1)
        else:
            positions = self.embed_bboxes(bboxes)
        return positions

    def generate(self, batch, strategy="greedy"):
        bs = batch[self.bbox_key].shape[0]
        running_batch = copy.deepcopy(batch)
        running_batch.update({
            "labels": torch.empty(bs, 1).type_as(batch["labels"]).fill_(self.category_dict.bos()),
            self.bbox_key: (
                torch.empty(bs, 1, batch["bboxes"].size(2))
                .type_as(batch[self.bbox_key])
                .fill_(self.bbox_padding)
            ),
            # 'bboxes_cont': (
            #     torch.empty(batch['bboxes_cont'].size(0), 1, batch['bboxes'].size(2))
            #         .type_as(batch['bboxes_cont'])
            #         .fill_(self.cfg.pos_cont_pad_id)
            # ),
        })
        unfinished_sequences = batch["attention_mask"].new(bs).fill_(1).long()

        probs, box_probs, bbox_logits_list = [], [], []
        for i in itertools.count():
            # self.eval()
            assert not self.training

            outputs = self(running_batch)
            # BS x L x num_class --> BS x num_class
            next_token_logits = outputs["label_logits"][:, -1]  # .detach().clone()
            # BS x L x 4 (x num_pos) --> BS x 4 (x num_pos)
            # next_bbox_logits = outputs['pred_boxes'][:, -1]     # .detach().clone()

            cur_length = running_batch["labels"].size(-1)
            next_tokens, next_boxes, next_probs, next_box_probs, bbox_logits = (
                self._get_next_tokens_and_boxes(
                    next_token_logits, outputs["obj_embed"][:, i].unsqueeze(1), strategy, cur_length
                )
            )
            next_tokens = next_tokens * unfinished_sequences + self.category_dict.pad() * (
                1 - unfinished_sequences
            )
            probs.append(next_probs * unfinished_sequences + -1 * (1 - unfinished_sequences))

            padded_next_boxes = next_boxes * unfinished_sequences.unsqueeze(
                -1
            ) + self.bbox_padding * (1 - unfinished_sequences.unsqueeze(-1))
            box_probs.append(
                next_box_probs * unfinished_sequences.unsqueeze(-1)
                + -1 * (1 - unfinished_sequences.unsqueeze(-1))
            )
            bbox_logits_list.append(bbox_logits)  # layer x BS x L x num_coords x num_pos

            new_labels = torch.cat([running_batch["labels"], next_tokens.unsqueeze(-1)], dim=-1)
            new_boxes = torch.cat(
                [running_batch[self.bbox_key], padded_next_boxes.unsqueeze(1)], dim=1
            )

            running_batch = {
                **batch,
                "labels": new_labels,
                self.bbox_key: new_boxes,
            }

            unfinished_sequences = unfinished_sequences.mul(
                (next_tokens != self.category_dict.eos()).long()
            )

            if (
                unfinished_sequences.max() == 0
                or new_labels.size(-1) >= self.cfg.detr.max_num_generated_objects
            ):
                break

        result = {
            **outputs,
            # todo new_labels and new_boxes do include bos, pred_boxes_dec does not
            "gen_labels": new_labels,
            "gen_labels_probs": torch.stack(probs, dim=-1),
            "gen_boxes": new_boxes,
            "gen_boxes_probs": torch.stack(box_probs, dim=1),
            # (1, bs, 1, 4)[] --> (1, bs, L, 4) --> (bs, L, 4)
            "bbox_logits": torch.cat(bbox_logits_list, dim=2)[0],
            "bbox_preds": (
                self.pos_dict.decode_tensor(new_boxes[:, 1:])
                if self.cfg.detr.probabilistic_bbox_predictions
                else new_boxes[:, 1:]
            ),
        }
        return result

    def _get_next_tokens_and_boxes(self, token_logits, hidden_states, strategy, cur_length):
        # TOKENS
        tmp_token_logits = token_logits.detach().clone() / self.cfg.detr.generation_temperature
        tmp_token_logits = self._mask_out_specials(cur_length, tmp_token_logits)

        if strategy == "greedy":
            token_probs = F.softmax(tmp_token_logits, dim=-1)
            next_probs, next_tokens = token_probs.max(-1)
        else:
            assert strategy in ("nucleus", "sample"), strategy
            next_probs, next_tokens, token_probs = self._sample_and_prob(strategy, tmp_token_logits)

        # BBOXES
        bbox_logits = self.bbox_project(
            hidden_states,
            label_logits=token_logits.unsqueeze(1),
            labels=next_tokens.unsqueeze(-1),
            label_probs=token_probs.unsqueeze(1),
        )
        if self.cfg.detr.probabilistic_bbox_predictions:
            bbox_strategy = (
                strategy if self.bbox_gen_strategy == "inherit" else self.bbox_gen_strategy
            )
            # layer (1) x BS x L (1) x 4 x num_pos
            tmp_bbox_logits = (
                bbox_logits.squeeze(0).squeeze(1) / self.cfg.detr.generation_temperature
            )
            bs, num_coords, num_pos = tmp_bbox_logits.shape

            if bbox_strategy == "greedy":
                next_box_probs, next_boxes = F.softmax(tmp_bbox_logits, dim=-1).max(-1)
            else:
                assert bbox_strategy in ("nucleus", "sample")
                tmp_bbox_logits = tmp_bbox_logits.reshape(-1, num_pos)

                next_box_probs, next_boxes, _ = self._sample_and_prob(
                    bbox_strategy, tmp_bbox_logits
                )

                next_boxes = next_boxes.reshape(bs, num_coords)
                next_box_probs = next_box_probs.reshape(bs, num_coords)
        else:
            next_boxes = bbox_logits.squeeze(0).squeeze(1)
            next_box_probs = torch.ones_like(next_boxes).type_as(next_boxes)

        return next_tokens, next_boxes, next_probs, next_box_probs, bbox_logits

    def _mask_out_specials(self, cur_length, tmp_token_logits):
        if cur_length < self.cfg.detr.min_num_generated_objects:
            tmp_token_logits[:, self.category_dict.eos()] = -float("inf")
        tmp_token_logits[
            :,
            [
                self.category_dict.pad(),
                self.category_dict.bos(),
                self.category_dict.unk(),
                self.category_dict.mask(),
            ],
        ] = -float("inf")
        return tmp_token_logits

    def _sample_and_prob(self, strategy, logits):
        if strategy == "nucleus":
            # https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
            top_p_filter = TopPLogitsWarper(top_p=self.cfg.detr.generation_nucleus_sample_top_p)
            logits = top_p_filter(None, logits)

        probs = F.softmax(logits, dim=-1)
        samples = probs.multinomial(1).squeeze(-1)
        sample_probs = probs.gather(dim=-1, index=samples.unsqueeze(-1)).squeeze(-1)

        return sample_probs, samples, probs


class DETRBasedEncoder(nn.Module):
    def __init__(self, cfg: DETRConfig, encoder_embed_dim, return_intermediate=False):
        super().__init__()
        encoder_layer = TransformerEncoderLayer(
            d_model=encoder_embed_dim,
            nhead=cfg.encoder_attention_heads,
            dim_feedforward=cfg.encoder_ffn_embed_dim,
            dropout=cfg.dropout,
            activation=cfg.activation,
            normalize_before=cfg.normalize_prenorm,
            batch_first=True,
        )
        encoder_norm = nn.LayerNorm(encoder_embed_dim) if cfg.normalize_prenorm else None
        self.encoder = TransformerEncoder(
            encoder_layer, cfg.encoder_layers, encoder_norm, return_intermediate
        )

    def forward(self, sequence_states, mask, pos_embed=None):
        return self.encoder(
            sequence_states,
            src_key_padding_mask=~mask.bool(),
            pos=pos_embed,
        )


class DETRBasedDecoder(nn.Module):
    def __init__(self, cfg: DETRConfig, encoder_embed_dim: int):
        super().__init__()
        self.cfg = cfg
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(
                d_model=cfg.decoder_embed_dim,
                nhead=cfg.decoder_attention_heads,
                dim_feedforward=cfg.decoder_ffn_embed_dim,
                dropout=cfg.dropout,
                activation=cfg.activation,
                normalize_before=cfg.normalize_prenorm,
                kdim_enc=encoder_embed_dim,
                vdim_enc=encoder_embed_dim,
                batch_first=True,
            )
            for _ in range(cfg.decoder_layers)
        ])
        self.norm = nn.LayerNorm(cfg.decoder_embed_dim)
        self.return_intermediate = cfg.aux_loss
        # self.query_pos_first_layer_only = cfg.query_pos_first_layer_only

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        output = tgt

        intermediate = []
        for i, layer in enumerate(self.layers):
            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                # memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos,  # if i == 0 or not self.query_pos_first_layer_only else None,
            )
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)
        return output.unsqueeze(0)


class BBoxDecoder(nn.Module):
    def __init__(
        self,
        cfg: Config,
        pos_dict: PositionDictionary,
        cat_dict: CategoryDictionary,
        label_embed_dim: int = -1,
        tie_label_embed: bool = False,
        label_embed_weights: nn.Parameter = None,
        tie_bbox_embed: bool = False,
        bbox_embed_weights: nn.Parameter = None,
    ):
        super().__init__()
        self.condition_on_label = cfg.detr.condition_bbox_prediction_on_label
        self.probabilistic = cfg.detr.probabilistic_bbox_predictions
        self.softmax_embed_training_argmax_inference = (
            cfg.detr.softmax_embed_training_argmax_inference
        )
        self.hard_sigmoid = cfg.detr.bbox_pred_hard_sigmoid

        decoder_dim = cfg.detr.decoder_embed_dim
        input_dim = (
            (decoder_dim + len(cat_dict))
            if self.condition_on_label in ("probs", "logits")
            else decoder_dim
        )

        if self.condition_on_label in ("embed", "softmax_embed"):
            label_embed_dim = label_embed_dim if label_embed_dim > 0 else decoder_dim
            input_dim = label_embed_dim + decoder_dim

            self.label_embed = nn.Embedding(
                num_embeddings=len(cat_dict),
                embedding_dim=label_embed_dim,
                padding_idx=cat_dict.pad(),
            )

            if tie_label_embed:
                assert label_embed_weights is not None
                self.label_embed.weight = label_embed_weights

        if self.probabilistic:
            self.bbox_project = BBoxMLMHead(
                input_dim,
                len(pos_dict),
                dropout=cfg.detr.dropout,
                fairseq_layernorm=False,
                decode_weights=bbox_embed_weights if tie_bbox_embed else None,
            )
        else:
            self.bbox_project = MLP(
                input_dim,
                cfg.detr.label_pos_head_ffn_dim,  # cfg.detr.decoder_ffn_embed_dim,
                4,
                cfg.detr.bbox_regression_layers,
                cfg.detr.dropout,
            )

    def forward(self, hidden_states, label_logits=None, labels=None, label_probs=None):
        if self.condition_on_label is not None:
            # if self.condition_on_label == 'logits' no additional operation needed
            label_states = label_logits

            if self.condition_on_label == "embed" or (
                self.softmax_embed_training_argmax_inference and not self.training
            ):
                # assert labels is None or hidden_states.dim() == 3
                label_states = self.label_embed(
                    label_logits.argmax(-1) if labels is None else labels
                )
            elif self.condition_on_label == "softmax_embed":
                # assert label_probs is None or hidden_states.dim() == 3
                label_states = F.linear(
                    F.softmax(label_logits, dim=-1) if label_probs is None else label_probs,
                    weight=self.label_embed.weight.T,
                )
            elif self.condition_on_label == "probs":
                # assert label_probs is None or hidden_states.dim() == 3
                label_states = (
                    F.softmax(label_logits, dim=-1) if label_probs is None else label_probs
                )

            # hs and logits might be layer x BS x L x emb or BS x L x emb so dim=-1
            # hidden states first, then label states
            input_states = torch.cat((hidden_states, label_states), dim=-1)
        else:
            input_states = hidden_states

        if input_states.dim() == 3:
            input_states = input_states.unsqueeze(0)

        projected = self.bbox_project(input_states)

        if self.probabilistic:
            # 4 (x,y,w,h) x LN x BS x num_obj x num_pos --> LN x BS x num_obj x 4 x num_pos
            return projected.permute(1, 2, 3, 0, 4)
        else:
            # LN x BS x num_obj x 4, comes from MLP instead of BBoxMLMHead
            return (
                nn.functional.hardsigmoid(projected) if self.hard_sigmoid else projected.sigmoid()
            )


class SingleLayerFC(nn.Module):
    """
    class RobertaLMHead(nn.Module):

        def __init__(self, config):
            super().__init__()
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
            self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

            self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            self.bias = nn.Parameter(torch.zeros(config.vocab_size))

            # Need a link between the two variables so that the bias is correctly
            resized with `resize_token_embeddings`
            self.decoder.bias = self.bias

        def forward(self, features, **kwargs):
            x = self.dense(features)
            x = gelu(x)
            x = self.layer_norm(x)

            # project back to size of vocabulary with bias
            x = self.decoder(x)

            return x
    """

    def __init__(self, hidden_size, output_size, dropout, fairseq_layernorm=True, bias=True):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        # fairseq layernorm
        self.layer_norm = (
            BertLayerNorm(hidden_size) if fairseq_layernorm else nn.LayerNorm(hidden_size)
        )
        self.decoder = nn.Linear(hidden_size, output_size, bias=bias)
        # self.dropout = nn.Dropout(p=dropout)      # no dropout based on huggingface bert/roberta
        self.activation = nn.GELU()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)
        # x = self.dropout(x)       # https://arxiv.org/pdf/1801.05134.pdf
        x = self.layer_norm(x)
        x = self.decoder(x)
        return x


class BBoxMLMHead(nn.Module):
    def __init__(
        self,
        hidden_size,
        vocab_size: int,
        dropout,
        fairseq_layernorm=True,
        bias=True,
        decode_weights=None,
    ):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = (
            BertLayerNorm(hidden_size) if fairseq_layernorm else nn.LayerNorm(hidden_size)
        )
        # self.pos_decoders = _get_clones(nn.Linear(hidden_size, vocab_size, bias=False), 4)
        if decode_weights is not None:
            self.pos_decoder = nn.Linear(hidden_size, 4 * vocab_size, bias=False)
            # https://pytorch.org/docs/stable/generated/torch.nn.Linear.html?highlight=linear#torch.nn.Linear
            # first columns of A become first rows of A.T, multiply first columns of row vector x,
            #  which correspond to bbox hidden states, last columns to label states
            self.pos_decoder.weight.data[:, : decode_weights.shape[1]] = decode_weights
        else:
            self.pos_decoder = nn.Linear(
                hidden_size, 4 * vocab_size, bias=bias
            )  # todo weight init zero (huggingface)

        self.size = vocab_size
        # self.dropout = nn.Dropout(p=dropout)
        self.activation = nn.GELU()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)
        x = self.layer_norm(x)
        # x = self.dropout(x)             # https://arxiv.org/pdf/1801.05134.pdf
        # LN x BS x num_obj x num_pos*4
        x = self.pos_decoder(x)
        ln, bs, num_obj, _ = x.shape
        # LN x BS x num_obj x num_pos x 4
        x = x.reshape(ln, bs, num_obj, self.size, 4)
        # 4 x LN x BS x num_obj x num_pos
        return x.permute(4, 0, 1, 2, 3)


class ObjGANGenerationModel(_SuperModel):
    def __init__(
        self,
        cfg: Config,
        tokenizer: Tokenizer,
        cat_dict: CategoryDictionary,
        pos_dict: PositionDictionary,
    ):
        super().__init__(cfg, tokenizer, cat_dict, pos_dict)

        self.dec_dim = cfg.obj_gan.hidden_dim
        # self.gaussian_dict = np.load(cfg.obj_gan.gaussian_dict_path, allow_pickle=True).item()
        # self.dimension_mean_stds = ObjGAN.read_mean_std(cfg.obj_gan.mean_std_path)
        with open(cfg.obj_gan.mean_std_path) as f:
            dimension_mean_stds = json.load(f)

        self.decoder_rnn = DecoderRNN(
            category_dict=cat_dict,
            dimension_means=[dimension_mean_stds[k][0] for k in ("x", "y", "w", "r")],
            max_len=self.cfg.detr.max_num_generated_objects,
            hidden_size=cfg.obj_gan.hidden_dim,
            # enc_out_size=self.text_enc_hdim,
            enc_out_size=self.dec_dim,
            box_hidden_size=cfg.obj_gan.box_hidden_dim,
            gmm_comp_num=cfg.obj_gan.gmm_comp_num,
            n_layers=cfg.obj_gan.n_layers,
            rnn_cell=cfg.obj_gan.rnn_cell,
            bidirectional_encoder=cfg.obj_gan.bidirectional_enc,
            input_dropout_p=cfg.obj_gan.input_dropout_p,
            dropout_p=cfg.obj_gan.dropout_p,
            use_attention=cfg.obj_gan.use_attention,
            temperature=cfg.obj_gan.temperature,
        )
        self.bbox_padding = (
            self.pos_dict.pad() if cfg.detr.probabilistic_bbox_predictions else cfg.pos_cont_pad_id
        )
        self.bbox_key = "bboxes" if cfg.detr.probabilistic_bbox_predictions else "bboxes_cont"
        if self.cfg.text_encoder.text_encoder == "attn_gan":
            numdir = 2 if cfg.text_encoder.attn_gan_text_encoder_bidirectional else 1
            self.enc_hidden_size = cfg.text_encoder.attn_gan_text_encoder_input_dim // numdir
            enc_hidden_to_dec_size = self.dec_dim // numdir
        else:
            self.enc_hidden_size = self.layout_enc_hdim
            enc_hidden_to_dec_size = self.dec_dim
            self.project_hidden_to_lstm_ctx = MLP(
                self.enc_hidden_size,
                2 * self.enc_hidden_size,
                self.enc_hidden_size,
                num_layers=1,
                dropout=cfg.obj_gan.dropout_p,
            )
        self.project_txt_enc_output = self.project_hidden = lambda x: x
        if self.enc_hidden_size != self.dec_dim:
            self.project_hidden = nn.Linear(
                self.enc_hidden_size, enc_hidden_to_dec_size, bias=False
            )
        if self.layout_enc_hdim != self.dec_dim:
            self.project_txt_enc_output = nn.Linear(self.layout_enc_hdim, self.dec_dim)
        self.hidden_norm = nn.LayerNorm(enc_hidden_to_dec_size)
        self.enc_out_norm = nn.LayerNorm(self.dec_dim)

    def forward(self, batch):
        text_encoder_out = super().text_encoder_forward(batch)
        decoder_out, bbox_logits = self.wrapped_forward(batch, text_encoder_out, self.training)

        out = {
            **text_encoder_out,
            **decoder_out,
            "label_logits": decoder_out["label_logits"],
            "bbox_logits": bbox_logits,
            "bbox_preds": None,
            "tgt_padding_mask": decoder_out["label_padding_mask"],
            "text_embed": text_encoder_out["encoder_out"],
            "text_lens": (~text_encoder_out["encoder_padding_mask"]).sum(-1),
            "obj_embed": decoder_out["obj_embed"],
            # 'obj_lens': (~tgt_key_padding_mask).sum(-1),
            "obj_lens": decoder_out["lengths"],
            # 'sent_embed': encoder_out['sent_embed'],
            # 'rest_embed': encoder_out['rest_embed'],
        }
        return out

    def generate(self, batch, *args, **kwargs):
        text_encoder_out = super().text_encoder_forward(batch)

        decoder_out, bbox_logits = self.wrapped_forward(batch, text_encoder_out, self.training)

        bbox_preds = torch.cat((decoder_out["xy"], decoder_out["wh"]), dim=-1)
        result = {
            **text_encoder_out,
            **decoder_out,
            "label_logits": decoder_out["label_logits"],
            "bbox_logits": bbox_logits,
            "bbox_preds": bbox_preds,
            "text_embed": text_encoder_out["encoder_out"],
            "text_lens": (~text_encoder_out["encoder_padding_mask"]).sum(-1),
            "obj_embed": decoder_out["obj_embed"],
            "tgt_padding_mask": decoder_out["label_padding_mask"],
            "obj_lens": decoder_out["lengths"],
            # todo new_labels and new_boxes do include bos, pred_boxes_dec does not
            "gen_labels": decoder_out["labels"],
            "gen_boxes": bbox_preds,
            # 'gen_labels_probs': torch.stack(probs, dim=-1),
            # 'gen_boxes_probs': torch.stack(box_probs, dim=1),
        }
        return result

    def wrapped_forward(self, batch, text_encoder_out, training=True):
        if training:
            bboxes = batch[self.bbox_key]
            bboxes_x, bboxes_y, bboxes_w, bboxes_h = (
                bboxes[:, :, 0],
                bboxes[:, :, 1],
                bboxes[:, :, 2],
                bboxes[:, :, 3],
            )
            tgt_input_ids = batch["labels"]
        else:
            bboxes_x, bboxes_y, bboxes_w, bboxes_h, tgt_input_ids = None, None, None, None, None

        if self.cfg.text_encoder.text_encoder == "attn_gan":
            # encoder_hidden: num_layers*num_directions x batch x hidden_size
            enc_hidden = text_encoder_out["lstm_hidden"]
            bs, hdim = enc_hidden[0].shape[-2:]
            nl, ndir = text_encoder_out["layers_directions"]
            # only use hidden state of last layer
            enc_hidden = tuple([t.view(nl, ndir, bs, hdim)[-1] for t in enc_hidden])
        else:
            enc_hidden = tuple([
                text_encoder_out["sent_embed"].unsqueeze(0),
                self.project_hidden_to_lstm_ctx(text_encoder_out["sent_embed"]).unsqueeze(0),
            ])
            # enc_hidden = self.project_hidden(enc_hidden)
        enc_hidden = tuple([self.hidden_norm(self.project_hidden(h)) for h in enc_hidden])
        enc_out_embs = self.enc_out_norm(
            self.project_txt_enc_output(text_encoder_out["encoder_out"])
        )

        decoder_out = self.decoder_rnn(
            encoder_hidden=enc_hidden,
            encoder_outputs=enc_out_embs,
            target_l_variables=tgt_input_ids,
            target_x_variables=bboxes_x,
            target_y_variables=bboxes_y,
            target_w_variables=bboxes_w,
            target_h_variables=bboxes_h,
            is_training=training,
            early_stop_len=self.cfg.detr.max_num_generated_objects,  # only used when not training
        )

        bbox_logits = (decoder_out["xy_gmm_params"], decoder_out["wh_gmm_params"])
        return decoder_out, bbox_logits
