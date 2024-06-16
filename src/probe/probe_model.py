import collections

import torch.nn.functional
from config import Config
from torch import nn

from probe.probe_utils import ConstituentDictionary


class ProbeModel(nn.Module):
    def __init__(self, cfg: Config, tag_dict: ConstituentDictionary, embedding_dim: int):
        """
        CLF head used in Tenney et al.:
        https://github.com/nyu-mll/jiant-v1-legacy/blob/8be1fade2339ec29a56d554c8e88dbd999f787a3/jiant/modules/simple_modules.py#L49
        """
        super().__init__()
        self.cfg = cfg
        # self.h_dim = cfg.model.encoder_embed_dim
        self.h_dim = embedding_dim

        self.self_attn_pool = SelfAttnPool(cfg, embedding_dim)

        self.mlp = nn.Sequential(
            collections.OrderedDict([
                ("linear1", nn.Linear(self.h_dim, self.h_dim)),
                ("relu", nn.LeakyReLU()),
                ("ln1", nn.LayerNorm(self.h_dim)),
                ("dropout", nn.Dropout(p=cfg.probe.dropout)),
                ("linear2", nn.Linear(self.h_dim, len(tag_dict))),
            ])
        )

    def forward(self, batch):
        # token_embs: bs x L x N x h, spans: bs x S x 2
        # print('ok')
        token_embs = batch["embeddings"]
        # bs, n_layers, n_tokens, h_dim = token_embs.shape

        # if self.cfg.probe.embedding_model == 'TG':
        #     # first mask out constituency tree tokens like (, ), and tags like NP, VP
        #     print('ok')
        #     m = ~batch['constituent_mask'][:, None, :, None].repeat((1, n_layers, 1, 1))
        #     token_embs = token_embs.masked_fill(mask=m, value=0.0)
        span_embs = self.self_attn_pool(
            token_embs, batch["constituent_mask"], batch["spans"]
        )  # bs x S_tot x h
        scores = self.mlp(span_embs)
        return scores


class SelfAttnPool(nn.Module):
    def __init__(self, cfg: Config, embedding_dim: int):
        super().__init__()
        self.cfg = cfg
        # self.h_dim = cfg.model.encoder_embed_dim
        self.h_dim = embedding_dim
        self.weight = nn.Linear(in_features=self.h_dim, out_features=1, bias=False)

    def forward(self, token_embs, constituent_mask, span_idcs):
        """
        Based on https://arxiv.org/pdf/1905.06316.pdf page 14

        params: token_embs: bs x L x N x h, spans: bs x S x 2
        return: span_embs: bx x sum_S x h
        """
        # bs, n_layers, n_tokens, h_dim = token_embs.shape
        bs, n_tokens, h_dim = token_embs.shape

        z = self.weight(token_embs).squeeze(-1)  # bs x N

        if self.cfg.use_tg and not self.cfg.probe.tg_include_parens:
            # first mask out constituency tree tokens like (, ), and tags like NP, VP
            # m = ~constituent_mask[:, None, :].repeat((1, n_layers, 1))
            m = ~constituent_mask
            z = z.masked_fill(mask=m, value=-float("inf"))

        a = nn.functional.softmax(z, dim=1)  # bs x L x N x 1

        # loop over samples in batch
        span_embs = []
        for span_list, weights, t_embs in zip(span_idcs, a, token_embs):
            for left, r in span_list:  # todo spans can be in [l, r] matrix!
                # weighted sum of token embeddings in span
                t = t_embs[left:r]
                w = weights[left:r]
                sp_emb = w.matmul(t)
                span_embs.append(sp_emb)
        span_embs = torch.stack(span_embs)

        return span_embs


class ProbeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # weights = todo inverse frequency weights
        self.loss = nn.CrossEntropyLoss()

    def forward(self, batch, scores):
        tags = torch.cat([torch.tensor(t) for t in batch["tags"]]).type_as(scores).long()
        return {
            "loss": self.loss(scores, tags),
            "tags": tags,
        }
