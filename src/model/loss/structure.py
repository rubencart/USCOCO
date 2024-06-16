import torch


def cosine_sim(im, s):
    return im.mm(s.t())


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, opt):
        super(ContrastiveLoss, self).__init__()
        self.min_val = 1e-8
        self.margin = opt.struct_loss_margin
        self.sim = cosine_sim

    def forward(self, img_embs, txt_embs, *args):
        # img: b x f or b x bxs x f, b x f
        scores = self.sim(img_embs, txt_embs)  # b x b
        return self.get_loss_for_scores(scores), None

    def get_loss_for_scores(self, scores):
        diagonal = scores.diag().view(scores.size(0), 1)  # b x 1
        d1 = diagonal.expand_as(scores)  # b x b
        d2 = diagonal.t().expand_as(scores)  # b x b
        # these are the formula (2) terms: scores are negative pairs, d1/d2 positive pairs
        loss_txt = (self.margin + scores - d1).clamp(min=self.min_val)  # b x b
        loss_img = (self.margin + scores - d2).clamp(min=self.min_val)  # b x b
        I_diag = (torch.eye(scores.size(0)) > 0.5).type_as(scores).bool()  # b x b, True on diag
        loss_txt = loss_txt.masked_fill_(I_diag, 0)  # fill diagonal with 0
        loss_img = loss_img.masked_fill_(I_diag, 0)
        m_loss_txt = loss_txt.mean(1)
        m_loss_img = loss_img.mean(0)
        loss = m_loss_txt + m_loss_img
        return loss


class MultiObjectContrastiveMaxLoss(ContrastiveLoss):
    def __init__(self, opt):
        super().__init__(opt)

    def max_sim(self, img, txt, img_len, txt_len):
        bs, tl, emb_dim = txt.shape
        _, il, _ = img.shape

        normalized_txt = torch.nn.functional.normalize(txt, dim=-1)
        normalized_img = torch.nn.functional.normalize(img, dim=-1)

        # b x tl
        txt_mask = torch.arange(tl).repeat((bs, 1)).type_as(txt_len) >= txt_len.unsqueeze(-1)
        # b x il
        img_mask = torch.arange(il).repeat((bs, 1)).type_as(txt_len) >= img_len.unsqueeze(-1)
        img_mask2 = (
            img_mask.unsqueeze(-2).repeat((1, tl, 1)).unsqueeze(0).repeat((bs, 1, 1, 1)).bool()
        )
        txt_mask2 = (
            txt_mask.unsqueeze(-1)
            .repeat((1, 1, il))
            .repeat_interleave(bs, 0)
            .view(bs, bs, tl, il)
            .bool()
        )

        img2 = normalized_img.reshape(-1, emb_dim)
        txt2 = normalized_txt.reshape(-1, emb_dim)
        sim_scores2 = torch.matmul(txt2, img2.transpose(-1, -2))
        # b,b,tl,il: (i,j) = sent i, img j
        sim_scores2 = sim_scores2.view(bs, tl, bs, il).permute(0, 2, 1, 3)
        sim_scores2.masked_fill_(txt_mask2 | img_mask2, -float("inf"))

        sim_scores2 = sim_scores2.exp()
        norm_denom2 = sim_scores2.sum(dim=-2)
        sim_scores2 = sim_scores2 / (norm_denom2.unsqueeze(-2) + self.min_val)  # eq. (8)

        # take max per object over txt words
        values, indices = sim_scores2.max(dim=-2)  # b, b, il
        # indices = indices.diagonal(dim1=0, dim2=1).T
        # sum over objects
        values = values.sum(-1)
        return values, sim_scores2

    def forward(self, img_embs, txt_embs, img_lens, txt_lens):
        if img_embs.shape[-1] != txt_embs.shape[-1]:
            raise ValueError
        # img: b x bxs x f, txt: b x L xf
        scores, sim_scores2 = self.max_sim(img_embs, txt_embs, img_lens, txt_lens)  # b x b, b x b
        loss = self.get_loss_for_scores(scores)
        loss = loss.mean()
        return loss, sim_scores2


class MultiObjectContrastiveAttnLoss(torch.nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.word_loss = DAMSMWordLoss(
            opt.struct_loss_margin,
            opt.struct_loss_gamma1,
            opt.struct_loss_gamma2,
            opt.struct_loss_gamma3,
            # opt.loss_word_wt, opt.loss_span_wt,
            opt.struct_loss_normalize_sim_scores_across_spans,
        )
        self.sent_loss = DAMSMSentLoss(opt.struct_loss_margin, opt.struct_loss_gamma3)

    def forward(
        self, img_embs, txt_embs, img_lens, txt_lens
    ):  # , span_mask, span_margs, word_mask):
        if img_embs.shape[-1] != txt_embs.shape[-1]:
            raise ValueError

        # switch
        img_embs, txt_embs, img_lens, txt_lens = txt_embs, img_embs, txt_lens, img_lens
        word_loss, alphas2 = self.word_loss(
            txt_embs,
            img_embs,
            txt_lens,
            img_lens,  # span_margs, span_mask, word_mask
        )
        return word_loss.mean(), alphas2


class DAMSMSentLoss(torch.nn.Module):
    def __init__(self, margin, gamma3):
        super().__init__()
        self.min_val = 1e-6
        self.margin = margin
        self.gamma_3 = gamma3

    def forward(self, txt, img, *args):
        """
        :param txt: shape BS x emb_dim
        :param img: shape BS x emb_dim
        :param args:
        :return: single elem tensor
        """
        normalized_txt = torch.nn.functional.normalize(txt, dim=-1)
        normalized_img = torch.nn.functional.normalize(img, dim=-1)
        scores = torch.mm(normalized_img, normalized_txt.T)  # b x b, rows: imgs, cols: txt

        scores = scores.mul(self.gamma_3).exp()
        diagonal = scores.diag()  # b

        loss_txt = diagonal / (
            scores.sum(dim=0) + self.min_val
        )  # txt fixed, sum over different imgs
        loss_img = diagonal / (
            scores.sum(dim=1) + self.min_val
        )  # img fixed, sum over different txts
        loss_txt = -loss_txt.log()  # .mean()
        loss_img = -loss_img.log()  # .mean()
        loss = loss_txt + loss_img
        return loss


class DAMSMWordLoss(torch.nn.Module):
    def __init__(
        self,
        margin,
        gamma1,
        gamma2,
        gamma3,
        # word_wt, span_wt,
        normalize_sim_scores_across_spans,
    ):
        super().__init__()
        self.min_val = 1e-6
        self.margin = margin
        self.gamma_1 = gamma1
        self.gamma_2 = gamma2
        self.gamma_3 = gamma3
        # self.word_wt = word_wt
        # self.span_wt = span_wt
        self.normalize_sim_scores_across_spans = normalize_sim_scores_across_spans

    def forward(self, txt, img, txt_len, img_len):  # , span_margs, span_mask, word_mask):
        """
        :param txt: shape BS x Lsp x emb_dim
        :param img: shape BS x L x emb_dim
        :param img_len: shape BS
        :param txt_len: shape BS
        # :param span_mask: shape BS x Lsp
        # :param span_margs: shape BS x Lsp
        :return:

        All eq. references are to AttnGAN paper, unless stated otherwise
        """
        bs, tl, emb_dim = txt.shape
        _, il, _ = img.shape

        normalized_txt = torch.nn.functional.normalize(txt, dim=-1)
        normalized_img = torch.nn.functional.normalize(img, dim=-1)

        txt_mask = torch.arange(tl).repeat((bs, 1)).type_as(txt_len) >= txt_len.unsqueeze(
            -1
        )  # b x tl
        img_mask = torch.arange(il).repeat((bs, 1)).type_as(txt_len) >= img_len.unsqueeze(
            -1
        )  # b x il
        img_mask2 = (
            img_mask.unsqueeze(-2).repeat((1, tl, 1)).unsqueeze(0).repeat((bs, 1, 1, 1)).bool()
        )
        txt_mask2 = (
            txt_mask.unsqueeze(-1)
            .repeat((1, 1, il))
            .repeat_interleave(bs, 0)
            .view(bs, bs, tl, il)
            .bool()
        )

        img2 = normalized_img.reshape(-1, emb_dim)
        txt2 = normalized_txt.reshape(-1, emb_dim)
        sim_scores2 = torch.matmul(txt2, img2.transpose(-1, -2))
        sim_scores2 = sim_scores2.view(bs, tl, bs, il).permute(
            0, 2, 1, 3
        )  # b,b,tl,il: (i,j) = sent i, img j
        sim_scores2.masked_fill_(txt_mask2 | img_mask2, -float("inf"))

        if self.normalize_sim_scores_across_spans:
            # one image part gets 1.0 and divides this over text spans
            sim_scores2 = sim_scores2.exp()
            norm_denom2 = sim_scores2.sum(dim=-2)
            sim_scores2 = sim_scores2 / (norm_denom2.unsqueeze(-2) + self.min_val)  # eq. (8)

        alphas2 = (
            sim_scores2.masked_fill_(txt_mask2 | img_mask2, -float("inf")).mul(self.gamma_1).exp()
        )
        alphas2 = alphas2 / (alphas2.sum(-1).unsqueeze(-1) + self.min_val)
        max_obj_scores, max_obj_idcs = alphas2.max(dim=-1)
        max_obj_idcs = max_obj_idcs.diagonal(dim1=0, dim2=1).T
        contexts2 = alphas2.matmul(normalized_img)  # b,b,tl,emb: (i,j,k) = sent i, img j, word k

        normalized_txt2 = normalized_txt.repeat_interleave(bs, dim=0).view(bs, bs, tl, -1)
        normalized_contexts2 = torch.nn.functional.normalize(contexts2, dim=-1)
        # eq. (11): we assume that when computing R(Q_i, D_j),
        # with i =/= j, so c_i en e_j come from a different
        # image/sentence pair, that c_i is computed with the
        # visual embs from Q_i but the word embeddings from D_j
        # and not the word embeddings from D_i.
        # Otherwise, normalized_txt2 should be computed as
        # normalized_txt2 = normalized_txt.unsqueeze(0).repeat((bs, 1, 1, 1))
        # b,b,tl,tl : (i,j,k,l) = sent i, img j, word k in sent i,
        # context l of img j computed with words of sent i
        single_rs2 = torch.matmul(normalized_txt2, normalized_contexts2.transpose(-1, -2))

        rs_txt_mask = (
            txt_mask.unsqueeze(-1).repeat((1, 1, tl)).repeat_interleave(bs, 0).view(bs, bs, tl, tl)
        )
        rs_context_mask = (
            txt_mask.unsqueeze(-1).repeat((1, tl, 1)).repeat_interleave(bs, 0).view(bs, bs, tl, tl)
        )
        single_rs2.masked_fill_(rs_txt_mask | rs_context_mask, -float("inf"))

        single_rs2 = single_rs2.diagonal(dim1=-1, dim2=-2)  # b,b,tl
        single_rs2 = single_rs2.mul(self.gamma_2).exp()  # eq. (10) | b,b,tl

        # weigh spans with their marginal, like eq. (6) in Titov
        # repeat_interleave because first dimension corresponds to sent i, second to img j
        # span_mask_rep = span_mask.repeat_interleave(bs, dim=0).view(bs, bs, -1)
        # span_margs_rep = span_margs.repeat_interleave(bs, dim=0).view(bs, bs, -1)
        # span_margs_mask = torch.arange(span_margs.shape[1]).repeat((bs, 1)).type_as(txt_len)
        #   < span_mask.sum(-1).unsqueeze(1)
        # span_margs_mask = span_margs_mask.repeat_interleave(bs, dim=0).view(bs, bs, -1)
        # wt_single_rs2 = single_rs2.clone()
        # wt_single_rs2[span_mask_rep] = wt_single_rs2[span_mask_rep]
        #   * span_margs_rep[span_margs_mask].view(-1)
        # wt_single_rs2[span_mask_rep] = wt_single_rs2[span_mask_rep] * self.span_wt
        # word_mask_rep = word_mask.repeat_interleave(bs, dim=0).view(bs, bs, -1)
        # wt_single_rs2[word_mask_rep] = wt_single_rs2[word_mask_rep] * self.word_wt

        sample_rs2 = single_rs2.sum(dim=-1).pow(1 / self.gamma_2).log()  # b,b
        prob_num = sample_rs2.mul(self.gamma_3).exp()  # eq. (11)
        loss_txt = prob_num / (prob_num.sum(0) + self.min_val)
        loss_img = prob_num / (prob_num.sum(-1).unsqueeze(-1) + self.min_val)
        loss_txt = -loss_txt.log().diagonal()  # .mean()
        loss_img = -loss_img.log().diagonal()  # .mean()
        return loss_txt + loss_img, alphas2.diagonal(dim1=0, dim2=1).T
