import logging

import numpy as np
import torch
import torch.nn.functional as F
from config import Config
from data.dictionary import PositionDictionary
from fairseq.data import Dictionary
from model.loss.structure import (
    MultiObjectContrastiveAttnLoss,
    MultiObjectContrastiveMaxLoss,
)
from torch.nn.modules.loss import _Loss

logger = logging.getLogger("pytorch_lightning")


class BBLoss(_Loss):
    def __init__(self, gmm_comp_num, label_pad_id):
        super().__init__()
        # self.batch_size = batch_size
        self.gmm_comp_num = gmm_comp_num
        self.label_pad_id = label_pad_id

        # super(BBLoss, self).__init__(self._NAME, criterion=None)

    def forward(self, xy_gmm_params, wh_gmm_params, gt_x, gt_y, gt_w, gt_h, gt_l):
        # xy_gmm_params: (pi, u_x, u_y, sigma_x, sigma_y, rho_xy)
        # wh_gmm_params: (pi, u_x, u_y, sigma_x, sigma_y, rho_xy)
        # pi: batch x gmm_comp_num
        # u_x: batch x gmm_comp_num
        # u_y: batch x gmm_comp_num
        # sigma_x: batch x gmm_comp_num
        # sigma_y: batch x gmm_comp_num
        # rho_xy: batch x gmm_comp_num
        # gt_x: batch

        # 1. get gmms
        pi_xy, u_x, u_y, sigma_x, sigma_y, rho_xy = xy_gmm_params
        pi_wh, u_w, u_h, sigma_w, sigma_h, rho_wh = wh_gmm_params

        batch_size, gmm_comp_num = pi_xy.size()
        pi_xy, u_x, u_y, sigma_x, sigma_y, rho_xy = (
            pi_xy.contiguous().view(-1),
            u_x.contiguous().view(-1),
            u_y.contiguous().view(-1),
            sigma_x.contiguous().view(-1),
            sigma_y.contiguous().view(-1),
            rho_xy.contiguous().view(-1),
        )
        pi_wh, u_w, u_h, sigma_w, sigma_h, rho_wh = (
            pi_wh.contiguous().view(-1),
            u_w.contiguous().view(-1),
            u_h.contiguous().view(-1),
            sigma_w.contiguous().view(-1),
            sigma_h.contiguous().view(-1),
            rho_wh.contiguous().view(-1),
        )

        # 3. calculate the bbox loss
        mask = (gt_l != self.label_pad_id).float()
        gt_x = gt_x.unsqueeze(1).repeat(1, gmm_comp_num).view(-1)
        gt_y = gt_y.unsqueeze(1).repeat(1, gmm_comp_num).view(-1)
        gt_w = gt_w.unsqueeze(1).repeat(1, gmm_comp_num).view(-1)
        gt_h = gt_h.unsqueeze(1).repeat(1, gmm_comp_num).view(-1)

        xy_pdf = self.pdf(
            pi_xy, gt_x, gt_y, u_x, u_y, sigma_x, sigma_y, rho_xy, batch_size, gmm_comp_num
        )
        wh_pdf = self.pdf(
            pi_wh, gt_w, gt_h, u_w, u_h, sigma_w, sigma_h, rho_wh, batch_size, gmm_comp_num
        )
        bbox_loss = -torch.sum(mask * xy_pdf) - torch.sum(
            mask * wh_pdf
        )  # /(gmm_comp_num*batch_size)
        # print('bbox_loss: ', bbox_loss)

        # self.acc_loss += bbox_loss
        # self.norm_term += torch.sum(mask)
        return bbox_loss  # / torch.sum(mask)

    # def get_loss(self):
    #     return self.lamda*self.acc_loss/self.norm_term

    def pdf(self, pi_xy, x, y, u_x, u_y, sigma_x, sigma_y, rho_xy, batch_size, gmm_comp_num):
        # all inputs have the same shape: batch*gmm_comp_num
        z_x = ((x - u_x) / sigma_x) ** 2
        z_y = ((y - u_y) / sigma_y) ** 2
        z_xy = (x - u_x) * (y - u_y) / (sigma_x * sigma_y)
        z = z_x + z_y - 2 * rho_xy * z_xy
        a = -z / (2 * (1 - rho_xy**2))
        a = a.view(batch_size, gmm_comp_num)
        a_max = torch.max(a, dim=1)[0]
        a_max = a_max.unsqueeze(1).repeat(1, gmm_comp_num)
        a, a_max = a.view(-1), a_max.view(-1)

        exp = torch.exp(a - a_max)
        norm = torch.clamp(2 * np.pi * sigma_x * sigma_y * torch.sqrt(1 - rho_xy**2), min=1e-5)
        raw_pdf = pi_xy * exp / norm
        raw_pdf = raw_pdf.view(batch_size, gmm_comp_num)
        raw_pdf = torch.log(torch.sum(raw_pdf, dim=1) + 1e-5)
        a_max = a_max.view(batch_size, gmm_comp_num)[:, 0]
        raw_pdf = raw_pdf + a_max

        return raw_pdf


class ObjGANCriterion(_Loss):
    def __init__(self, cfg: Config, category_dict: Dictionary, pos_dict: PositionDictionary):
        super().__init__()
        self.cfg = cfg

        self.padding_idx = category_dict.pad()
        self.eos_idx = category_dict.eos()
        self.bos_idx = category_dict.bos()

        self.pos_padding_id = pos_dict.pad()
        self.pos_cont_pad_id = cfg.pos_cont_pad_id
        self.pos_dict = pos_dict
        self.prob_bboxes = cfg.detr.probabilistic_bbox_predictions

        self.eps = cfg.detr.label_smoothing
        self.eps_pos = cfg.detr.label_smoothing_pos

        # self.prob_bbox_loss_wt = cfg.detr.prob_bbox_loss_coef
        # self.weight_prob_bbox_loss_by_dist = cfg.detr.weight_prob_bbox_loss_by_dist
        self.struct_loss_wt = cfg.detr.struct_loss_coef
        self.giou_loss_wt = cfg.detr.giou_loss_coef
        self.bbox_rel_loss_wt = cfg.detr.bbox_rel_loss_coef
        self.bbox_rel_loss_type = cfg.detr.bbox_rel_loss_type

        self.bbox_loss_wt = cfg.obj_gan.bbox_loss_coef
        self.label_loss_wt = cfg.obj_gan.label_loss_coef
        self.bbox_loss = BBLoss(cfg.obj_gan.gmm_comp_num, self.padding_idx)

        if self.struct_loss_wt > 0.0:
            struct_loss_class = {
                "max": MultiObjectContrastiveMaxLoss,
                "attn": MultiObjectContrastiveAttnLoss,
            }[cfg.detr.struct_loss_type]
            self.struct_loss_fn = struct_loss_class(cfg.detr)

    def forward(self, outputs, batch):
        # for step, step_output in enumerate(decoder_outputs):
        #     batch_size = target_l_variables.size(0)
        #
        #     target_l = target_l_variables[:, step + 1]
        #     target_x = target_x_variables[:, step + 1]
        #     target_y = target_y_variables[:, step + 1]
        #     target_w = target_w_variables[:, step + 1]
        #     target_h = target_h_variables[:, step + 1]
        #
        #     lloss.eval_batch(step_output.contiguous().view(batch_size, -1), target_l)
        #     bloss.eval_batch(xy_gmm_params[step], wh_gmm_params[step], target_x,
        #                      target_y, target_w, target_h, target_l)
        # cur_lloss = lloss.get_loss()
        # cur_bloss = bloss.get_loss()
        # loss = cur_lloss + cur_bloss

        # LABEL LOSS
        # bos token is never predicted
        labels = batch["labels"][:, 1:]  # BS x L
        target_bboxes = batch["bboxes"][:, 1:] if self.prob_bboxes else batch["bboxes_cont"][:, 1:]
        label_logits = outputs["label_logits"]  # BS x L x T
        # bbox_logits = outputs['bbox_logits']

        # this should only occur during inference, when model generates less objects than GT
        if labels.shape[1] > label_logits.shape[1]:
            # raise ValueError
            labels = labels[:, : label_logits.shape[1]]
            target_bboxes = target_bboxes[:, : label_logits.shape[1]]
            # target_bboxes_cont = target_bboxes_cont[:, :label_logits.shape[1]]

        # this should only occur during inference.
        # If model generated more objects than there are GT objects
        if labels.shape[1] < label_logits.shape[1]:
            # raise ValueError
            label_logits = label_logits[:, : labels.shape[1]]

        label_non_padding_mask = labels.ne(self.padding_idx)  # BS x L
        # label_non_special_mask = (
        #     labels.ne(self.padding_idx) & labels.ne(self.bos_idx) & labels.ne(self.eos_idx)
        # )
        ntokens = label_non_padding_mask.long().sum()  # 1

        label_logprobs = F.log_softmax(label_logits, dim=-1)  # BS x L x T
        # print(label_logprobs.shape)
        # print(labels.shape)
        label_nll_loss = -label_logprobs.gather(dim=-1, index=labels.unsqueeze(-1))  # BS x L x 1
        label_smoothed_loss = -label_logprobs.sum(dim=-1, keepdim=True)  # BS x L x 1

        label_nll_loss = label_nll_loss[label_non_padding_mask]
        label_smoothed_loss = label_smoothed_loss[label_non_padding_mask]

        label_nll_loss = label_nll_loss.sum() / ntokens
        label_smoothed_loss = label_smoothed_loss.sum() / ntokens
        label_loss = (1 - self.eps) * label_nll_loss + self.eps / label_logprobs.size(
            -1
        ) * label_smoothed_loss

        # BBOX LOSS
        bbox_loss = torch.tensor(0.0).type_as(label_logits)
        for step in range(labels.shape[1]):
            target_l = labels[:, step]
            target_x = target_bboxes[:, step, 0]
            target_y = target_bboxes[:, step, 1]
            target_w = target_bboxes[:, step, 2]
            target_h = target_bboxes[:, step, 3]

            bbox_loss += self.bbox_loss(
                outputs["xy_gmm_params"][step],
                outputs["wh_gmm_params"][step],
                target_x,
                target_y,
                target_w,
                target_h,
                target_l,
            )
        bbox_loss /= ntokens * 4
        loss = self.label_loss_wt * label_loss + self.bbox_loss_wt * bbox_loss

        struct_loss = torch.tensor(0.0).type_as(label_logits)
        if self.cfg.detr.struct_loss_coef > 0.0:
            #  always predict one box that is the entire image
            #  so its representation can be matched
            #   against the sentence embedding?
            # we switch img and text because AttnGAN wants similarity between
            # 1) text-weighted visual embs and
            #   2) text embs, and we want similarity between 1) visual obj embs and
            #   2) vis-obj weighted struct embs
            txt_embs, txt_lens = outputs["text_embed"], outputs["text_lens"]
            if self.cfg.detr.struct_loss_input == "tree_pos_embs":
                txt_embs, txt_lens = outputs["span_tree_embed"], outputs["span_tree_embed_mask"]
                txt_lens = txt_lens.sum(-1)
            struct_loss, _ = self.struct_loss_fn(
                txt_embs=txt_embs,
                txt_lens=txt_lens,
                img_embs=outputs["obj_embed"],
                img_lens=outputs["obj_lens"],
            )
            loss += self.struct_loss_wt * struct_loss

        # this loss is only used during training so OK
        bs = label_logits.shape[0]
        all_objects = label_logits.argmax(-1)
        all_objects = all_objects[all_objects.ne(self.padding_idx) & all_objects.ne(self.eos_idx)]
        avg_num_objects = (all_objects != outputs["label_logits"].shape[-1] - 1).sum().div(bs)
        avg_num_nobj = (all_objects == outputs["label_logits"].shape[-1] - 1).sum().div(bs)

        losses = {
            "loss_ce_nll": label_nll_loss,
            # 'loss_bbox_prob_nll': bbox_nll_loss,
            "loss_ce_smooth": label_smoothed_loss,
            "avg_num_objects": avg_num_objects,
            "avg_num_nobj": avg_num_nobj,
            "loss_ce": label_loss,
            "loss_bbox_prob": bbox_loss,
            "struct_loss": struct_loss,
            "loss": loss,
        }
        return losses
