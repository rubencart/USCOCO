import collections

import numpy as np
import torch
import torch.distributions as distributions
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import MultivariateNormal

from data.dictionary import CategoryDictionary


def coord_converter(bbox_preds, dimension_mean_stds):
    # # params: coord_seq, mean_x, std_x, mean_y, std_y
    # coord_x_seq, coord_y_seq = [], []
    # for i in range(len(coord_seq)):
    #     x, y = coord_seq[i]
    #     coord_x_seq.append(x * std_x + mean_x)
    #     coord_y_seq.append(y * std_y + mean_y)
    #
    # return np.array(coord_x_seq), np.array(coord_y_seq)
    bbox_preds[:, :, 0] = (
        bbox_preds[:, :, 0] * dimension_mean_stds["x"][1] + dimension_mean_stds["x"][0]
    )
    bbox_preds[:, :, 1] = (
        bbox_preds[:, :, 1] * dimension_mean_stds["y"][1] + dimension_mean_stds["y"][0]
    )
    bbox_preds[:, :, 2] = (
        bbox_preds[:, :, 2] * dimension_mean_stds["w"][1] + dimension_mean_stds["w"][0]
    )
    bbox_preds[:, :, 3] = (
        bbox_preds[:, :, 3] * dimension_mean_stds["r"][1] + dimension_mean_stds["r"][0]
    )
    bbox_preds[:, :, 3] *= bbox_preds[:, :, 2]
    return bbox_preds


def validity_indices(labels, bboxes, scores):
    # # old params: x_seq, y_seq, w_seq, h_seq, l_seq
    # x_valid_indices = x_seq > 0
    # y_valid_indices = y_seq > 0
    # w_valid_indices = w_seq > 0
    # h_valid_indices = h_seq > 0
    # valid_indices = np.multiply(np.multiply(np.multiply(x_valid_indices, y_valid_indices), w_valid_indices),
    #                             h_valid_indices)
    # x_seq = x_seq[valid_indices]
    # y_seq = y_seq[valid_indices]
    # w_seq = w_seq[valid_indices]
    # h_seq = h_seq[valid_indices]
    # l_seq = l_seq[valid_indices]
    #
    # return x_seq, y_seq, w_seq, h_seq, l_seq
    # bs = labels.shape[0]
    x_seq, y_seq, w_seq, h_seq = bboxes.unbind(dim=2)
    mask = (x_seq > 0) & (y_seq > 0) & (w_seq > 0) & (h_seq > 0)
    # n_x_seq, n_y_seq, n_w_seq, n_h_seq, new_labels = x_seq[mask], y_seq[mask], w_seq[mask], h_seq[mask], labels[mask]
    # scores = scores[mask]
    # new_bboxes = torch.stack((n_x_seq, n_y_seq, n_w_seq, n_h_seq), dim=1).unsqueeze(0).type_as(bboxes)
    # return new_labels.unsqueeze(0), new_bboxes, scores.unsqueeze(0)
    return mask


def read_mean_std(filename):
    with open(filename) as f:
        lines = open(filename).read().strip().split("\n")
    x_mean, x_std = [float(val) for val in lines[0].split(" ")]
    y_mean, y_std = [float(val) for val in lines[1].split(" ")]
    w_mean, w_std = [float(val) for val in lines[2].split(" ")]
    r_mean, r_std = [float(val) for val in lines[3].split(" ")]

    return {
        "x": (x_mean, x_std),
        "y": (y_mean, y_std),
        "w": (w_mean, w_std),
        "r": (r_mean, r_std),
    }


def post_process(labels, boxes, cat_dict: CategoryDictionary, gaussian_dict):
    # out_coco, out_labels, out_boxes = [], [], []
    # for boxes, labels in zip(in_boxes, in_labels):
    # xs, ys, ws, hs = boxes.unbind(dim=1)
    # ls = labels
    masks = []
    for ls in labels:
        if len(ls) > 1:
            coco_labels = cat_dict.convert_cat_labels_to_coco_ids(
                ls, map_special_to=True, special=-1
            )
            # xs = xs[:-1]
            # ys = ys[:-1]
            # ws = ws[:-1]
            # hs = hs[:-1]
            # ls = ls[:-1]
            # coco_labels = coco_labels[:-1]
            # ls = [int(self.category_dict.index2word[int(l)]) for l in ls]
            # coco_labels = cat_dict.convert_cat_labels_to_coco_ids(ls)
            coco_labels = np.array(coco_labels)

            # filter redundant labels
            counter = collections.Counter(coco_labels)
            unique_labels, label_counts = list(counter.keys()), list(counter.values())
            kept_indices = []
            for label_index in range(len(unique_labels)):
                label = unique_labels[label_index]
                label = int(label)
                if label in (-1, cat_dict.nobj_coco_id):
                    continue
                label_num = label_counts[label_index]
                # sample an upper-bound threshold for this label
                mu, sigma = gaussian_dict[label]
                threshold = max(int(np.random.normal(mu, sigma, 1)), 2)
                old_indices = np.where(coco_labels == label)[0].tolist()
                new_indices = old_indices
                if threshold < len(old_indices):
                    new_indices = old_indices[:threshold]
                kept_indices += new_indices

            kept_indices.sort()
            # xs = xs[kept_indices]
            # ys = ys[kept_indices]
            # ws = ws[kept_indices]
            # hs = hs[kept_indices]
            # coco_labels = coco_labels[kept_indices]
            # ls = ls[kept_indices]
            # # ls = [str(l) for l in ls]
            mask = torch.zeros_like(ls).type_as(ls)
            mask[kept_indices] = 1

            # xs = xs - ws / 2.0
            # # xs = np.clip(xs, 1, self.std_img_size - 1)
            # xs = np.clip(xs, 0, 1)
            # ys = ys - hs / 2.0
            # ys = np.clip(ys, 0, 1)
            # # ys = np.clip(ys, 1, self.std_img_size - 1)
            # # ws = np.minimum(ws, self.std_img_size - xs)
            # # hs = np.minimum(hs, self.std_img_size - ys)
            # ws = np.minimum(ws, 1 - xs)
            # hs = np.minimum(hs, 1 - ys)
        else:
            mask = torch.ones_like(ls)
        masks.append(mask)
    # new_bboxes = torch.stack((xs, ys, ws, hs), dim=1)
    # out_coco.append(coco_labels)
    # out_labels.append(ls)
    # out_boxes.append(new_bboxes)
    # return out_coco, out_labels, out_boxes
    # return coco_labels, ls, new_bboxes
    return torch.stack(masks, dim=0)


class Attention(nn.Module):
    def __init__(self, method, hidden_size, enc_out_size):
        super(Attention, self).__init__()

        self.method = method
        self.hidden_size = hidden_size

        if self.method == "general":
            self.attn = nn.Linear(enc_out_size, hidden_size)

        elif self.method == "concat":
            self.attn = nn.Linear(hidden_size + enc_out_size, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size).normal_())

    def forward(self, hidden, encoder_outputs):
        bs = encoder_outputs.size(0)
        max_len = encoder_outputs.size(1)

        # # Create variable to store attention energies
        # # attn_energies = torch.zeros(this_batch_size, max_len, requires_grad=True).type_as(encoder_outputs)  # B x S
        # attn_energies = torch.zeros(bs, max_len).type_as(encoder_outputs)  # B x S
        #
        # # if torch.cuda.is_available():
        # #     attn_energies = attn_energies.cuda()
        #
        # # For each batch of encoder outputs
        # for b in range(bs):
        #     # Calculate energy for each encoder output
        #     for i in range(max_len):
        #         attn_energies[b, i] = self.score(hidden[:, b], encoder_outputs[b, i].unsqueeze(0))
        #         # attn_energies[b, i] =
        attn_energies = self.attn(
            torch.cat((hidden.permute(1, 0, 2).repeat(1, max_len, 1), encoder_outputs), dim=-1)
        )
        attn_energies = attn_energies.matmul(self.v)

        # Normalize energies to weights in range 0 to 1, resize to 1 x B x S
        return F.softmax(attn_energies, dim=-1).unsqueeze(1)

    def score(self, hidden, encoder_output):

        if self.method == "dot":
            energy = hidden.dot(encoder_output)
            return energy

        elif self.method == "general":
            energy = self.attn(encoder_output)
            energy = hidden.dot(energy)
            return energy

        elif self.method == "concat":
            energy = self.attn(torch.cat((hidden, encoder_output), 1))
            energy = energy.squeeze(0)
            energy = self.v.dot(energy)
            return energy


class BaseRNN(nn.Module):
    r"""
    Applies a multi-layer RNN to an input sequence.
    Note:
        Do not use this class directly, use one of the sub classes.
    Args:
        vocab_size (int): size of the vocabulary
        max_len (int): maximum allowed length for the sequence to be processed
        hidden_size (int): number of features in the hidden state `h`
        input_dropout_p (float): dropout probability for the input sequence
        dropout_p (float): dropout probability for the output sequence
        n_layers (int): number of recurrent layers
        rnn_cell (str): type of RNN cell (Eg. 'LSTM' , 'GRU')

    Inputs: ``*args``, ``**kwargs``
        - ``*args``: variable length argument list.
        - ``**kwargs``: arbitrary keyword arguments.

    Attributes:
        SYM_MASK: masking symbol
        SYM_EOS: end-of-sequence symbol
    """

    SYM_MASK = "MASK"
    SYM_EOS = "EOS"

    def __init__(
        self, vocab_size, max_len, hidden_size, input_dropout_p, dropout_p, n_layers, rnn_cell
    ):
        super(BaseRNN, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.input_dropout_p = input_dropout_p
        self.input_dropout = nn.Dropout(p=input_dropout_p)
        if rnn_cell.lower() == "lstm":
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == "gru":
            self.rnn_cell = nn.GRU
        else:
            raise ValueError("Unsupported RNN Cell: {0}".format(rnn_cell))

        self.dropout_p = dropout_p

    def forward(self, *args, **kwargs):
        raise NotImplementedError()


class DecoderRNN(BaseRNN):
    KEY_ATTN_SCORE = "attention_score"
    # KEY_LENGTH = 'length'
    # KEY_SEQUENCE = 'sequence'
    # KEY_XYS = 'xy'
    # KEY_WHS = 'wh'

    def __init__(
        self,
        category_dict: CategoryDictionary,
        dimension_means,
        # batch_size,
        max_len,  # 150 --> chars?
        hidden_size=128,  # 128, word 300
        enc_out_size=128,
        box_hidden_size=50,
        gmm_comp_num=5,  # 5
        n_layers=1,
        rnn_cell="lstm",
        bidirectional_encoder=True,
        input_dropout_p=0.5,
        dropout_p=0.5,  # 0.5
        use_attention=False,
        temperature=0.4,
    ):
        super(DecoderRNN, self).__init__(
            len(category_dict), max_len, hidden_size, input_dropout_p, dropout_p, n_layers, rnn_cell
        )

        self.bidirectional_encoder = bidirectional_encoder
        self.l_output_size = len(category_dict)
        self.aug_size = box_hidden_size
        self.enc_out_size = enc_out_size

        # self.batch_size = batch_size
        self.max_length = max_len
        # gmm_comp_num: number of gaussian mixture components
        self.gmm_comp_num = gmm_comp_num
        self.gmm_param_num = 6  # pi, u_x, u_y, sigma_x, sigma_y, rho_xy
        # self.clip_val = 0.01
        # self.std_norm = distributions.Normal(torch.Tensor([[0.0] * 2] * self.batch_size),
        #                                      torch.Tensor([[1.0] * 2] * self.batch_size))
        self.use_attention = use_attention
        self.l_eos_id = category_dict.eos()
        self.l_sos_id = category_dict.bos()
        self.cat_dict = category_dict
        self.x_mean, self.y_mean, self.w_mean, self.r_mean = dimension_means

        self.temperature = temperature

        self.init_input = None
        self.l_embedding = nn.Embedding(self.l_output_size, self.hidden_size)
        self.xy_embedding = nn.Linear(2, self.aug_size)
        self.xy_input_dropout = nn.Dropout(p=self.input_dropout_p)
        self.wh_embedding = nn.Linear(2, self.aug_size)
        self.wh_input_dropout = nn.Dropout(p=self.input_dropout_p)
        self.next_xy_embedding = nn.Linear(2, self.aug_size)
        self.next_xy_input_dropout = nn.Dropout(p=self.input_dropout_p)
        if use_attention:
            self.attention = Attention("concat", self.hidden_size, self.enc_out_size)
            # rnn inputs: input_size (l_output_size+4), hidden_size, num_layers
            self.rnn = self.rnn_cell(
                2 * hidden_size + 2 * self.aug_size,
                hidden_size,
                n_layers,
                batch_first=True,
                dropout=dropout_p,
            )
        else:
            # rnn inputs: input_size (l_output_size+4), hidden_size, num_layers
            self.rnn = self.rnn_cell(
                hidden_size + 2 * self.aug_size,
                hidden_size,
                n_layers,
                batch_first=True,
                dropout=dropout_p,
            )

        self.l_softmax = F.log_softmax
        self.l_out = nn.Linear(self.hidden_size, self.l_output_size)
        # 1*gmm_comp_num for pi, 2*gmm_comp_num for u, 3*gmm_comp_num for lower triangular matrix
        self.xy_out = nn.Linear(
            self.hidden_size + self.l_output_size, self.gmm_comp_num * self.gmm_param_num
        )
        self.wh_out = nn.Linear(
            self.hidden_size + self.l_output_size + self.aug_size,
            self.gmm_comp_num * self.gmm_param_num,
        )

    def forward_step(
        self,
        l_decoder_input,
        x_decoder_input,
        y_decoder_input,
        w_decoder_input,
        h_decoder_input,
        hidden,
        encoder_outputs,
        next_l_decoder_input=None,
        next_x_decoder_input=None,
        next_y_decoder_input=None,
        is_training=0,
    ):
        bs = l_decoder_input.size(0)

        ### 1. get the RNN input ###
        # l_decoder_input: batch x output_size (1)
        l_decoder_input = l_decoder_input.unsqueeze(1)
        x_decoder_input = x_decoder_input.unsqueeze(1)
        y_decoder_input = y_decoder_input.unsqueeze(1)
        w_decoder_input = w_decoder_input.unsqueeze(1)
        h_decoder_input = h_decoder_input.unsqueeze(1)

        output_size = 1
        l_decoder_input_emb = self.l_embedding(l_decoder_input)
        l_decoder_input_emb = self.input_dropout(l_decoder_input_emb)

        xy_decoder_input = self.xy_embedding(torch.cat((x_decoder_input, y_decoder_input), dim=1))
        xy_decoder_input = self.xy_input_dropout(xy_decoder_input)
        xy_decoder_input = xy_decoder_input.unsqueeze(1)
        wh_decoder_input = self.wh_embedding(torch.cat((w_decoder_input, h_decoder_input), dim=1))
        wh_decoder_input = self.wh_input_dropout(wh_decoder_input)
        wh_decoder_input = wh_decoder_input.unsqueeze(1)

        attn = None
        if self.use_attention:
            # encoder_outputs: batch x in_seq_len x hidden_size
            # hidden[0]: output_size (1) x batch x hidden_size
            # attn: batch x output_size x in_seq_len
            attn = self.attention(hidden[0], encoder_outputs)
            # context: batch x output_size x hidden_size
            context = attn.bmm(encoder_outputs)

            # combined_decoder_input: batch x output_size (1) x input_size (l_output_size+hidden_size+4)
            combined_decoder_input = torch.cat(
                (xy_decoder_input, wh_decoder_input, l_decoder_input_emb, context), dim=2
            )
        else:
            combined_decoder_input = torch.cat(
                (xy_decoder_input, wh_decoder_input, l_decoder_input_emb), dim=2
            )

        ### 2. get the RNN hidden and output ###
        # output: batch x output_size (1) x hidden_size
        # hidden[0]: output_size (1) x batch x hidden_size
        output, hidden = self.rnn(combined_decoder_input, hidden)

        ### 3. sample the bbox labels ###
        # label_softmax: batch x l_output_size
        label_logits = self.l_out(output.contiguous().view(-1, self.hidden_size))
        label_softmax = nn.Softmax(dim=-1)(label_logits).clamp(1e-5, 1)
        scores, labels = label_softmax.max(-1)
        ### 4. sample bbox xy ###
        # xy_hidden: batch x (hidden_size+l_output_size)

        if is_training:
            xy_hidden = torch.cat((hidden[0].squeeze(0), next_l_decoder_input), dim=1)
        else:
            xy_hidden = torch.cat((hidden[0].squeeze(0), label_softmax), dim=1)
        # raw_xy_gmm_param: batch x gmm_comp_num*gmm_param_num
        raw_xy_gmm_param = self.xy_out(xy_hidden)
        # xy: batch x 2
        xy_gmm_param = self.get_gmm_params(bs, raw_xy_gmm_param)

        ### 5. sample bbox wh ###
        # wh_hidden: batch x (hidden_size+l_output_size+gmm_comp_num*gmm_param_num)
        sampled_xy, sampled_wh = None, None
        if is_training:
            next_x_decoder_input = next_x_decoder_input.unsqueeze(1)
            next_y_decoder_input = next_y_decoder_input.unsqueeze(1)
            next_xy_decoder_input = self.next_xy_embedding(
                torch.cat((next_x_decoder_input, y_decoder_input), dim=1)
            )
            next_xy_decoder_input = self.next_xy_input_dropout(next_xy_decoder_input)
            wh_hidden = torch.cat((xy_hidden, next_xy_decoder_input), dim=1)
        else:
            # sampling x and y
            pi_xy, u_x, u_y, sigma_x, sigma_y, rho_xy = xy_gmm_param
            next_x_decoder_input, next_y_decoder_input = self.sample_next_state_pt(
                pi_xy, u_x, u_y, sigma_x, sigma_y, rho_xy
            )
            sampled_xy = (next_x_decoder_input, next_y_decoder_input)
            # next_x_decoder_input = torch.FloatTensor([next_x_decoder_input] * bs).type_as(x_decoder_input)
            # next_y_decoder_input = torch.FloatTensor([next_y_decoder_input] * bs).type_as(x_decoder_input)
            # if torch.cuda.is_available():
            #     next_x_decoder_input = next_x_decoder_input.cuda()
            #     next_y_decoder_input = next_y_decoder_input.cuda()
            next_x_decoder_input = next_x_decoder_input.unsqueeze(-1)
            next_y_decoder_input = next_y_decoder_input.unsqueeze(-1)
            # next_xy_decoder_input = self.next_xy_embedding(torch.cat((next_x_decoder_input,
            #                                                           y_decoder_input), dim=1))
            next_xy_decoder_input = self.next_xy_embedding(
                torch.cat((next_x_decoder_input, next_y_decoder_input), dim=1)
            )
            next_xy_decoder_input = self.next_xy_input_dropout(next_xy_decoder_input)
            wh_hidden = torch.cat((xy_hidden, next_xy_decoder_input), dim=1)
        # raw_wh_gmm_param: batch x gmm_comp_num*gmm_param_num
        raw_wh_gmm_param = self.wh_out(wh_hidden)
        # wh: batch x 2
        wh_gmm_param = self.get_gmm_params(bs, raw_wh_gmm_param)

        if not is_training:
            pi_wh, u_w, u_h, sigma_w, sigma_h, rho_wh = wh_gmm_param
            next_w_decoder_input, next_h_decoder_input = self.sample_next_state_pt(
                pi_wh, u_w, u_h, sigma_w, sigma_h, rho_wh
            )
            sampled_wh = (next_w_decoder_input, next_h_decoder_input)

        return (
            label_softmax,
            hidden,
            attn,
            xy_gmm_param,
            wh_gmm_param,
            sampled_xy,
            sampled_wh,
            label_logits,
            output,
        )

    def forward(
        self,
        encoder_hidden=None,
        encoder_outputs=None,
        target_l_variables=None,
        target_x_variables=None,
        target_y_variables=None,
        target_w_variables=None,
        target_h_variables=None,
        is_training=0,
        early_stop_len=None,
    ):
        bs = encoder_outputs.shape[0]
        ret_dict = dict()
        if self.use_attention:
            ret_dict[DecoderRNN.KEY_ATTN_SCORE] = list()

        target_l_variables, batch_size, max_length = self._validate_args(
            target_l_variables, encoder_hidden, encoder_outputs
        )
        # encoder_hidden[0]: layers*num_directions x batch x hidden_size
        # decoder_hidden[0]: layers x batch x hidden_size*num_directions
        decoder_hidden = self._init_state(encoder_hidden)

        # decoder_hidden_list = [decoder_hidden[0]]
        decoder_hidden_list, logit_list = [], []
        decoder_l_outputs = []
        # decoder_l_outputs = torch.empty(bs, 1).type_as(encoder_outputs).fill_(self.category_dict.bos())
        sequence_labels = torch.empty(bs, 1).type_as(encoder_outputs).long().fill_(self.l_sos_id)
        xy_gmm_params, wh_gmm_params = [], []
        sampled_xys = []
        sampled_whs = []
        unf_sequences = encoder_outputs.new(bs).fill_(1).long()
        # lengths = np.array([max_length] * batch_size)

        def decode(step_l_output, step_attn, seq_labels, unfinished):
            decoder_l_outputs.append(step_l_output)
            if self.use_attention:
                ret_dict[DecoderRNN.KEY_ATTN_SCORE].append(step_attn)

            labels = step_l_output.topk(1)[1].squeeze(1)
            # sequence_labels.append(labels)
            # seq_labels = torch.cat((seq_labels, next_tokens.unsqueeze(1)), dim=1)

            # eos_batches = labels.eq(self.l_eos_id)
            # if eos_batches.dim() > 0:
            #     # eos_batches = eos_batches.cpu().view(-1).numpy()
            #     update_idx = ((lengths > step) & eos_batches) != 0
            #     lengths[update_idx] = len(seq_labels)

            next_tokens = labels * unfinished + self.cat_dict.pad() * (1 - unfinished)
            unfinished = unfinished.mul((next_tokens != self.l_eos_id).long())
            seq_labels = torch.cat((seq_labels, next_tokens.unsqueeze(1)), dim=1)

            return next_tokens, seq_labels, unfinished

        if is_training:
            l_decoder_input = torch.LongTensor([self.l_sos_id] * batch_size).type_as(
                target_l_variables
            )
            x_decoder_input = torch.FloatTensor([self.x_mean] * batch_size).type_as(encoder_outputs)
            y_decoder_input = torch.FloatTensor([self.y_mean] * batch_size).type_as(encoder_outputs)
            w_decoder_input = torch.FloatTensor([self.w_mean] * batch_size).type_as(encoder_outputs)
            h_decoder_input = torch.FloatTensor([self.r_mean] * batch_size).type_as(encoder_outputs)
            next_l_decoder_input = (
                torch.FloatTensor(batch_size, self.l_output_size).zero_().type_as(encoder_outputs)
            )
            # if torch.cuda.is_available():
            #     l_decoder_input = l_decoder_input.cuda()
            #     x_decoder_input = x_decoder_input.cuda()
            #     y_decoder_input = y_decoder_input.cuda()
            #     w_decoder_input = w_decoder_input.cuda()
            #     h_decoder_input = h_decoder_input.cuda()
            #     next_l_decoder_input = next_l_decoder_input.cuda()

            for di in range(max_length):
                next_l_decoder_input[next_l_decoder_input != 0] = 0
                for batch_index in range(batch_size):
                    next_l_decoder_input[
                        batch_index, int(target_l_variables[batch_index, di + 1])
                    ] = 1
                next_x_decoder_input = target_x_variables[:, di + 1]
                next_y_decoder_input = target_y_variables[:, di + 1]
                next_w_decoder_input = target_w_variables[:, di + 1]
                next_h_decoder_input = target_h_variables[:, di + 1]
                # param: tuple 6 x tensor: BS x 5
                (
                    step_l_output,
                    decoder_hidden,
                    step_attn,
                    xy_gmm_param,
                    wh_gmm_param,
                    sampled_xy,
                    sampled_wh,
                    label_logits,
                    rnn_output,
                ) = self.forward_step(
                    l_decoder_input,
                    x_decoder_input,
                    y_decoder_input,
                    w_decoder_input,
                    h_decoder_input,
                    decoder_hidden,
                    encoder_outputs,
                    next_l_decoder_input,
                    next_x_decoder_input,
                    next_y_decoder_input,
                    is_training=is_training,
                )

                decoder_hidden_list.append(rnn_output)
                logit_list.append(label_logits)
                xy_gmm_params.append(xy_gmm_param)
                wh_gmm_params.append(wh_gmm_param)

                labels, sequence_labels, unf_sequences = decode(
                    step_l_output, step_attn, sequence_labels, unf_sequences
                )

                l_decoder_input = target_l_variables[:, di + 1]
                x_decoder_input = next_x_decoder_input
                y_decoder_input = next_y_decoder_input
                w_decoder_input = next_w_decoder_input
                h_decoder_input = next_h_decoder_input
        else:
            l_decoder_input = torch.LongTensor([self.l_sos_id] * batch_size).type_as(
                target_l_variables
            )  # .unsqueeze(0)
            x_decoder_input = torch.FloatTensor([self.x_mean] * batch_size).type_as(
                encoder_outputs
            )  # .unsqueeze(0)
            y_decoder_input = torch.FloatTensor([self.y_mean] * batch_size).type_as(
                encoder_outputs
            )  # .unsqueeze(0)
            w_decoder_input = torch.FloatTensor([self.w_mean] * batch_size).type_as(
                encoder_outputs
            )  # .unsqueeze(0)
            h_decoder_input = torch.FloatTensor([self.r_mean] * batch_size).type_as(
                encoder_outputs
            )  # .unsqueeze(0)
            # if torch.cuda.is_available():
            #     l_decoder_input = l_decoder_input.cuda()
            #     x_decoder_input = x_decoder_input.cuda()
            #     y_decoder_input = y_decoder_input.cuda()
            #     w_decoder_input = w_decoder_input.cuda()
            #     h_decoder_input = h_decoder_input.cuda()

            for di in range(early_stop_len):
                (
                    step_l_output,
                    decoder_hidden,
                    step_attn,
                    xy_gmm_param,
                    wh_gmm_param,
                    sampled_xy,
                    sampled_wh,
                    label_logits,
                    rnn_output,
                ) = self.forward_step(
                    l_decoder_input,
                    x_decoder_input,
                    y_decoder_input,
                    w_decoder_input,
                    h_decoder_input,
                    decoder_hidden,
                    encoder_outputs,
                    is_training=is_training,
                )

                labels, sequence_labels, unf_sequences = decode(
                    step_l_output, step_attn, sequence_labels, unf_sequences
                )
                # l_decoder_input = labels
                # x_decoder_input = sampled_xy[0]
                # y_decoder_input = sampled_xy[1]
                # w_decoder_input = sampled_wh[0]
                # h_decoder_input = sampled_wh[1]
                l_decoder_input = labels
                x_decoder_input = sampled_xy[0]
                y_decoder_input = sampled_xy[1]
                w_decoder_input = sampled_wh[0]
                h_decoder_input = sampled_wh[1]

                decoder_hidden_list.append(rnn_output)
                logit_list.append(label_logits)
                xy_gmm_params.append(xy_gmm_param)
                wh_gmm_params.append(wh_gmm_param)
                sampled_xys.append(torch.stack(sampled_xy, dim=-1))
                sampled_whs.append(torch.stack(sampled_wh, dim=-1))

                # if int(labels.data) == self.l_eos_id:
                if torch.sum(labels == self.l_eos_id) == batch_size:
                    break

        ret_dict["labels"] = sequence_labels[:, 1:]
        ret_dict["label_logits"] = torch.stack(logit_list, dim=1)
        # ret_dict['lengths'] = lengths.tolist()
        ret_dict["label_padding_mask"] = sequence_labels[:, 1:].eq(self.cat_dict.pad())
        ret_dict["lengths"] = (~ret_dict["label_padding_mask"]).sum(-1)  # might include eos
        ret_dict["xy"] = torch.stack(sampled_xys, dim=1) if sampled_xys else None
        ret_dict["wh"] = torch.stack(sampled_whs, dim=1) if sampled_whs else None
        ret_dict["xy_gmm_params"] = xy_gmm_params
        ret_dict["wh_gmm_params"] = wh_gmm_params
        ret_dict["obj_embed"] = torch.cat(decoder_hidden_list, dim=1)

        # return decoder_l_outputs, xy_gmm_params, wh_gmm_params, decoder_hiddens, ret_dict
        return ret_dict

    def _init_state(self, encoder_hidden):
        """Initialize the encoder hidden state."""
        if encoder_hidden is None:
            return None
        if isinstance(encoder_hidden, tuple):
            encoder_hidden = tuple([self._cat_directions(h) for h in encoder_hidden])
        else:
            encoder_hidden = self._cat_directions(encoder_hidden)
        return encoder_hidden

    def _cat_directions(self, h):
        """If the encoder is bidirectional, do the following transformation.
        (#directions * #layers, #batch, hidden_size) -> (#layers, #batch, #directions * hidden_size)
        """
        if self.bidirectional_encoder:
            h = torch.cat([h[0 : h.size(0) : 2], h[1 : h.size(0) : 2]], 2)
        return h

    def _validate_args(self, inputs, encoder_hidden, encoder_outputs):
        # print(inputs, encoder_hidden, encoder_outputs)
        # print(inputs.shape, encoder_hidden[0].shape, encoder_hidden[1].shape, encoder_outputs.shape)

        if self.use_attention:
            if encoder_outputs is None:
                raise ValueError("Argument encoder_outputs cannot be None when attention is used.")

        # inference batch size
        if inputs is None and encoder_hidden is None:
            batch_size = 1
        else:
            if inputs is not None:
                batch_size = inputs.size(0)
            else:
                if isinstance(encoder_hidden, tuple):  # self.rnn_cell is nn.LSTM:
                    batch_size = encoder_hidden[0].size(1)
                else:  # if self.rnn_cell is nn.GRU:
                    batch_size = encoder_hidden.size(1)

        # set default input and max decoding length
        if inputs is None:
            inputs = (
                torch.LongTensor([self.l_sos_id] * batch_size)
                .view(batch_size, 1)
                .type_as(encoder_outputs)
                .long()
            )
            # if torch.cuda.is_available():
            #     inputs = inputs.cuda()
            max_length = self.max_length
        else:
            max_length = inputs.size(1) - 1  # minus the start of sequence symbol

        # print('%s, %s' % (self.batch_size, batch_size))
        # assert self.batch_size == batch_size

        return inputs, batch_size, max_length

    def sample_next_state(self, pi, u_x, u_y, sigma_x, sigma_y, rho_xy):
        temperature = self.temperature

        def adjust_temp(pi_pdf):
            pi_pdf = np.log(pi_pdf) / temperature
            pi_pdf -= pi_pdf.max()
            pi_pdf = np.exp(pi_pdf)
            pi_pdf /= pi_pdf.sum()
            return pi_pdf

        # get mixture indice:
        pi = pi.data[0, :].cpu().numpy()
        pi = adjust_temp(pi)
        pi_idx = np.random.choice(self.gmm_comp_num, p=pi)
        # get mixture params:
        u_x = u_x.data[0, pi_idx]
        u_y = u_y.data[0, pi_idx]
        sigma_x = sigma_x.data[0, pi_idx]
        sigma_y = sigma_y.data[0, pi_idx]
        rho_xy = rho_xy.data[0, pi_idx]
        x, y = self.sample_bivariate_normal(
            u_x, u_y, sigma_x, sigma_y, rho_xy, temperature, greedy=False
        )
        return x, y

    def sample_next_state_pt(self, pi, u_x, u_y, sigma_x, sigma_y, rho_xy):
        temperature = self.temperature

        def adjust_temp(pi_pdf: Tensor):
            pi_pdf = torch.log(pi_pdf) / temperature
            pi_pdf -= pi_pdf.max()
            pi_pdf = torch.exp(pi_pdf)
            pi_pdf /= pi_pdf.sum()
            return pi_pdf

        # get mixture indice:
        # pi = pi.data[0, :].cpu().numpy()
        pi = adjust_temp(pi)
        pi_idx = torch.multinomial(pi, num_samples=1)
        # get mixture params:
        # u_x = u_x[:, pi_idx]
        u_x2 = u_x.gather(dim=-1, index=pi_idx)
        u_y2 = u_y.gather(dim=-1, index=pi_idx)
        sigma_x2 = sigma_x.gather(dim=-1, index=pi_idx)
        sigma_y2 = sigma_y.gather(dim=-1, index=pi_idx)
        rho_xy2 = rho_xy.gather(dim=-1, index=pi_idx)
        x, y = self.sample_bivariate_normal_pt(
            u_x2, u_y2, sigma_x2, sigma_y2, rho_xy2, temperature, greedy=False
        )
        return x, y

    def get_gmm_params(self, batch_size, gmm_params):
        # parse gmm_params: pi (gmm_comp_num), u_x (gmm_comp_num), u_y (gmm_comp_num),
        #                   sigma_x (gmm_comp_num), sigma_y (gmm_comp_num), rho_xy (gmm_comp_num)
        # pi: batch x gmm_comp_num
        pi, u_x, u_y, sigma_x, sigma_y, rho_xy = torch.split(gmm_params, self.gmm_comp_num, dim=1)

        pi = nn.Softmax(dim=1)(pi)
        sigma_x = torch.exp(sigma_x)
        sigma_y = torch.exp(sigma_y)
        rho_xy = torch.tanh(rho_xy)

        return (pi, u_x, u_y, sigma_x, sigma_y, rho_xy)

    def repackage_hidden(self, hidden):
        """Wraps hidden states in new Variables, to detach them from their history."""
        # if type(hidden) == Variable:
        #     hidden = Variable(hidden.data)
        #     if torch.cuda.is_available():
        #         hidden = hidden.cuda()
        #     return hidden
        if isinstance(hidden, torch.Tensor):
            # hidden = hidden.clone().detach().type_as(hidden)
            return hidden
        else:
            return tuple(self.repackage_hidden(v) for v in hidden)

    def sample_bivariate_normal(
        self, u_x, u_y, sigma_x, sigma_y, rho_xy, temperature, greedy=False
    ):
        # inputs must be floats
        if greedy:
            return u_x, u_y
        u_x, u_y, sigma_x, sigma_y, rho_xy = (
            u_x.cpu(),
            u_y.cpu(),
            sigma_x.cpu(),
            sigma_y.cpu(),
            rho_xy.cpu(),
        )
        mean = [u_x, u_y]
        sigma_x *= np.sqrt(temperature)
        sigma_y *= np.sqrt(temperature)
        cov = [
            [sigma_x * sigma_x, rho_xy * sigma_x * sigma_y],
            [rho_xy * sigma_x * sigma_y, sigma_y * sigma_y],
        ]
        x = np.random.multivariate_normal(mean, cov, 1)
        return x[0][0], x[0][1]

    def sample_bivariate_normal_pt(
        self, u_x, u_y, sigma_x, sigma_y, rho_xy, temperature, greedy=False
    ):
        # inputs must be floats
        if greedy:
            return u_x.squeeze(-1), u_y.squeeze(-1)
        mean = torch.cat((u_x, u_y), dim=-1)
        sigma_x *= np.sqrt(temperature)
        sigma_y *= np.sqrt(temperature)
        cov = torch.cat(
            (
                sigma_x * sigma_x,
                rho_xy * sigma_x * sigma_y,
                rho_xy * sigma_x * sigma_y,
                sigma_y * sigma_y,
            ),
            dim=-1,
        )
        cov = cov.view(sigma_x.shape[0], 2, 2)
        distrib = MultivariateNormal(loc=mean, covariance_matrix=cov)
        sample = distrib.sample()
        return sample.unbind(-1)
