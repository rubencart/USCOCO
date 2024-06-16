import logging

import torch
from config import Config, TextEncoderConfig
from torch import nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

logger = logging.getLogger("pytorch_lightning")


class AttnGanTextEncoder(nn.Module):
    def __init__(self, cfg: Config, ntoken, drop_prob=0.5, nlayers=1, bidirectional=True):
        super().__init__()
        self.hparams = cfg
        self.n_steps = cfg.num_words
        self.ntoken = ntoken  # size of the dictionary
        self.ninput = (
            cfg.text_encoder.attn_gan_text_encoder_input_dim
        )  # size of each embedding vector
        self.drop_prob = drop_prob  # probability of an element to be zeroed
        self.nlayers = nlayers  # Number of recurrent layers
        self.bidirectional = bidirectional
        self.rnn_type = "LSTM"  # hparams.model.rnn_type
        if bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1

        # number of features in the hidden state
        self.nhidden = cfg.text_encoder.attn_gan_text_encoder_hidden_dim // self.num_directions

        self.define_module()
        self.init_weights()

    def train(self, mode: bool = True):
        # This is here to make sure this model can never be put in training mode,
        # otherwise pytorch lightning will do so at the beginning of the training loop!
        # This changes batchnorm and dropout behavior!
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/2824
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/2551
        return super().train(False)

    def define_module(self):
        """
        nn.LSTM and nn.GRU will give a warning if nlayers=1 and dropout>0,
            saying dropout is only used
            when nlayers>1. That's okay.
        """
        self.encoder = nn.Embedding(self.ntoken, self.ninput)
        self.drop = nn.Dropout(self.drop_prob)
        if self.rnn_type == "LSTM":
            # dropout: If non-zero, introduces a dropout layer on
            # the outputs of each RNN layer except the last layer
            self.rnn = nn.LSTM(
                self.ninput,
                self.nhidden,
                self.nlayers,
                batch_first=True,
                dropout=self.drop_prob,
                bidirectional=self.bidirectional,
            )
        elif self.rnn_type == "GRU":
            self.rnn = nn.GRU(
                self.ninput,
                self.nhidden,
                self.nlayers,
                batch_first=True,
                dropout=self.drop_prob,
                bidirectional=self.bidirectional,
            )
        else:
            raise NotImplementedError

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        # Do not need to initialize RNN parameters, which have been initialized
        # http://pytorch.org/docs/master/_modules/torch/nn/modules/rnn.html#LSTM
        # self.decoder.weight.data.uniform_(-initrange, initrange)
        # self.decoder.bias.data.fill_(0)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == "LSTM":
            return (
                Variable(weight.new(self.nlayers * self.num_directions, bsz, self.nhidden).zero_()),
                Variable(weight.new(self.nlayers * self.num_directions, bsz, self.nhidden).zero_()),
            )
        else:
            return Variable(
                weight.new(self.nlayers * self.num_directions, bsz, self.nhidden).zero_()
            )

    def forward(self, captions, cap_lens, hidden, mask=None):
        # input: torch.LongTensor of size batch x n_steps
        # --> emb: batch x n_steps x ninput
        emb = self.drop(self.encoder(captions))
        #
        # Returns: a PackedSequence object
        cap_lens = cap_lens.data.tolist()
        # enforce_sorted=False, see
        # https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_padded_sequence.html
        # emb = pack_padded_sequence(emb, cap_lens, batch_first=True, enforce_sorted=False)
        emb = pack_padded_sequence(emb, cap_lens, batch_first=True, enforce_sorted=True)
        # #hidden and memory (num_layers * num_directions, batch, hidden_size):
        # tensor containing the initial hidden state for each element in batch.
        # #output (batch, seq_len, hidden_size * num_directions)
        # #or a PackedSequence object:
        # tensor containing output features (h_t) from the last layer of RNN
        output, hidden = self.rnn(emb, hidden)
        # PackedSequence object
        # --> (batch, seq_len, hidden_size * num_directions)
        output = pad_packed_sequence(output, batch_first=True)[0]
        # output = self.drop(output)
        # --> batch x hidden_size*num_directions x seq_len
        words_emb = output.transpose(1, 2)
        # --> batch x num_directions*hidden_size
        if self.rnn_type == "LSTM":
            sent_emb = hidden[0].transpose(0, 1).contiguous()
        else:
            sent_emb = hidden.transpose(0, 1).contiguous()
        sent_emb = sent_emb.view(-1, self.nhidden * self.num_directions)
        return words_emb, sent_emb


class RNNTextEncoder(nn.Module):
    def __init__(self, cfg: TextEncoderConfig, tokenizer):
        super().__init__()
        # super().__init__(cfg.text_encoder)
        self.hparams = cfg
        # self.n_steps = cfg.num_words
        self.ntoken = tokenizer.original_length  # size of the dictionary
        self.ninput = cfg.attn_gan_text_encoder_input_dim  # size of each embedding vector
        self.drop_prob = cfg.attn_gan_text_encoder_dropout  # probability of an element to be zeroed
        self.nlayers = cfg.attn_gan_text_encoder_nlayers  # Number of recurrent layers
        self.bidirectional = cfg.attn_gan_text_encoder_bidirectional
        self.rnn_type = "LSTM"  # hparams.model.rnn_type
        self.initrange = 0.1
        if self.bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1

        # number of features in the hidden state
        self.hidden_dim_direction = cfg.attn_gan_text_encoder_hidden_dim // self.num_directions
        self.hidden_dim = cfg.attn_gan_text_encoder_hidden_dim

        self.define_module()

        if not cfg.txt_enc_pretrained:
            logger.info("Initializing new AttnGAN text encoder")
            self.init_weights()

        if len(tokenizer) > self.encoder.num_embeddings:
            num = len(tokenizer) - self.encoder.num_embeddings
            self.encoder.weight = nn.Parameter(
                torch.cat(
                    (
                        self.encoder.weight,
                        torch.rand(num, self.ninput).mul(2 * self.initrange).sub(self.initrange),
                    )
                )
            )
            self.ntoken += num

    def define_module(self):
        """
        nn.LSTM and nn.GRU will give a warning if nlayers=1 and dropout>0,
            saying dropout is only used
            when nlayers>1. That's okay.
        """
        self.encoder = nn.Embedding(self.ntoken, self.ninput)
        self.drop = nn.Dropout(self.drop_prob)
        if self.rnn_type == "LSTM":
            # dropout: If non-zero, introduces a dropout layer on
            # the outputs of each RNN layer except the last layer
            self.rnn = nn.LSTM(
                self.ninput,
                self.hidden_dim_direction,
                self.nlayers,
                batch_first=True,
                dropout=self.drop_prob,
                bidirectional=self.bidirectional,
            )
        elif self.rnn_type == "GRU":
            self.rnn = nn.GRU(
                self.ninput,
                self.hidden_dim_direction,
                self.nlayers,
                batch_first=True,
                dropout=self.drop_prob,
                bidirectional=self.bidirectional,
            )
        else:
            raise NotImplementedError

    def init_weights(self):
        self.encoder.weight.data.uniform_(-self.initrange, self.initrange)
        # Do not need to initialize RNN parameters, which have been initialized
        # http://pytorch.org/docs/master/_modules/torch/nn/modules/rnn.html#LSTM
        # self.decoder.weight.data.uniform_(-initrange, initrange)
        # self.decoder.bias.data.fill_(0)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == "LSTM":
            return (
                Variable(
                    weight.new(
                        self.nlayers * self.num_directions, bsz, self.hidden_dim_direction
                    ).zero_()
                ),
                Variable(
                    weight.new(
                        self.nlayers * self.num_directions, bsz, self.hidden_dim_direction
                    ).zero_()
                ),
            )
        else:
            return Variable(
                weight.new(
                    self.nlayers * self.num_directions, bsz, self.hidden_dim_direction
                ).zero_()
            )

    def forward(self, batch, return_hidden=False, hidden=None, mask=None):
        captions, cap_lens = batch["input_ids"], batch["attention_mask"].sum(-1)
        batch_size = captions.size(0)
        hidden = self.init_hidden(batch_size)

        # input: torch.LongTensor of size batch x n_steps
        # --> emb: batch x n_steps x ninput
        emb = self.drop(self.encoder(captions))
        #
        # Returns: a PackedSequence object
        emb = pack_padded_sequence(emb, cap_lens.cpu(), batch_first=True, enforce_sorted=False)
        # #hidden and memory (num_layers * num_directions, batch, hidden_size):
        # tensor containing the initial hidden state for each element in batch.
        # #output (batch, seq_len, hidden_size * num_directions)
        # #or a PackedSequence object:
        # tensor containing output features (h_t) from the last layer of RNN

        # input:
        # emb: seq_len, batch, input_size
        # hidden: num_layers * num_directions, batch, hidden_size
        # output:
        # output: batch x seq_len x num_directions * hidden_size
        # hidden: num_layers * num_directions x batch x hidden_size
        output, hidden = self.rnn(emb, hidden)

        # PackedSequence object
        # --> (batch, seq_len, hidden_size * num_directions)
        output = pad_packed_sequence(output, batch_first=True)[0]

        # return output, hidden
        return output if not return_hidden else (output, hidden)


class RNNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, bidirectional, dropout):
        super().__init__()
        # super().__init__(cfg.text_encoder)
        # self.n_steps = cfg.num_words
        # self.ntoken = vocab_size  # size of the dictionary
        self.ninput = input_dim  # size of each embedding vector
        self.drop_prob = dropout  # probability of an element to be zeroed
        self.nlayers = num_layers  # Number of recurrent layers
        self.bidirectional = bidirectional
        self.rnn_type = "LSTM"  # hparams.model.rnn_type
        self.initrange = 0.1
        if self.bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1

        # number of features in the hidden state
        self.nhidden = hidden_dim // self.num_directions
        self.nhidden_total = hidden_dim

        self.define_module()

        logger.info("Initializing new RNN encoder")

    def define_module(self):
        """
        nn.LSTM and nn.GRU will give a warning if nlayers=1 and dropout>0,
            saying dropout is only used
            when nlayers>1. That's okay.
        """
        self.drop = nn.Dropout(self.drop_prob)
        if self.rnn_type == "LSTM":
            # dropout: If non-zero, introduces a dropout layer on
            # the outputs of each RNN layer except the last layer
            self.rnn = nn.LSTM(
                self.ninput,
                self.nhidden,
                self.nlayers,
                batch_first=True,
                dropout=self.drop_prob,
                bidirectional=self.bidirectional,
            )
        elif self.rnn_type == "GRU":
            self.rnn = nn.GRU(
                self.ninput,
                self.nhidden,
                self.nlayers,
                batch_first=True,
                dropout=self.drop_prob,
                bidirectional=self.bidirectional,
            )
        else:
            raise NotImplementedError

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == "LSTM":
            return (
                Variable(weight.new(self.nlayers * self.num_directions, bsz, self.nhidden).zero_()),
                Variable(weight.new(self.nlayers * self.num_directions, bsz, self.nhidden).zero_()),
            )
        else:
            return Variable(
                weight.new(self.nlayers * self.num_directions, bsz, self.nhidden).zero_()
            )

    def forward(self, word_embs, src_key_padding_mask, return_hidden=False, hidden=None, mask=None):
        batch_size = word_embs.size(0)
        hidden = self.init_hidden(batch_size)

        # input: torch.LongTensor of size batch x n_steps
        # --> emb: batch x n_steps x ninput
        # emb = self.drop(self.encoder(captions))
        emb = self.drop(word_embs)
        #
        # Returns: a PackedSequence object
        cap_lens = (~src_key_padding_mask.bool()).sum(-1)
        emb = pack_padded_sequence(emb, cap_lens.cpu(), batch_first=True, enforce_sorted=False)
        # #hidden and memory (num_layers * num_directions, batch, hidden_size):
        # tensor containing the initial hidden state for each element in batch.
        # #output (batch, seq_len, hidden_size * num_directions)
        # #or a PackedSequence object:
        # tensor containing output features (h_t) from the last layer of RNN

        # input:
        # emb: seq_len, batch, input_size
        # hidden: num_layers * num_directions, batch, hidden_size
        # output:
        # output: batch x seq_len x num_directions * hidden_size
        # hidden: num_layers * num_directions x batch x hidden_size
        output, hidden = self.rnn(emb, hidden)

        # PackedSequence object
        # --> (batch, seq_len, hidden_size * num_directions)
        output = pad_packed_sequence(output, batch_first=True)[0]

        return output if not return_hidden else (output, hidden)
