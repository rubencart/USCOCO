import logging
import re
import sys
import time

import numpy as np
import torch
from torch import nn
from transformers import GPT2Config

from model.plm.gpt2_model import GPT2PatchedLMHeadModel

logger = logging.getLogger("pytorch_lightning")


class TG(nn.Module):
    def __init__(
        self,
        is_random_init,
        tokenizer,
        device="cuda",
        model_name="gpt2",
        cache_dir="pretrained/gpt2",
        huggingface_offline=False,
    ):
        super().__init__()
        # Load pretrained tokenizer
        # self.tokenizer = GPT2Tokenizer.from_pretrained(model_name, cache_dir=cache_dir, local_files_only=huggingface_offline)
        self.tokenizer = tokenizer
        self.ROOT = "[START]"

        if is_random_init:
            # self.config = GPT2Config(len(self.tokenizer))
            self.config = GPT2Config.from_pretrained(model_name, vocab_size=len(self.tokenizer))
            self.model = GPT2PatchedLMHeadModel(self.config)
        else:
            logger.info("Initialize with pretrained weights")
            self.model = GPT2PatchedLMHeadModel.from_pretrained(
                model_name, cache_dir=cache_dir, local_files_only=huggingface_offline
            )

        self.model.resize_token_embeddings(len(self.tokenizer))

        # # specify special tokens
        # self.SPECIAL_BRACKETS = ['-LRB-', '-RRB-', '-LCB-', '-RCB-']
        #
        # # specify GEN actions
        # self.GEN_VOCAB = self.tokenizer.convert_ids_to_tokens(range(len(self.tokenizer)-1)) + self.SPECIAL_BRACKETS  # '<|endoftext|>' not included
        #
        # # specify non-GEN parsing actions
        # self.NT_CATS = NT_CATS
        # self.REDUCE = REDUCE
        # self.self.ROOT = self.ROOT
        #
        # self.NT_ACTIONS = ["NT({})".format(cat) for cat in self.NT_CATS]
        # self.NT_ACTIONS_SET = set(self.NT_ACTIONS)
        # self.NT_ACTIONS2NT_CAT = dict([["NT({})".format(cat), cat] for cat in self.NT_CATS])
        # self.ACTIONS_SET = set(self.NT_ACTIONS + self.REDUCE)  # the set of non-terminal actions and reduce
        #
        # self.w_boundary_char = b'\xc4\xa0'.decode()
        #
        # self.a2str = {}
        # for cat in self.NT_CATS:
        #     a = "NT({})".format(cat)
        #     self.a2str[a] = '('+cat
        #
        # self.num_added_toks = self.tokenizer.add_tokens(self.SPECIAL_BRACKETS + self.NT_ACTIONS + self.REDUCE + [self.self.ROOT])
        # self.model.resize_token_embeddings(len(self.tokenizer))
        #
        # self.GEN_ids = self.tokenizer.convert_tokens_to_ids(self.GEN_VOCAB)
        # self.NT_ids = self.tokenizer.convert_tokens_to_ids(self.NT_ACTIONS)
        # self.REDUCE_ids = self.tokenizer.convert_tokens_to_ids(self.REDUCE)

        self.log_softmax = torch.nn.LogSoftmax(-1)

    def is_valid_action(
        self, action, nt_count, reduce_count, prev_action, buffer_size, max_open_nt=50
    ):
        """
        Given a parsing state, check if an action is valid or not.
        buffer_size set to -1 if sampling action sequences
        """
        flag = True
        if action in self.NT_ACTIONS_SET:
            if (buffer_size == 0) or (nt_count - (reduce_count // 2) > max_open_nt):
                flag = False
        elif action in self.REDUCE:
            if (
                (prev_action in self.NT_ACTIONS_SET)
                or (buffer_size > 0 and nt_count - (reduce_count // 2) == 1)
                or prev_action == self.ROOT
            ):
                flag = False
        else:
            if (buffer_size == 0) or (prev_action == self.ROOT):
                flag = False
        return flag

    def get_sample(self, prefix, top_k=False, device="cuda"):
        """
        Given a prefix of a generative parsing action sequence, sample a continuation
        from the model, subject to the constraints of valid actions.
        Return a bracketed tree string.
        """
        nt_count = 0
        reduce_count = 0
        tree_str = ""

        prefix_tokens = self.tokenizer.tokenize(prefix)
        prefix_ids = self.tokenizer.convert_tokens_to_ids(prefix_tokens)
        prev_token = prefix_tokens[-1]

        while (nt_count - reduce_count != 0 or nt_count == 0) and nt_count < 40:
            input_ids = torch.tensor(prefix_ids).unsqueeze(0).to(device)
            attention_mask = get_attention_mask_from_actions(
                prefix_tokens, device=self.device, n_heads=self.config.n_head
            )
            prediction_scores = self.model(input_ids, attention_mask=attention_mask)[
                0
            ]  # batch size = 1

            while True:
                token_id = sample_from_scores(prediction_scores[:, -1], top_k=top_k)[0]
                token = self.tokenizer.convert_ids_to_tokens(token_id)
                if self.is_valid_action(
                    token, nt_count, reduce_count, prev_token, buffer_size=-1, max_open_nt=50
                ):
                    break

            prefix_tokens.append(token)
            prefix_ids.append(token_id)

            if token in self.NT_ACTIONS_SET:
                if prev_token not in self.NT_ACTIONS_SET and nt_count > 0:
                    tree_str += " " + self.a2str[token] + " "
                else:
                    tree_str += self.a2str[token] + " "
                nt_count += 1
            elif token in self.REDUCE:
                if reduce_count % 2 == 0:
                    tree_str += ")"
                reduce_count += 1
            else:
                if token.startswith(self.w_boundary_char):
                    token_new = token.replace(self.w_boundary_char, "")
                    if prev_token in self.NT_ACTIONS_SET:
                        tree_str += token_new
                    else:
                        tree_str += " " + token_new
                else:
                    tree_str += token

            prev_token = token
        return tree_str

    def get_actions_and_ids(self, nt_count, reduce_count, buffer_size, prev_action, max_open_nt=50):
        """
        Given parameters of a parser state, return a list of all possible valid actions and a list of their ids.
        """
        valid_actions = []
        valid_action_ids = []

        # NT actions
        if (buffer_size < 1) or (nt_count - reduce_count > max_open_nt):
            pass
        else:
            valid_actions += self.NT_ACTIONS
            valid_action_ids += self.NT_ids

        # REDUCE action
        if (
            (prev_action in self.NT_ACTIONS_SET)
            or (buffer_size > 0 and nt_count - reduce_count == 1)
            or prev_action == self.ROOT
        ):
            pass
        else:
            valid_actions += self.REDUCE
            valid_action_ids += self.REDUCE_ids

        # GEN action
        if buffer_size < 1 or prev_action == self.ROOT:
            pass
        else:
            valid_actions += self.GEN_VOCAB
            valid_actions += self.GEN_ids

        return valid_actions, valid_action_ids

    def select_valid_actions(
        self, actions, nt_count, reduce_count, buffer_size, prev_action, max_open_nt=50
    ):
        """
        Given parameters of a parser state as well as a subset of all possible actions, select the valid actions.
        """
        valid_actions = []
        for action in actions:
            flag = self.is_valid_action(
                action,
                nt_count=nt_count,
                reduce_count=reduce_count,
                buffer_size=buffer_size,
                prev_action=prev_action,
                max_open_nt=max_open_nt,
            )
            if flag:
                valid_actions.append(action)
        return valid_actions

    def add_gen_vocab(self, valid_actions):
        if valid_actions[-1] in self.ACTIONS_SET:
            action_all = valid_actions
        else:
            action_all = valid_actions[:-1] + self.GEN_VOCAB
        return action_all

    def get_adist(self, scores, valid_actions, index=-1):
        """
        Given logit scores and a list of valid actions, return normalized probability distribution over valid actions
        and a dictionary mapping specific action tokens to corresponding position index in the valid action list.
        """
        if valid_actions[-1] in self.ACTIONS_SET:
            action_all = valid_actions
        else:
            action_all = valid_actions[:-1] + self.GEN_VOCAB

        valid_action_ids = self.tokenizer.convert_tokens_to_ids(action_all)
        action_scores = scores[:, index, valid_action_ids]
        adist = self.log_softmax(action_scores.squeeze())
        a2idx = dict([[a, idx] for idx, a in enumerate(action_all)])
        return adist, a2idx

    def decode_tree_str(self, prefix_actions):
        """
        Given a sequence of actions, return a bracketed tree string.
        """
        tree_str = ""
        reduce_count = 0
        for a in prefix_actions:
            if a in self.REDUCE:
                if reduce_count % 2 == 0:
                    tree_str += ")"
                reduce_count += 1
            elif a in self.NT_ACTIONS_SET:
                a_cat = self.NT_ACTIONS2NT_CAT[a]
                tree_str += " (" + a_cat
            else:
                if a.startswith(self.w_boundary_char):
                    term_new = a.replace(self.w_boundary_char, "")
                    tree_str += " " + term_new
                else:
                    tree_str += a
        return tree_str

    def get_adist_batch(self, pq_this, token, buffer_size, batch_size=50, device="cuda"):
        """
        Given a list of incremental parser states, get the probability distribution
        of valid incoming actions for each parser state. Perform batched computations.
        """
        pq_this_adist = None

        # compute total number of batches needed
        pq_this_len = len(pq_this)
        if pq_this_len % batch_size == 0:
            num_batches = pq_this_len // batch_size
        else:
            num_batches = pq_this_len // batch_size + 1

        for i in range(num_batches):
            start_idx = batch_size * i
            end_idx = batch_size * (i + 1)
            # set_trace(context=10)
            # print(start_idx, end_idx)
            pq_this_batch = pq_this[start_idx:end_idx]

            prefix_max_len = np.max([len(p_this.prefix_actions) for p_this in pq_this_batch])

            input_ids_batch = torch.tensor([
                self.tokenizer.convert_tokens_to_ids(
                    p_this.prefix_actions
                    + ["#" for _ in range(prefix_max_len - len(p_this.prefix_actions))]
                )
                for p_this in pq_this_batch
            ]).to(device)

            # attention_mask_batch = torch.ones(len(pq_this_batch), 12, prefix_max_len, prefix_max_len).to(device)
            # for b_idx, p_this in enumerate(pq_this_batch):
            #     attention_mask_batch[b_idx, :, :, :] = get_attention_mask_from_actions(p_this.prefix_actions, max_len=prefix_max_len, device=device)
            # attention_mask_batch = torch.ones(len(pq_this_batch), 12, prefix_max_len, prefix_max_len).to(device)
            attention_mask_batch = torch.ones(
                len(pq_this_batch), self.config.n_head, prefix_max_len, prefix_max_len
            ).to(device)
            for b_idx, p_this in enumerate(pq_this_batch):
                attention_mask_batch[b_idx, :, :, :] = get_attention_mask_from_actions(
                    p_this.prefix_actions,
                    n_heads=self.config.n_head,
                    max_len=prefix_max_len,
                    device=self.device,
                )

            prediction_scores_batch = self.model(
                input_ids_batch, attention_mask=attention_mask_batch
            )[0]

            pq_this_all_valid_actions_batch = []
            pq_this_all_valid_action_ids_batch = []
            for p_this in pq_this[batch_size * i : batch_size * (i + 1)]:
                valid_actions, valid_action_ids = self.get_actions_and_ids(
                    p_this.nt_count, p_this.reduce_count, buffer_size, p_this.prev_a, max_open_nt=50
                )
                pq_this_all_valid_actions_batch.append(valid_actions)
                pq_this_all_valid_action_ids_batch.append(valid_action_ids)

            scores_index_mask = torch.zeros(len(pq_this_batch), len(self.tokenizer)).to(device)

            indice_dim_0 = []
            indice_dim_1 = []
            for p_index, valid_action_ids in enumerate(pq_this_all_valid_action_ids_batch):
                indice_dim_0 += [p_index for _ in range(len(valid_action_ids))]
                indice_dim_1 += valid_action_ids
            indice_dim_0 = torch.tensor(indice_dim_0).to(device)
            indice_dim_1 = torch.tensor(indice_dim_1).to(device)

            scores_index_mask[indice_dim_0, indice_dim_1] = 1

            pq_this_action_scores_batch = prediction_scores_batch[
                torch.arange(len(pq_this_batch)),
                [len(p_this.prefix_actions) - 1 for p_this in pq_this_batch],
                :,
            ]
            pq_this_action_scores_batch.masked_fill(scores_index_mask == 0, -np.inf)
            if pq_this_adist == None:
                pq_this_adist = self.log_softmax(pq_this_action_scores_batch).detach()
            else:
                pq_this_adist = torch.cat(
                    (pq_this_adist, self.log_softmax(pq_this_action_scores_batch).detach()), 0
                )
        return pq_this_adist

    def get_surprisals_with_beam_search(
        self,
        sent,
        beam_size=100,
        word_beam_size=10,
        fast_track_size=5,
        batched=True,
        debug=False,
        device="cuda",
    ):
        """
        Estimate surprisals -log2(P(x_t|x_1 ... x_{t-1})) at subword token level.
        Return a list of subword tokens and surprisals.
        """
        tokens = self.tokenizer.tokenize(sent)

        prefix = self.ROOT
        prefix_tokens = self.tokenizer.tokenize(prefix)

        surprisals = []
        log_probs = []

        pq_this = []
        init = ParserState(
            prefix_actions=[self.ROOT], score=0, nt_count=0, reduce_count=0, prev_a=self.ROOT
        )
        pq_this.append(init)

        for k, token in enumerate(tokens):
            print("Token index: {} {}".format(k, token), file=sys.stderr)
            pq_next = []

            while len(pq_next) < beam_size:
                fringe = []

                if batched:
                    start_time = time.time()

                    if k <= 80:
                        eval_batch_size = 100
                    else:
                        eval_batch_size = 50

                    pq_this_adist_batch = self.get_adist_batch(
                        pq_this, token, buffer_size=len(tokens) - k, batch_size=eval_batch_size
                    )

                start_time = time.time()

                for p_index, p_this in enumerate(pq_this):
                    actions = self.NT_ACTIONS + self.REDUCE + [token]
                    buffer_size = len(tokens) - k
                    current_valid_actions = self.select_valid_actions(
                        actions,
                        p_this.nt_count,
                        p_this.reduce_count,
                        buffer_size,
                        p_this.prev_a,
                        max_open_nt=50,
                    )

                    if batched:
                        # using batched computation
                        adist = pq_this_adist_batch[p_index]
                    else:
                        # not using batched computation
                        input_ids = (
                            torch.tensor(
                                self.tokenizer.convert_tokens_to_ids(p_this.prefix_actions)
                            )
                            .unsqueeze(0)
                            .to(device)
                        )
                        prediction_scores = self.model(input_ids)[0]
                        adist, a2idx = self.get_adist(prediction_scores, current_valid_actions)

                    for action in current_valid_actions:
                        if batched:
                            a_idx = self.tokenizer.convert_tokens_to_ids(action)
                        else:
                            a_idx = a2idx[action]
                        new_score = p_this.score + adist[a_idx].item()
                        new_nt_count = (
                            p_this.nt_count + 1
                            if action in self.NT_ACTIONS_SET
                            else p_this.nt_count
                        )
                        new_reduce_count = (
                            p_this.reduce_count + 1
                            if action in self.REDUCE
                            else p_this.reduce_count
                        )
                        p_state = ParserState(
                            prefix_actions=p_this.prefix_actions + [action],
                            score=new_score,
                            nt_count=new_nt_count,
                            reduce_count=new_reduce_count,
                            prev_a=action,
                        )
                        fringe.append(p_state)

                fringe = prune(fringe, len(fringe))
                fast_track_count = 0
                cut = np.max([len(fringe) - beam_size, 0])

                pq_this_new = []
                for k in range(len(fringe) - 1, -1, -1):
                    if k >= cut:
                        if fringe[k].prev_a not in self.ACTIONS_SET:
                            pq_next.append(fringe[k])
                        else:
                            pq_this_new.append(fringe[k])
                    else:
                        if (
                            fringe[k].prev_a not in self.ACTIONS_SET
                            and fast_track_count < fast_track_size
                        ):
                            pq_next.append(fringe[k])
                            fast_track_count += 1
                pq_this = pq_this_new

                if debug:
                    logger.info(
                        "--- %s seconds for sorting parser states---" % (time.time() - start_time)
                    )

            pruned_pq_next = prune(pq_next, word_beam_size)

            print("List of partial parses:", file=sys.stderr)
            for beam_index, pstate in enumerate(pruned_pq_next):
                print(
                    "{} {:.3f} {}".format(
                        beam_index, pstate.score, self.decode_tree_str(pstate.prefix_actions)
                    ),
                    file=sys.stderr,
                )

            # Use log-sum-exp
            log_probs.append(-logsumexp([ps.score for ps in pruned_pq_next]) / np.log(2))

            pq_this = pruned_pq_next

        for k in range(len(log_probs)):
            if k == 0:
                surprisals.append(log_probs[k])
            else:
                surprisals.append(log_probs[k] - log_probs[k - 1])

        for k in range(len(surprisals)):
            print(tokens[k], surprisals[k], file=sys.stderr)

        return tokens, surprisals

    def tokenize_batch(self, line_batch):
        # Tokenize a batch of sequences. Add prefix space.
        words_batch = [line.strip().split() for line in line_batch]
        tokens_batch = [
            [
                token
                for word in words
                for token in self.tokenizer.tokenize(word, add_prefix_space=True)
            ]
            for words in words_batch
        ]
        return tokens_batch

    def get_hidden_states_old(self, lines, device="cuda"):
        line_batch = [self.ROOT + " " + line for line in lines]
        tokens_batch = self.tokenize_batch(line_batch)

        token_count_batch = [len(tokens) for tokens in tokens_batch]
        batch_max_len = np.max(token_count_batch)

        tokens_padded_batch = [
            tokens + [self.tokenizer.bos_token for _ in range(batch_max_len - len(tokens))]
            for tokens in tokens_batch
        ]

        ids_batch = [
            self.tokenizer.convert_tokens_to_ids(tokens_padded)
            for tokens_padded in tokens_padded_batch
        ]
        input_ids = torch.tensor(ids_batch).to(device)

        label_ids_batch = [
            self.tokenizer.convert_tokens_to_ids(tokens)
            + [-100 for _ in range(batch_max_len - len(tokens))]
            for tokens in tokens_batch
        ]
        label_ids = torch.tensor(label_ids_batch).to(device)

        # attention_mask = torch.ones(len(tokens_batch), 12, batch_max_len, batch_max_len).to(self.device)
        attention_mask = torch.ones(
            len(tokens_batch), self.config.n_head, batch_max_len, batch_max_len
        ).to(self.device)

        for b_idx, tokens in enumerate(tokens_batch):
            attention_mask[b_idx, :, :, :] = get_attention_mask_from_actions(
                tokens_batch[b_idx],
                max_len=batch_max_len,
                device=device,
                n_heads=self.config.n_head,
            )

        hidden_states = self.model(
            input_ids,
            labels=label_ids,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True,
        )
        out_mask = torch.ones(len(tokens_batch), batch_max_len).to(device)
        for b_idx, tokens in enumerate(tokens_batch):
            out_mask[b_idx, len(tokens) :] = 0
        return hidden_states, out_mask

    def get_hidden_states(
        self,
        input_ids,
        attention_mask_2d,
        tokens_batch,
    ):
        # line_batch = [self.ROOT + ' ' + line for line in lines]
        # tokens_batch = self.tokenize_batch(line_batch)
        #
        # token_count_batch = [len(tokens) for tokens in tokens_batch]
        # batch_max_len = np.max(token_count_batch)
        #
        # tokens_padded_batch = [tokens + [self.tokenizer.bos_token for _ in range(batch_max_len - len(tokens))] for
        #                        tokens in tokens_batch]
        #
        # ids_batch = [self.tokenizer.convert_tokens_to_ids(tokens_padded) for tokens_padded in tokens_padded_batch]
        # input_ids = torch.tensor(ids_batch).to(device)
        #
        # label_ids_batch = [
        #     self.tokenizer.convert_tokens_to_ids(tokens) + [-100 for _ in range(batch_max_len - len(tokens))] for tokens
        #     in tokens_batch]
        # label_ids = torch.tensor(label_ids_batch).to(device)

        # todo
        # # attention_mask = torch.ones(len(tokens_batch), 12, batch_max_len, batch_max_len).to(self.device)
        batch_max_len = int(max(attention_mask_2d.sum(-1)))
        attention_mask = torch.ones(
            len(tokens_batch), self.config.n_head, batch_max_len, batch_max_len
        )
        attention_mask = attention_mask.type_as(input_ids)

        for b_idx, tokens in enumerate(tokens_batch):
            sample_attn_mask = get_attention_mask_from_actions(
                tokens, max_len=batch_max_len, n_heads=self.config.n_head
            )
            attention_mask[b_idx, :, :, :] = sample_attn_mask.type_as(input_ids)

        hidden_states = self.model(
            input_ids, attention_mask=attention_mask, return_dict=True, output_hidden_states=True
        )
        # out_mask = torch.ones(len(input_ids), batch_max_len)
        # for b_idx, tokens in enumerate(tokens_batch):
        #     out_mask[b_idx, len(tokens):] = 0
        return hidden_states

    def get_validation_loss(self, dev_lines, device="cuda"):
        loss_sum = 0
        token_count = 0

        for line in dev_lines:
            words = line.strip().split()
            tokens = [self.ROOT] + [
                token
                for word in words
                for token in self.tokenizer.tokenize(word, add_prefix_space=True)
            ]
            ids = self.tokenizer.convert_tokens_to_ids(tokens)
            input_ids = torch.tensor([ids]).to(device)  # batch size = 1

            # play actions on RNNG state machine to get the different states
            # and derive mask values from them
            # size [1, num_heads, from_seq_length, to_seq_length]
            attention_mask = get_attention_mask_from_actions(
                tokens, device=self.device, n_heads=self.config.n_head
            )

            # Update model
            loss = self.model(input_ids, labels=input_ids, attention_mask=attention_mask)[0].item()

            loss_sum += loss * (len(tokens) - 1)
            token_count += len(tokens) - 1
        return loss_sum / token_count

    def get_batch_loss(self, lines, device="cuda"):
        """
        Compute the loss for one batch of action sequences.
        """
        line_batch = [self.ROOT + " " + line for line in lines]
        tokens_batch = self.tokenize_batch(line_batch)

        token_count_batch = [len(tokens) for tokens in tokens_batch]
        batch_max_len = np.max(token_count_batch)

        tokens_padded_batch = [
            tokens + [self.tokenizer.bos_token for _ in range(batch_max_len - len(tokens))]
            for tokens in tokens_batch
        ]

        ids_batch = [
            self.tokenizer.convert_tokens_to_ids(tokens_padded)
            for tokens_padded in tokens_padded_batch
        ]
        input_ids = torch.tensor(ids_batch).to(device)

        label_ids_batch = [
            self.tokenizer.convert_tokens_to_ids(tokens)
            + [-100 for _ in range(batch_max_len - len(tokens))]
            for tokens in tokens_batch
        ]
        label_ids = torch.tensor(label_ids_batch).to(device)

        # attention_mask = torch.ones(len(tokens_batch), 12, batch_max_len, batch_max_len).to(device)
        attention_mask = torch.ones(
            len(tokens_batch), self.config.n_head, batch_max_len, batch_max_len
        ).to(device)
        for b_idx, tokens in enumerate(tokens_batch):
            attention_mask[b_idx, :, :, :] = get_attention_mask_from_actions(
                tokens_batch[b_idx],
                max_len=batch_max_len,
                device=device,
                n_heads=self.config.n_head,
            )
        loss = self.model(input_ids, labels=label_ids, attention_mask=attention_mask)[0]

        batch_token_count = np.sum(token_count_batch) - len(
            tokens_batch
        )  # substract the count since len(tokens)-1 words are counted
        return loss, batch_token_count

    def get_loss(self, lines, batch_size, device="cuda"):
        """
        Compute the loss on a list of action sequences via batched computations.
        """
        total_loss = 0
        total_token_count = 0

        for data_batch in get_batches(lines, batch_size):
            loss, batch_token_count = self.get_batch_loss(data_batch, device=device)
            total_loss += loss.item() * batch_token_count
            total_token_count += batch_token_count

        return total_loss / total_token_count

    def estimate_word_ppl(self, dev_lines, device="cuda"):
        loss_sum = 0
        word_count = 0

        for line in dev_lines:
            words = line.strip().split()
            tokens = [self.ROOT] + [
                token
                for word in words
                for token in self.tokenizer.tokenize(word, add_prefix_space=True)
            ]
            ids = self.tokenizer.convert_tokens_to_ids(tokens)
            input_ids = torch.tensor([ids]).to(device)  # batch size = 1

            # play actions on RNNG state machine to get the different states
            # and derive mask values from them
            # size [1, num_heads, from_seq_length, to_seq_length]
            attention_mask = get_attention_mask_from_actions(
                tokens, device=self.device, n_heads=self.config.n_head
            )

            # Update model
            loss = self.model(input_ids, labels=input_ids, attention_mask=attention_mask)[0].item()

            loss_sum += loss * (len(tokens) - 1)
            word_count += len([word for word in words if word not in self.ACTIONS_SET])
        return np.exp(loss_sum / word_count)


def load_data(path):
    with open(path) as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    lines = [line for line in lines if line != ""]
    return lines


def get_batches(lines, batch_size):
    if len(lines) % batch_size == 0:
        num_batches = len(lines) // batch_size
    else:
        num_batches = len(lines) // batch_size + 1
    batches = []
    for i in range(num_batches):
        start_index = i * batch_size
        end_index = (i + 1) * batch_size
        batch = lines[start_index:end_index]
        batches.append(batch)
    return batches


def sample_from_scores(logits, top_k=50):
    kth_vals, kth_idx = logits.topk(top_k, dim=-1)
    sample_dist = torch.distributions.categorical.Categorical(logits=kth_vals)
    token_idx_new = kth_idx.gather(dim=1, index=sample_dist.sample().unsqueeze(-1)).squeeze(-1)
    return token_idx_new.tolist()


class ParserState:
    def __init__(self, prefix_actions, score, nt_count, reduce_count, prev_a):
        self.prefix_actions = prefix_actions
        self.score = score
        self.nt_count = nt_count
        self.reduce_count = reduce_count
        self.prev_a = prev_a


def prune(p_state_list, k):
    if len(p_state_list) == 1:
        return p_state_list
    if k > len(p_state_list):
        k = len(p_state_list)
    score_list = [item.score for item in p_state_list]
    sorted_indice = np.argsort(score_list)[-k:]
    pruned_p_state_list = [p_state_list[i] for i in sorted_indice]
    return pruned_p_state_list


nt_re = re.compile("NT\((.*)\)")
reduce_re = re.compile("REDUCE\((.*)\)")


class RNNGMachine:
    def __init__(self):
        self.nt_stack = []
        self.previous_stacks = []
        self.actions = []
        self.need_reduce = False
        self.composed_nt = None

    def update(self, action, mask_size):
        mask = torch.zeros(mask_size)

        if nt_re.match(action):
            assert not self.need_reduce
            label = nt_re.match(action).groups()[0]
            self.nt_stack.append(len(self.actions))
            self.previous_stacks.append(len(self.actions))

        elif reduce_re.match(action):

            # specify that start position of the non-terminal phrase to be composed
            if not self.need_reduce:
                self.need_reduce = True
                assert len(self.nt_stack)
                nt_idx = self.nt_stack.pop()
                stack_idx = self.previous_stacks.index(nt_idx)
                self.previous_stacks, compose = (
                    self.previous_stacks[:stack_idx],
                    self.previous_stacks[stack_idx:],
                )

                self.previous_stacks.append(len(self.actions))
                mask[compose] = 1
                mask[len(self.actions)] = 1
                self.actions.append(action)
                return mask

            else:
                self.need_reduce = False
                mask[len(self.actions)] = 1

        else:
            assert not self.need_reduce
            self.previous_stacks.append(len(self.actions))

        # Store action
        mask[self.previous_stacks] = 1
        self.actions.append(action)
        return mask


def get_attention_mask_from_actions(tokens, max_len=None, n_heads=12):
    """
    Given a list of actions, it returns the attention head masks for all the
    parser states
    """

    # Start RNNG Machine
    rnng_machine = RNNGMachine()
    if max_len is None:
        # single sentence
        # attention_mask = torch.ones(1, 12, len(tokens), len(tokens))
        attention_mask = torch.ones(1, n_heads, len(tokens), len(tokens))
    else:
        # multiple sentence, we need to pad to max_len
        # attention_mask = torch.ones(1, 12, max_len, max_len)
        attention_mask = torch.ones(1, n_heads, max_len, max_len)

    attention_mask = attention_mask  # .to(device)
    for t, action in enumerate(tokens):
        # update machine
        mask = rnng_machine.update(action, attention_mask.size()[-1])
        # store state as masks of transformer
        attention_mask[:, :, t, :] = mask  # .to(device)

    # ensure pad is zero at testing
    if max_len is not None:
        attention_mask[0, :, :, len(tokens) :] = 0

    return attention_mask


def logsumexp(x):
    c = np.max(x)
    return c + np.log(np.sum(np.exp(x - c)))
