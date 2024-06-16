import logging

import numpy as np
import torch
from torch import log_softmax, nn
from transformers import GPT2Config

from model.plm.gpt2_model import GPT2PatchedLMHeadModel

logger = logging.getLogger("pytorch_lightning")


class LM(nn.Module):
    def __init__(
        self,
        # is_random_init,
        # device='cuda',
        model_name="gpt2",
        # cache_dir='cache/gpt2'
        huggingface_offline=False,
    ):
        super().__init__()

        logger.info("Initialize with random weights")
        self.config = GPT2Config.from_pretrained(
            model_name,
            cache_dir="cache",
            local_files_only=huggingface_offline,
        )
        if self.config.pad_token_id is None:
            # self.tokenizer.pad_token = self.tokenizer.eos_token
            self.config.pad_token_id = self.config.eos_token_id

        self.model = GPT2PatchedLMHeadModel(self.config)

    def get_hidden_states(self, input_ids, attention_mask):
        hidden_states = self.model(input_ids, attention_mask=attention_mask)
        return hidden_states

    def get_batch_loss(self, data_batch, device="cuda"):
        """
        Assume a data batch as a list of sequences.
        """
        tokens_batch = [self.tokenizer.tokenize(line) for line in data_batch]

        token_count_batch = [len(tokens) for tokens in tokens_batch]
        batch_max_len = np.max(token_count_batch)

        tokens_padded_batch = [
            tokens + [self.tokenizer.bos_token for _ in range(batch_max_len - len(tokens))]
            for tokens in tokens_batch
        ]

        attention_mask = torch.ones(len(tokens_batch), 12, batch_max_len, batch_max_len).to(device)
        for b_idx, tokens in enumerate(tokens_batch):
            attention_mask[b_idx, :, :, len(tokens) :] = 0

        input_ids_padded_batch = [
            self.tokenizer.convert_tokens_to_ids(tokens_padded)
            for tokens_padded in tokens_padded_batch
        ]
        input_ids = torch.tensor(input_ids_padded_batch).to(device)
        label_ids_padded_batch = [
            self.tokenizer.convert_tokens_to_ids(tokens)
            + [-100 for _ in range(batch_max_len - len(tokens))]
            for tokens in tokens_batch
        ]
        label_ids = torch.tensor(label_ids_padded_batch).to(device)

        output = self.model(
            input_ids, labels=label_ids, attention_mask=attention_mask, return_dict=True
        )

        loss = output.loss
        batch_token_count = np.sum(token_count_batch) - len(
            tokens_batch
        )  # substract the count since len(tokens)-1 words are counted
        return loss, batch_token_count

    def get_loss(self, data, batch_size, device="cuda"):
        total_loss = 0
        total_token_count = 0

        for data_batch in get_batches(data, batch_size):
            loss, batch_token_count = self.get_batch_loss(data_batch, device=device)
            total_loss += loss.item() * batch_token_count
            total_token_count += batch_token_count

        return total_loss / total_token_count

    def generate(
        self, prompt, max_len=50, top_k=50, top_p=0.92, temperature=1, n_sample=1, device="cuda"
    ):
        """
        Sample from the model.
        """
        tokens = self.tokenizer.tokenize(prompt)
        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids = torch.tensor([ids]).to(device)
        output_ids_batch = self.model.generate(
            input_ids,
            do_sample=True,
            max_length=max_len,
            pad_token_id=50256,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            num_return_sequences=n_sample,
        )
        samples = [
            self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
            for output_ids in output_ids_batch
        ]
        return samples

    def get_word_level_perplexity(self, dev_lines, add_bos_token=True):
        loss_sum = 0
        total_word_count = 0

        for line in dev_lines:
            if add_bos_token:
                tokens = [self.tokenizer.bos_token] + self.tokenizer.tokenize(line)
            else:
                tokens = self.tokenizer.tokenize(line)
                if len(tokens) <= 1:
                    continue

            ids = self.tokenizer.convert_tokens_to_ids(tokens)
            input_ids = torch.tensor([ids])  # .to(device) # batch size = 1

            loss = self.model(input_ids, labels=input_ids)[0].item()
            loss_sum += loss * (len(tokens) - 1)
            total_word_count += len(line.strip().split())
        return np.exp(loss_sum / total_word_count)

    def get_surprisals(self, tokens, add_bos_token=True):
        surprisals = []
        for i in range(len(tokens)):
            token_id = self.tokenizer.convert_tokens_to_ids(tokens[i])
            if add_bos_token:
                # add BOS token
                prefix_tokens = [self.tokenizer.bos_token] + tokens[:i]
            else:
                if i == 0:
                    surprisals.append(0.0)
                    continue
                else:
                    prefix_tokens = tokens[:i]
            ids = self.tokenizer.convert_tokens_to_ids(prefix_tokens)
            input_ids = torch.tensor([ids])  # .to(device)
            output = self.model(input_ids)
            logits = output[0]
            next_token_logits = logits[:, -1, :].squeeze()
            log_probs = log_softmax(next_token_logits)
            surprisal = -log_probs[token_id] / np.log(2)
            surprisals.append(surprisal)
        return surprisals


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
