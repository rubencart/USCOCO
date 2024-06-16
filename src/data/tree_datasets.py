import json
import logging
import re
from itertools import repeat
from typing import Any, Dict, List, cast

import torch
from config import Config
from model.plm.model import get_actions_and_terms, is_valid_action_sequence
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm

from data.tree_utils import tg_to_right_branching

logger = logging.getLogger("pytorch_lightning")


class TreeDataset(Dataset):
    def __init__(self, node_features, positions, masks=None, n=2, k=32, mask_value=-1):
        self.node_features = node_features
        self.positions = positions
        self.masks = masks
        self.max_n = n
        self.max_k = k

        self.rootpos = torch.zeros(n * k)

        self.infer_masks = False
        self.MASK_VALUE = mask_value
        if masks is None:
            self.infer_masks = True

    def __len__(self):
        return len(self.node_features)

    def __getitem__(self, idx):
        if self.node_features is None:
            node_feat = None
        else:
            node_feat = torch.tensor(self.node_features[idx])
        positions = torch.tensor(self.positions[idx])
        if self.infer_masks:
            mask = (positions[:, 0] != self.MASK_VALUE).int()
        else:
            mask = torch.tensor(self.masks[idx])

        positions = self.transform_bintree_positions(positions, mask)

        return {"node_feat": node_feat, "node_pos": positions, "node_mask": mask}

    def collate(self, list_of_samples: List[Dict[str, Any]]):
        node_feat, node_pos, node_mask = zip(
            *((s["node_feat"], s["node_pos"], s["node_mask"]) for s in list_of_samples)
        )
        node_pos, node_feat, node_mask = (
            torch.stack(node_pos),
            torch.stack(node_feat),
            torch.stack(node_mask),
        )
        used_nodes_mask = cast(Tensor, node_mask).sum(dim=0).bool()
        return {
            "node_feat": cast(Tensor, node_feat)[:, used_nodes_mask],
            "node_pos": cast(Tensor, node_pos)[:, used_nodes_mask],
            "node_mask": cast(Tensor, node_mask)[:, used_nodes_mask],
        }

    def U(self, x):
        return torch.cat((x[self.max_n :], torch.zeros(self.max_n)), 0)

    def D(self, x, n):
        assert n < self.max_n
        e = torch.zeros(self.max_n)
        e[n] = 1
        return torch.cat((e, x[: -self.max_n]), 0)

    def _pos_from_path(self, path):
        return self._pos_from_path_rec(path, self.rootpos)

    def _pos_from_path_rec(self, path, parent):
        if not path:
            return parent
        return self._pos_from_path_rec(path[1:], self.D(parent, path[0]))

    def transform_bintree_positions(self, positions, mask):
        assert self.max_n == 2
        m_positions = positions[mask.bool()]

        # leafs of format [x, x] to format [-1, x]
        leaf_mask = m_positions[:, 0] == m_positions[:, 1]
        m_positions[leaf_mask, 0] = -1

        max_idx = torch.max(m_positions)
        m_result = -torch.ones((m_positions.size(0), self.rootpos.size(0)))
        root = torch.tensor([0, max_idx])
        root_mask = (m_positions == root).sum(dim=1) == 2
        m_result[root_mask] = self.rootpos
        m_result = self._transform_bintree_positions_rec(m_positions, root, self.rootpos, m_result)
        result = -torch.ones((positions.size(0), self.rootpos.size(0)))
        result[mask.bool()] = m_result
        return result

    def _transform_bintree_positions_rec(self, m_positions, curr_node, curr_node_res, m_result):
        left_mask = (m_positions[:, 0] == curr_node[0]) * (m_positions[:, 1] < curr_node[1])
        if torch.all(left_mask == False):
            left_pos = torch.tensor([-1, curr_node[0]])
            left_pos_mask = (m_positions == left_pos).sum(dim=1) == 2
            left_pos_res = self.D(curr_node_res, 0)
            m_result[left_pos_mask] = left_pos_res
        else:
            biggest_idx = torch.max(m_positions[left_mask])
            left_pos = torch.tensor([curr_node[0], biggest_idx])
            left_pos_mask = (m_positions == left_pos).sum(dim=1) == 2
            left_pos_res = self.D(curr_node_res, 0)
            m_result[left_pos_mask] = left_pos_res
            m_result = self._transform_bintree_positions_rec(
                m_positions, left_pos, left_pos_res, m_result
            )

        right_mask = (m_positions[:, 1] == curr_node[1]) * (m_positions[:, 0] > curr_node[0])
        if torch.all(right_mask == False):
            right_pos = torch.tensor([-1, curr_node[1]])
            right_pos_mask = (m_positions == right_pos).sum(dim=1) == 2
            right_pos_res = self.D(curr_node_res, 1)
            m_result[right_pos_mask] = right_pos_res
        else:
            smalles_idx = torch.min(m_positions[right_mask])
            right_pos = torch.tensor([smalles_idx, curr_node[1]])
            right_pos_mask = (m_positions == right_pos).sum(dim=1) == 2
            right_pos_res = self.D(curr_node_res, 1)
            m_result[right_pos_mask] = right_pos_res
            m_result = self._transform_bintree_positions_rec(
                m_positions, right_pos, right_pos_res, m_result
            )

        return m_result


class CocoTreeDataset(TreeDataset):
    def __init__(self, img_ids, node_features, positions, masks=None, n=2, k=32, mask_value=-1):
        super().__init__(node_features, positions, masks, n=n, k=k, mask_value=mask_value)
        self.img_ids = img_ids

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        item["image_id"] = self.img_ids[idx]
        return item

    def collate(self, list_of_samples: List[Dict[str, Any]]):
        image_ids = tuple(s["image_id"] for s in list_of_samples)
        return {"image_ids": image_ids, **super().collate(list_of_samples)}


class CocoSyntaxTreeDataset(CocoTreeDataset):
    def __init__(self, path_to_json: str, vocab, n=2, k=32, mask_value=-1):
        self.path_to_json = path_to_json
        self.vocab = vocab
        (
            img_ids,
            self.captions,
            self.positions,
            self.caption_ids,
            self.tok_capts,
            self.img_id_to_indices,
        ) = self.read_json()
        super().__init__(
            img_ids=img_ids,
            node_features=None,
            positions=self.positions,
            masks=None,
            n=n,
            k=k,
            mask_value=mask_value,
        )

    def read_json(self):
        idx = 0
        img_ids, captions, positions, caption_ids, tok_capts, img_id_to_indices = (
            [],
            [],
            [],
            [],
            [],
            {},
        )
        with open(self.path_to_json, "r") as f:
            for line in f:
                loaded_line = json.loads(line)
                if len(loaded_line) == 4:
                    img_id, caption, span, capt_id = loaded_line
                else:
                    raise ValueError()

                if span != []:
                    try:
                        max_pos = span[-1][-1]
                    except IndexError:
                        print("skipped empty line")
                    pos = []
                    for i in range(max_pos + 1):
                        pos.append([i, i])
                    pos.extend(span)
                    positions.append(pos)

                    img_ids.append(img_id)
                    caption_ids.append(capt_id)
                    captions.append(caption)
                    caption = [clean_number(w) for w in caption.strip().lower().split()]
                    tok_caption = [self.vocab(token) for token in caption]
                    tok_capts.append(tok_caption)
                    img_id_to_indices.setdefault(int(img_id), []).append(idx)
                    idx += 1
        return img_ids, captions, positions, caption_ids, tok_capts, img_id_to_indices

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        return {
            "caption": self.captions[idx],
            "tokens": torch.tensor(self.tok_capts[idx]),
            "caption_id": self.caption_ids[idx],
            "no_enc_pos": self.positions[idx],
            **item,
        }

    def collate(self, list_of_samples: List[Dict[str, Any]]):
        captions, tokens, caption_ids, no_enc_pos = zip(
            *(
                (s["caption"], s["tokens"], s["caption_id"], s["no_enc_pos"])
                for s in list_of_samples
            )
        )
        return {
            "captions": captions,
            "tokens": tokens,
            "caption_ids": caption_ids,
            "no_enc_pos": no_enc_pos,
            **super().collate(list_of_samples),
        }

    def get_indices_for_image_id(self, image_id, max_size=None):
        indices = self.img_id_to_indices[image_id]
        if max_size is not None:
            indices = list(filter(lambda img_id: self._size_of_id(img_id) <= max_size, indices))
        return indices

    def _size_of_id(self, image_id):
        item = self[image_id]
        return len(item["node_mask"][item["node_mask"].bool()])


def clean_number(w):
    new_w = re.sub("[0-9]{1,}([,.]?[0-9]*)*", "N", w)
    return new_w


class CocoNotatedSyntaxTreeDataset:
    def __init__(
        self,
        cfg: Config,
        captions_file: str,
        use_tg=None,
        right_branching=None,
        index_by_capt_id=False,
    ):
        self.num_words = cfg.num_words
        self.tg_encoding = cfg.use_tg if use_tg is None else use_tg
        self.rb = (
            cfg.text_encoder.tg_right_branching if right_branching is None else right_branching
        )
        self.captions_file = captions_file
        self.capt_idx = index_by_capt_id
        self.captions_ds = self._preprocess_caption_ds(captions_file)

    def __getitem__(self, item):
        return self.captions_ds[item]

    def _preprocess_caption_ds(self, caption_file):
        with open(caption_file) as f:
            lines = f.readlines()

        res = {}
        errors = {
            "empty": [],
            "invalid": [],
            "too_long": [],
        }
        total, chunksize = len(lines), min(30000, len(lines) // 2)
        with torch.multiprocessing.Pool(max(2, 1 + total // chunksize // 3)) as pool:
            result = pool.starmap(
                self._process_line,
                tqdm(
                    zip(lines, repeat(self.num_words), repeat(self.tg_encoding), repeat(self.rb)),
                    total=total,
                ),
                chunksize=chunksize,
            )
        for i, (line, actions, terms, err) in enumerate(result):
            if err is not None:
                errors[err].append(i)
            else:
                idx = line[0] if not self.capt_idx else line[3]
                res.setdefault(idx, []).append({
                    "image_id": line[0],
                    "caption": line[1],
                    "trees": " ".join(actions),
                    # 'tree': line[2],
                    "actions": actions,
                    "terms": terms,
                    "caption_id": line[3],
                })
                # res[line[3]] = {'image_id': line[0],
                #                 'caption': line[1],
                #                 'tree': line[2],
                #                 'actions': actions,
                #                 'terms': terms,
                #                 'caption_id': line[3],
                #                 }

        logger.info(
            "Removed %s captions because empty tree, %s because invalid action, "
            "%s because too long sentence"
            % (len(errors["empty"]), len(errors["invalid"]), len(errors["too_long"]))
        )
        return res

    @staticmethod
    def _process_line(line, max_words, tg_encoding=False, right_branching=False):
        load_line = json.loads(line)
        syntree = load_line[2].strip()

        if syntree == "":
            logger.debug("empty sentence in line %s" % str(syntree))
            return None, None, None, "empty"

        # assert that the parenthesis are balanced
        if syntree.count("(") != syntree.count(")"):
            raise NotImplementedError("Unbalanced number of parenthesis in line %s" % str(syntree))

        output_actions, output_terms = get_actions_and_terms(syntree, is_generative=True)

        if not is_valid_action_sequence(output_actions):
            logger.debug("invalid action seq in line %s" % str(output_actions))
            return None, None, None, "invalid"

        if len(output_terms) > max_words:  # 500:
            logger.debug("too long sentence in line %s" % str(output_terms))
            return None, None, None, "too_long"

        for i, a in enumerate(output_actions):
            if a == "NT(NML)":
                output_actions[i] = "NT(NP)"

        if tg_encoding:
            new_output_actions = []
            stack = []
            for action in output_actions:
                if action.startswith("NT("):
                    stack.append(action[3:-1])
                    new_output_actions.append(action)
                elif action == "REDUCE()":
                    nt = stack.pop()
                    new_output_actions.append("REDUCE({})".format(nt))
                    new_output_actions.append("REDUCE({})".format(nt))
                else:
                    new_output_actions.append(action)
            output_actions = new_output_actions

            if right_branching:
                output_actions = tg_to_right_branching(output_actions)

        return load_line, output_actions, output_terms, None
