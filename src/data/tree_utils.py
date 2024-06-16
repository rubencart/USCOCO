from typing import List, Tuple, Union, cast

from torch import Tensor


def extract_spans_from_string_tree(
    tree: str, contains_pos_tags: bool = True, include_length_1: bool = False
) -> List[Tuple[int, int]]:
    """
    Takes a str tree as input and computes a list of spans
    Caution: every '(' has to be followed by a space ' ', and every ')'
    has to be preceded by one!
    e.g.
    in: '( S ( NP ( DT A ) ( NN woman ) ) ( VP ( VBZ stands ) ( PP ( IN in )
        ( NP ( DT the ) ( NN dining )
        ( NN area ) ) ) ( PP ( IN at ) ( NP ( DT the ) ( NN table ) ) ) ) )',
        remove_pos_tags=True
    out: [(0, 0), (1, 1), (0, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (4, 6),
        (3, 6), (7, 7), (8, 8),
        (9, 9), (8, 9), (7, 9), (2, 9), (0, 9)]
    """
    answer = []
    stack: List[Union[str, Tuple[int, int], Tuple[str, str]]] = []
    items = tree.split()
    curr_index = 0
    for item in items:
        if contains_pos_tags and len(stack) > 0 and stack[-1] == ("(", "REMOVE_POS"):
            stack[-1] = "("
            continue
        if item == ")":
            pos = -1
            right_margin = stack[pos][1]
            left_margin = None
            while stack[pos] != "(":
                left_margin = stack[pos][0]
                pos -= 1
            assert left_margin is not None
            assert right_margin is not None
            stack = stack[:pos] + [(cast(int, left_margin), cast(int, right_margin))]
            if include_length_1 or left_margin != right_margin:
                answer.append((int(left_margin), int(right_margin)))
        elif item == "(":
            stack.append((item, "REMOVE_POS") if contains_pos_tags else "(")
        else:
            stack.append((curr_index, curr_index))
            curr_index += 1
    return answer


def extract_spans_and_tags(
    tokens: List[str], include_parens: bool = False
) -> List[Tuple[int, int, str]]:
    """
        Extracts Non-Terminal span indices (left incl, right excl)
        and corresponding constituent tags from a
            tree action sequence.
            Attention, indices may include NT() and REDUCE(),
            use with filter_non_words function below

    IN: ('[START]', 'NT(NP)', 'NT(NP)', 'ĠA', 'Ġliving', 'Ġarea',
    'REDUCE(NP)', 'REDUCE(NP)', 'NT(PP)', 'Ġwith',
        'NT(NP)', 'NT(NP)', 'Ġa', 'Ġtelevision', 'REDUCE(NP)',
        'REDUCE(NP)', 'Ġand', 'NT(NP)', 'Ġa', 'Ġtable',
        'REDUCE(NP)', 'REDUCE(NP)', 'REDUCE(NP)', 'REDUCE(NP)',
        'REDUCE(PP)', 'REDUCE(PP)', 'REDUCE(NP)', 'REDUCE(NP)')
    OUT:
        [(3, 6, 'NT(NP)'), (12, 14, 'NT(NP)'), (18, 20, 'NT(NP)'),
        (12, 20, 'NT(NP)'), (9, 20, 'NT(PP)'),
        (3, 20, 'NT(NP)')]
    >>> answer = extract_spans_and_tags(tokens)
    >>> print(' -- '.join([f'{tag} {tokens[left:right]}' for left, right, tag in answer]))
        NT(NP) ('ĠA', 'Ġliving', 'Ġarea')
     -- NT(NP) ('Ġa', 'Ġtelevision')
     -- NT(NP) ('Ġa', 'Ġtable')
     -- NT(NP) ('Ġa', 'Ġtelevision', 'REDUCE(NP)', 'REDUCE(NP)',
     'Ġand', 'NT(NP)', 'Ġa', 'Ġtable')
     -- NT(PP) ('Ġwith', 'NT(NP)', 'NT(NP)', 'Ġa', 'Ġtelevision',
     'REDUCE(NP)', 'REDUCE(NP)', 'Ġand', 'NT(NP)', 'Ġa', 'Ġtable')
     -- NT(NP) ('ĠA', 'Ġliving', 'Ġarea', 'REDUCE(NP)', 'REDUCE(NP)',
     'NT(PP)', 'Ġwith', 'NT(NP)', 'NT(NP)', 'Ġa', 'Ġtelevision', 'REDUCE(NP)',
     'REDUCE(NP)', 'Ġand', 'NT(NP)', 'Ġa', 'Ġtable')
    """
    answer = []
    stack: List[Union[str, Tuple[int, int], Tuple[str, str]]] = []

    for i, tok in enumerate(tokens):
        if "NT(" in tok:
            stack.append(tok)
            if include_parens:
                stack.append((i, i))
        elif "REDUCE(" in tok:
            if "REDUCE(" not in stack[-1]:
                if include_parens:
                    stack.append((i, i))
                stack.append(tok)
            else:
                reduce = stack.pop()
                if include_parens:
                    stack.append((i, i))

                nxt = stack.pop()
                right_margin, left_margin = nxt[1], None
                while "NT(" not in nxt:
                    left_margin = nxt[0]
                    nxt = stack.pop()
                assert left_margin is not None
                left_margin, right_margin = cast(int, left_margin), cast(int, right_margin)

                tag = nxt
                stack.append((left_margin, right_margin))
                answer.append((left_margin, right_margin + 1, tag))
        else:
            stack.append((i, i))

    return cast(List[Tuple[int, int, str]], answer)


def extract_spans_and_tags_other_tokenizer(
    tg_tokens: List[str], tokens: List[str]
) -> List[Tuple[int, int, str]]:
    """
        Extracts Non-Terminal span indices (left incl, right excl) and
            corresponding constituent tags for a sequence
            of `tokens` using a tree action sequence representing a constituency
            tree parse (`tg_tokens`).
            The indices index into `tokens`, which are assumed not to include tags etc.
            The words between tags in `tg_tokens` are assumed to have been tokenized by the
            same tokenizer as the
            words in `tokens`.

    IN: tg_tokens:
        ('[START]', 'NT(NP)', 'NT(NP)', 'ĠA', 'Ġliving', 'Ġarea', 'REDUCE(NP)',
        'REDUCE(NP)', 'NT(PP)', 'Ġwith',
        'NT(NP)', 'NT(NP)', 'Ġa', 'Ġtelevision', 'REDUCE(NP)', 'REDUCE(NP)',
        'Ġand', 'NT(NP)', 'Ġa', 'Ġtable',
        'REDUCE(NP)', 'REDUCE(NP)', 'REDUCE(NP)', 'REDUCE(NP)', 'REDUCE(PP)',
        'REDUCE(PP)', 'REDUCE(NP)', 'REDUCE(NP)')

        tokens:
        ['A', 'Ġliving', 'Ġarea', 'Ġwith', 'Ġa', 'Ġtelevision', 'Ġand', 'Ġa', 'Ġtable']

    OUT:
        [(0, 3, 'NT(NP)'), (4, 6, 'NT(NP)'), (7, 10, 'NT(NP)'), (4, 10, 'NT(NP)'),
        (3, 10, 'NT(PP)'),
        (0, 10, 'NT(NP)')]
    >>> answer = extract_spans_and_tags(tg_tokens, tokens)
    >>> print('\\n'.join([f'{t} {tokens[l:r]}' for l, r, t in answer]))
        NT(NP) ['A', 'Ġliving', 'Ġarea']
        NT(NP) ['Ġa', 'Ġtelevision']
        NT(NP) ['Ġa', 'Ġtable']
        NT(NP) ['Ġa', 'Ġtelevision', 'Ġand', 'Ġa', 'Ġtable']
        NT(PP) ['Ġwith', 'Ġa', 'Ġtelevision', 'Ġand', 'Ġa', 'Ġtable']
        NT(NP) ['A', 'Ġliving', 'Ġarea', 'Ġwith', 'Ġa', 'Ġtelevision', 'Ġand', 'Ġa', 'Ġtable']
    """
    answer = []
    answer_o = []
    stack: List[Union[Tuple[int, int, int, int], str]] = []
    j, end_i = 0, -1
    for i, tok in enumerate(tg_tokens):
        if "NT(" in tok:
            stack.append(tok)
        elif "REDUCE(" in tok:
            if "REDUCE(" not in stack[-1]:
                stack.append(tok)
            else:
                reduce = stack.pop()  # because TG has 2 reduces
                nxt = stack.pop()
                right_margin, rm_o, left_margin, lm_o = nxt[1], nxt[3], None, None
                while "NT(" not in nxt:
                    left_margin = nxt[0]
                    lm_o = nxt[2]
                    nxt = stack.pop()
                assert left_margin is not None
                tag = nxt
                stack.append((left_margin, right_margin, lm_o, rm_o))
                answer.append((left_margin, right_margin + 1, tag))
                answer_o.append((lm_o, rm_o + 1, tag))
        #                        means we covered it in last iteration
        elif "[START]" in tok or not tok.startswith("Ġ") or end_i >= i:
            continue
        else:
            # if next TG token does not start with a space,
            # join it with this token since they form 1 word
            end_i = _expand_no_space_right(i, tg_tokens)
            end_i, end_j = _expand_overlap(end_i, i, j, tg_tokens, tokens)

            while not _overlap_l_in_r(tokens[j : end_j + 1], tg_tokens[i : end_i + 1]):
                j += 1
                end_j = j

            end_i, end_j = _expand_overlap(end_i, i, j, tg_tokens, tokens)

            stack.append((i, end_i, j, end_j))
            j = end_j + 1

    return answer_o


def extract_spans_and_tags_rb_tokenizer(
    tg_tokens: List[str], rb_tokens: List[str], include_parens: bool = False
) -> List[Tuple[int, int, str]]:
    """
        Extracts Non-Terminal span indices (left incl, right excl) and
        corresponding constituent tags for a sequence
            of `tokens` using a tree action sequence representing a
            constituency tree parse (`tg_tokens`).
            The indices index into `tokens`, which are assumed not to
            include tags etc.
            The words between tags in `tg_tokens` are assumed to have
            been tokenized by the same tokenizer as the
            words in `tokens`.

    IN: tg_tokens:
        ('[START]', 'NT(NP)', 'NT(NP)', 'ĠA', 'Ġliving', 'Ġarea',
        'REDUCE(NP)', 'REDUCE(NP)', 'NT(PP)', 'Ġwith',
        'NT(NP)', 'NT(NP)', 'Ġa', 'Ġtelevision', 'REDUCE(NP)',
        'REDUCE(NP)', 'Ġand', 'NT(NP)', 'Ġa', 'Ġtable',
        'REDUCE(NP)', 'REDUCE(NP)', 'REDUCE(NP)', 'REDUCE(NP)',
        'REDUCE(PP)', 'REDUCE(PP)', 'REDUCE(NP)', 'REDUCE(NP)')

        rb_tokens:

    OUT:
    >>> answer = extract_spans_and_tags(tg_tokens, rb_tokens)
    >>> print('\\n'.join([f'{t} {rb_tokens[l:r]}' for l, r, t in answer]))
    """
    answer = list()
    answer_o = list()
    stack = list()
    j = 0
    for i, tok in enumerate(tg_tokens):
        if "NT(" in tok:
            stack.append(tok)
            if include_parens:
                stack.append((i, i, j, j))
            j += 1  # NT( also in right-branching
        elif "REDUCE(" in tok:
            if "REDUCE(" not in stack[-1]:
                # if include_parens: stack.append((i, i, j, j))
                stack.append(tok)
            else:
                _ = stack.pop()  # because TG has 2 reduces

                nxt = stack.pop()
                right_margin, rm_o, left_margin, lm_o = nxt[1], nxt[3], None, None
                while "NT(" not in nxt:
                    left_margin = nxt[0]
                    lm_o = nxt[2]
                    nxt = stack.pop()
                assert left_margin is not None
                tag = nxt
                stack.append((left_margin, right_margin, lm_o, rm_o))
                answer.append((left_margin, right_margin + 1, tag))
                answer_o.append((lm_o, rm_o + 1, tag))
        else:
            stack.append((i, i, j, j))
            j += 1

    return answer_o


def _expand_overlap(end_i, i, j, tg_tokens, tokens):
    changed = True
    end_j = j
    while changed:
        changed = False
        while end_j < len(tokens) - 1 and _overlap_l_in_r(
            tokens[j : end_j + 2], tg_tokens[i : end_i + 1]
        ):
            end_j += 1
            changed = True
        while end_i < len(tg_tokens) - 1 and _overlap_l_in_r(
            tg_tokens[i : end_i + 2], tokens[j : end_j + 1]
        ):
            end_i += 1
            changed = True
    return end_i, end_j


def _expand_no_space_right(i, tg_tokens):
    end_i = i
    while not (
        tg_tokens[end_i + 1].startswith("Ġ")
        or tg_tokens[end_i + 1].startswith("NT(")
        or tg_tokens[end_i + 1].startswith("REDUCE(")
    ):
        end_i += 1
    return end_i


def _drop_space_start(token: str) -> str:
    if token.startswith("Ġ"):
        return token[1:]
    else:
        return token


def _join_wo_space(tokens: List[str]) -> str:
    return "".join([_drop_space_start(t) for t in tokens])


def _overlap_l_in_r(tokens: List[str], tg_tokens: List[str]) -> bool:
    joined_tg = _join_wo_space(tg_tokens)
    joined = _join_wo_space(tokens)

    overlap = False
    if joined in joined_tg:
        overlap = True
    elif (
        joined[-1] == "n"
        and joined[:-1] in joined_tg
        and _drop_space_start(tg_tokens[-1]).endswith("n")
    ):
        overlap = True
    elif (
        joined[-3:] == "not"
        and joined[:-3] in joined_tg
        and _drop_space_start(tg_tokens[-1]).endswith("not")
    ):
        overlap = True
    return overlap


def extract_spans_and_tags_plm_tokenizer(
    tg_tokens: List[str], include_parens: bool = False
) -> List[Tuple[int, int, str]]:
    """
        Extracts Non-Terminal span indices (left incl, right excl) and
        corresponding constituent tags for a sequence
            of `tokens` using a tree action sequence representing a
            constituency tree parse (`tg_tokens`).
            The indices index into `tokens`, which are assumed not
            to include tags etc.
            The words between tags in `tg_tokens` are assumed to have
            been tokenized by the same tokenizer as the
            words in `tokens`.

    IN: tg_tokens:
        ('[START]', 'NT(NP)', 'NT(NP)', 'ĠA', 'Ġliving', 'Ġarea', 'REDUCE(NP)',
        'REDUCE(NP)', 'NT(PP)', 'Ġwith',
        'NT(NP)', 'NT(NP)', 'Ġa', 'Ġtelevision', 'REDUCE(NP)', 'REDUCE(NP)',
        'Ġand', 'NT(NP)', 'Ġa', 'Ġtable',
        'REDUCE(NP)', 'REDUCE(NP)', 'REDUCE(NP)', 'REDUCE(NP)', 'REDUCE(PP)',
        'REDUCE(PP)', 'REDUCE(NP)', 'REDUCE(NP)')

        plm_tokens:

    OUT:

    >>> answer = extract_spans_and_tags(tg_tokens, tokens)
    >>> print('\\n'.join([f'{t} {tokens[l:r]}' for l, r, t in answer]))
        NT(NP) ['A', 'Ġliving', 'Ġarea']
        NT(NP) ['Ġa', 'Ġtelevision']
        NT(NP) ['Ġa', 'Ġtable']
        NT(NP) ['Ġa', 'Ġtelevision', 'Ġand', 'Ġa', 'Ġtable']
        NT(PP) ['Ġwith', 'Ġa', 'Ġtelevision', 'Ġand', 'Ġa', 'Ġtable']
        NT(NP) ['A', 'Ġliving', 'Ġarea', 'Ġwith', 'Ġa', 'Ġtelevision', 'Ġand', 'Ġa', 'Ġtable']
    """
    answer = list()
    answer_o = list()
    stack = list()
    j = 0
    for i, tok in enumerate(tg_tokens):
        if "NT(" in tok:
            stack.append(tok)
            if include_parens:
                stack.append((i, i, j, j))
            j += 1  # NT( also in right-branching
        elif "REDUCE(" in tok:
            if "REDUCE(" not in stack[-1]:
                if include_parens:
                    stack.append((i, i, j, j))
                j += 1
                stack.append(tok)
            else:
                _ = stack.pop()  # because TG has 2 reduces

                nxt = stack.pop()
                right_margin, rm_o, left_margin, lm_o = nxt[1], nxt[3], None, None
                while "NT(" not in nxt:
                    left_margin = nxt[0]
                    lm_o = nxt[2]
                    nxt = stack.pop()
                assert left_margin is not None
                tag = nxt
                stack.append((left_margin, right_margin, lm_o, rm_o))
                answer.append((left_margin, right_margin + 1, tag))
                answer_o.append((lm_o, rm_o + 1, tag))
        elif "[START]" in tok:
            j += 1
        else:
            stack.append((i, i, j, j))
            j += 1

    return answer_o


def tg_to_right_branching(actions):
    new_output_actions = []
    reduce_outputs = []
    for action in actions:
        if action.startswith("NT("):
            nt = action[3:-1]
            new_output_actions.append(action)
            reduce_outputs.append("REDUCE({})".format(nt))
            reduce_outputs.append("REDUCE({})".format(nt))
        elif action.startswith("REDUCE("):
            pass
        else:
            new_output_actions.append(action)
    reduce_outputs.reverse()
    new_output_actions.extend(reduce_outputs)
    return new_output_actions
