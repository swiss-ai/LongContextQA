"""Synthetic needle-in-haystack dataset generation.

Builds multi-turn dialogues over a list of small integer-keyed dictionaries.
On each user turn the assistant must retrieve a different "gold" key planted
somewhere in the listing. Outputs are tokenized via the model's chat template
and written as Megatron indexed datasets.
"""
import argparse
import multiprocessing as mp
import os
import random
import resource
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

from megatron.core.datasets.indexed_dataset import IndexedDatasetBuilder

DEFAULT_TOKENIZER = "swiss-ai/Apertus-8B-Instruct-2509"
SYSTEM_MSG = "You are a helpful assistant. Be concise."

PROMPT_FIRST = (
    "Do a task using the list of dictionaries below.\n\n"
    "{dictionaries}\n\n"
    "Above is a list of dictionaries such that each key and value is an integer. "
    "Report the value of key {key} and the dictionary it is in. "
    "Answer in the following template:\n"
    "The value of key {key} is <fill-in-value> and it is in Dictionary [<fill-in-dictionary-name>]."
)
PROMPT_FOLLOWUP = (
    "Now report the value of key {key} and the dictionary it is in, "
    "using the same template as before."
)
ANSWER = "The value of key {key} is {value} and it is in Dictionary [{name}]."


@dataclass
class DataConfig:
    num_dicts: int = 85
    gold_dict_size: int = 3
    dict_size_range: Tuple[int, int] = (3, 4)
    subkey_size_range: Tuple[int, int] = (3, 4)
    val_size_range: Tuple[int, int] = (3, 4)


@dataclass
class TaskConfig:
    min_token: int = 0
    max_token: int = 16384
    name_random: bool = False
    num_turns: int = 1


_WORKER_TOKENIZER = None


def _worker_init(tokenizer_name: str) -> None:
    """Pool initializer — runs once per worker after fork."""
    global _WORKER_TOKENIZER
    _WORKER_TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_name)


def _rand_int(rng: Tuple[int, int]) -> int:
    """Random integer whose digit count is drawn uniformly from rng (inclusive)."""
    n = random.randint(*rng)
    return random.randint(10 ** (n - 1), 10 ** n - 1)


def _fresh_kv(cfg: DataConfig, bad_keys: set, bad_vals: set) -> Tuple[Tuple[int], int]:
    """Sample a (key, value) pair avoiding any forbidden keys/values."""
    while True:
        k = (_rand_int(cfg.subkey_size_range),)
        v = _rand_int(cfg.val_size_range)
        if k not in bad_keys and v not in bad_vals:
            return k, v


def _build_dict(cfg: DataConfig, size: int, gold_kv, bad_keys: set, bad_vals: set) -> dict:
    """Build a dict of given size; if gold_kv is provided, place it at a random index."""
    d: dict = {}
    gold_pos = random.randint(0, size - 1) if gold_kv is not None else -1
    for j in range(size):
        if j == gold_pos:
            d[gold_kv[0]] = gold_kv[1]
        else:
            k, v = _fresh_kv(cfg, bad_keys | d.keys(), bad_vals)
            d[k] = v
    return d



def _fmt_dict(d: dict) -> str:
    return "{" + ", ".join(f"{k[0]}: {v}" for k, v in d.items()) + "}"


def _fmt_listing(dicts: List[dict], names: Sequence[int]) -> str:
    return "\n".join(f"Dictionary [{names[i]}] {_fmt_dict(d)}" for i, d in enumerate(dicts))


class Task:
    """One multi-turn needle-in-haystack example."""

    OVERHEAD_PER_TURN = 200  # rough token reserve for question + answer + chat-template wrapping

    def __init__(self, cfg: DataConfig, task_cfg: TaskConfig, tokenizer):
        self.cfg = cfg
        self.tcfg = task_cfg
        self.tokenizer = tokenizer
        self._build()

    def _count(self, text: str) -> int:
        if self.tokenizer is None:
            return int(len(text.split()) / 0.75)
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def _listing_tokens(self, gold_dicts: List[dict], ng_dicts: List[dict]) -> int:
        names = list(range(1, len(gold_dicts) + len(ng_dicts) + 1))
        return self._count(_fmt_listing(gold_dicts + ng_dicts, names))

    def _build(self) -> None:
        cfg, tcfg = self.cfg, self.tcfg
        n_turns = tcfg.num_turns
        if not 1 <= n_turns <= cfg.num_dicts:
            raise ValueError(f"need 1 <= num_turns({n_turns}) <= num_dicts({cfg.num_dicts})")
        slack = self.OVERHEAD_PER_TURN * n_turns + 50
        if tcfg.max_token <= slack:
            raise ValueError(f"max_token={tcfg.max_token} too small for num_turns={n_turns}")
        target = random.randint(max(0, tcfg.min_token - slack), tcfg.max_token - slack)

        for _ in range(100):
            # 1) Pick num_turns mutually distinct gold (key, value) pairs.
            bad_keys: set = set()
            bad_vals: set = set()
            gold_kvs: List[Tuple[Tuple[int], int]] = []
            for _ in range(n_turns):
                k, v = _fresh_kv(cfg, bad_keys, bad_vals)
                gold_kvs.append((k, v))
                bad_keys.add(k)
                bad_vals.add(v)

            def new_ng():
                return _build_dict(cfg, random.randint(*cfg.dict_size_range), None, bad_keys, bad_vals)

            # 2) Build gold dicts (one per turn) and the initial distractor pool.
            gold_dicts = [_build_dict(cfg, cfg.gold_dict_size, kv, bad_keys, bad_vals) for kv in gold_kvs]
            ng_dicts = [new_ng() for _ in range(cfg.num_dicts - n_turns)]

            # 3) Pad with extra distractors to reach the min_token target.
            listing_tokens = self._listing_tokens(gold_dicts, ng_dicts)
            if tcfg.min_token > 0 and listing_tokens < target:
                tpd = max(1, self._count(f"Dictionary [999] {_fmt_dict(new_ng())}\n"))
                ng_dicts.extend(new_ng() for _ in range(max(0, (target - listing_tokens) // tpd)))
                listing_tokens = self._listing_tokens(gold_dicts, ng_dicts)
                while listing_tokens < target:
                    ng_dicts.append(new_ng())
                    listing_tokens += tpd            # cheap incremental estimate
                listing_tokens = self._listing_tokens(gold_dicts, ng_dicts)  # exact recount

            # 4) Reject if we blew the budget; retry.
            if listing_tokens >= tcfg.max_token - slack:
                continue

            # 5) Commit.
            names = list(range(1, len(gold_dicts) + len(ng_dicts) + 1))
            if tcfg.name_random:
                random.shuffle(names)
            self.gold_dicts = gold_dicts
            self.ng_dicts = ng_dicts
            self.gold_kvs = gold_kvs
            self.names = names
            return
        raise RuntimeError("Could not build prompt in token range after 100 tries.")

    @property
    def num_dicts(self) -> int:
        return len(self.gold_dicts) + len(self.ng_dicts)

    def get_messages(self, gold_positions: Sequence[int]) -> list:
        """Build the multi-turn messages.

        gold_positions: distinct indices in [0, num_dicts), one per turn,
        specifying where each turn's gold dict sits in the listing.
        """
        n_turns = self.tcfg.num_turns
        if len(gold_positions) != n_turns or len(set(gold_positions)) != n_turns:
            raise ValueError(f"need {n_turns} distinct gold positions, got {gold_positions!r}")

        # Place gold dicts at the requested positions; fill the rest with distractors in order.
        listing: List[dict] = [None] * self.num_dicts
        for i, pos in enumerate(gold_positions):
            listing[pos] = self.gold_dicts[i]
        ng_iter = iter(self.ng_dicts)
        for i in range(self.num_dicts):
            if listing[i] is None:
                listing[i] = next(ng_iter)
        listing_str = _fmt_listing(listing, self.names)

        messages = [{"role": "system", "content": SYSTEM_MSG}]
        for t in range(n_turns):
            (key,), value = self.gold_kvs[t]
            content = (PROMPT_FIRST.format(dictionaries=listing_str, key=key)
                       if t == 0 else PROMPT_FOLLOWUP.format(key=key))
            messages.append({"role": "user", "content": content})
            messages.append({"role": "assistant",
                             "content": ANSWER.format(key=key, value=value,
                                                      name=self.names[gold_positions[t]])})
        return messages



def tokenize_example(messages: list, tokenizer) -> np.ndarray:
    """Apply the chat template and append </s> as the document-final separator.
    """
    ids = list(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False))
    sep_id = tokenizer.convert_tokens_to_ids("</s>")
    if sep_id is not None and sep_id != tokenizer.unk_token_id and (not ids or ids[-1] != sep_id):
        ids.append(sep_id)
    return np.asarray(ids, dtype=np.int32)



def sample_gold_positions(num: int, num_turns: int, num_dicts: int,
                          frac_0: float, frac_top5: float) -> List[List[int]]:
    """Per example, return num_turns distinct gold positions with optional bias.

    frac_0  → forced to position 0.
    frac_top5 → forced into the top 5% of positions.
    Otherwise uniform in [0, num_dicts).
    Collisions across turns within an example are re-rolled uniformly.
    """
    top5_cap = max(1, int(num_dicts * 0.05))
    out: List[List[int]] = []
    for _ in range(num):
        used: set = set()
        picks: List[int] = []
        for _ in range(num_turns):
            r = random.random()
            if r < frac_0:
                p = 0
            elif r < frac_0 + frac_top5:
                p = random.randint(0, top5_cap - 1)
            else:
                p = random.randint(0, num_dicts - 1)
            while p in used:
                p = random.randint(0, num_dicts - 1)
            used.add(p)
            picks.append(p)
        out.append(picks)
    return out



def _worker_generate_shard(args: Tuple) -> str:
    worker_idx, gold_positions, base_seed, data_cfg, task_cfg, prefix = args
    seed = base_seed + worker_idx * 1_000_003
    random.seed(seed)
    np.random.seed(seed % (2 ** 31))
    builder = IndexedDatasetBuilder(prefix + ".bin", dtype=np.int32)
    for picks in gold_positions:
        task = Task(data_cfg, task_cfg, _WORKER_TOKENIZER)
        # picks were sampled in [0, cfg.num_dicts) but min_token padding may have grown
        # the listing, so picks remain valid; clamp+resample defensively in case of dup.
        clamped = [min(p, task.num_dicts - 1) for p in picks]
        if len(set(clamped)) != len(clamped):
            clamped = random.sample(range(task.num_dicts), task_cfg.num_turns)
        ids = tokenize_example(task.get_messages(clamped), _WORKER_TOKENIZER)
        builder.add_document(ids, [len(ids)])
    builder.finalize(prefix + ".idx")
    return prefix


def _raise_fd_limit() -> None:
    _, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))


def build_dataset_parallel(gold_positions: List[List[int]], base_seed: int,
                           tokenizer_name: str, data_cfg: DataConfig, task_cfg: TaskConfig,
                           out_prefix: str, num_workers: int, tmp_dir: str, desc: str) -> None:
    if not gold_positions:
        return
    os.makedirs(tmp_dir, exist_ok=True)
    _raise_fd_limit()
    num_workers = max(1, min(num_workers, len(gold_positions)))
    chunks = np.array_split(np.arange(len(gold_positions)), num_workers)
    worker_args = [
        (i, [gold_positions[k] for k in idx], base_seed,
         data_cfg, task_cfg, os.path.join(tmp_dir, f"shard_{i:05d}"))
        for i, idx in enumerate(chunks) if len(idx) > 0
    ]
    ctx = mp.get_context("fork")
    shard_prefixes: List[str] = []
    with ctx.Pool(processes=len(worker_args), initializer=_worker_init,
                  initargs=(tokenizer_name,)) as pool:
        for p in tqdm(pool.imap(_worker_generate_shard, worker_args),
                      total=len(worker_args), desc=f"{desc} shards"):
            shard_prefixes.append(p)
    final = IndexedDatasetBuilder(out_prefix + ".bin", dtype=np.int32)
    for p in shard_prefixes:
        final.add_index(p)
    final.finalize(out_prefix + ".idx")
    for p in shard_prefixes:
        for ext in (".idx", ".bin"):
            try:
                os.remove(p + ext)
            except OSError:
                pass



def main(seed: int, num_train: int, num_eval: int, frac_0: float, frac_top5: float,
         alias: str, out_dir: str, tokenizer: str, num_workers: int,
         num_dicts: int, min_token: int, max_token: int, num_turns: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    data_cfg = DataConfig(num_dicts=num_dicts)
    task_cfg = TaskConfig(min_token=min_token, max_token=max_token, num_turns=num_turns)

    train_gold = sample_gold_positions(num_train, num_turns, num_dicts, frac_0, frac_top5)
    eval_gold = sample_gold_positions(num_eval, num_turns, num_dicts, frac_0, frac_top5)

    os.makedirs(out_dir, exist_ok=True)
    tmp_root = os.path.join(out_dir, "_tmp_shards")
    build_dataset_parallel(train_gold, seed, tokenizer, data_cfg, task_cfg,
                           f"{out_dir}/{alias}_train", num_workers,
                           os.path.join(tmp_root, "train"), "Train")
    build_dataset_parallel(eval_gold, seed + 99, tokenizer, data_cfg, task_cfg,
                           f"{out_dir}/{alias}_eval", max(1, num_workers // 4),
                           os.path.join(tmp_root, "eval"), "Eval")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-train", type=int, default=350)
    parser.add_argument("--num-eval", type=int, default=150)
    parser.add_argument("--frac-0", type=float, default=0.05)
    parser.add_argument("--frac-top5", type=float, default=0.03)
    parser.add_argument("--alias", type=str, default="synthetic_needle")
    parser.add_argument("--out-dir", type=str, default="./dataset")
    parser.add_argument("--tokenizer", type=str, default=DEFAULT_TOKENIZER)
    parser.add_argument("--num-workers", type=int, default=max(1, mp.cpu_count() - 1))
    parser.add_argument("--num-dicts", type=int, default=85)
    parser.add_argument("--min-token", type=int, default=4096)
    parser.add_argument("--max-token", type=int, default=32768)
    parser.add_argument("--num-turns", type=int, default=10)
    args = parser.parse_args()
    main(**vars(args))