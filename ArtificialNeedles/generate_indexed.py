"""
Synthetic needle-in-a-haystack data generator — wtemplate variant.
Saves output as Megatron MMap indexed datasets (.idx/.bin pairs).

Each document is tokenized with the chat template and stored as:
  <bos> <chat-formatted tokens> <eos>

Output files (in --out-dir):
  <alias>_seed<N>_train_<num_train>.idx / .bin
  <alias>_seed<N>_eval_<num_eval>.idx  / .bin

Token-window control
--------------------
--min-token / --max-token define the inclusive-lower / exclusive-upper bounds
on the prompt length (in tokens, before chat-template wrapping).

The _build() method grows the distractor list one dictionary at a time until
the prompt exceeds min_token, then accepts it only if it also stays below
max_token.  This guarantees every sample lands in [min_token, max_token).

If --min-token is 0 (default) the old behaviour is preserved: any prompt
that fits under max_token is accepted immediately.

Usage:
  python generate_wtemplate_indexed.py --seed 0
  python generate_wtemplate_indexed.py --seed 0 --min-token 16384 --max-token 32768
  python generate_wtemplate_indexed.py --seed 0 --num-train 122070 --num-workers 15
"""

import argparse
import copy
import os
import random
import resource
import warnings
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import multiprocessing as mp

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

from megatron.core.datasets.indexed_dataset import IndexedDatasetBuilder

DEFAULT_TOKENIZER = "swiss-ai/Apertus-8B-Instruct-2509"


_WORKER_TOKENIZER = None


def _worker_init(tokenizer_name: str) -> None:
    """Pool initializer — runs once per worker process after fork."""
    global _WORKER_TOKENIZER
    _WORKER_TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_name)


PROMPT_TEMPLATE = (
    "Do a task using the list of dictionaries below.\n\n"
    "{disctionaries}\n\n"
    "Above is a list of dictionaries such that each key and value is an integer. "
    "Report the value of key {gold_key_str} and the dictionary it is in. "
    "Answer in the following template:\n"
    "The value of key {gold_key_str} is <fill-in-value> and it is in Dictionary [<fill-in-dictionary-name>]."
)

ANSWER_TEMPLATE = (
    "The value of key {gold_key} is {gold_value} and it is in Dictionary [{gold_dict_name}]."
)


@dataclass
class DataConfig:
    # num_dicts is now a *starting point* / minimum number of distractors.
    # When min_token > 0, _build() will add more distractors until the prompt
    # crosses min_token, so this effectively becomes a floor.
    num_dicts: int = 85
    gold_dict_size: int = 3
    dict_size_range: tuple = (3, 4)
    gold_key_idx: int = -1          # -1 → random
    subkey_size_range: tuple = (3, 4)
    val_size_range: tuple = (3, 4)
    subkey_param: str = "numerical"
    val_param: str = "numerical"


@dataclass
class TaskConfig:
    min_token: int = 0        # 0 = no lower bound (old behaviour)
    max_token: int = 16384
    name_random: bool = False


def _generate_val(val_size_range: tuple) -> int:
    size = random.randint(*val_size_range)
    return random.randint(10 ** (size - 1), 10 ** size - 1)


def _generate_subkey(subkey_size_range: tuple) -> int:
    size = random.randint(*subkey_size_range)
    return random.randint(10 ** (size - 1), 10 ** size - 1)


def _generate_ng_kv(gold_key, gold_val, subkey_size_range, val_size_range) -> Tuple:
    key = (_generate_subkey(subkey_size_range),)
    val = _generate_val(val_size_range)
    while key == gold_key or val == gold_val:
        key = (_generate_subkey(subkey_size_range),)
        val = _generate_val(val_size_range)
    return key, val


def _build_ng_dict(dict_size, gold_key, gold_val, subkey_size_range, val_size_range) -> dict:
    d = {}
    for _ in range(dict_size):
        k, v = _generate_ng_kv(gold_key, gold_val, subkey_size_range, val_size_range)
        d[k] = v
    return d


def _new_ng_dict(cfg: DataConfig, gold_key, gold_val) -> dict:
    """Convenience: generate one fresh distractor dict."""
    return _build_ng_dict(
        random.randint(*cfg.dict_size_range),
        gold_key, gold_val,
        cfg.subkey_size_range, cfg.val_size_range,
    )


def build_dicts(cfg: DataConfig):
    """Return (gold_dict, ng_dict_lst, gold_key, gold_val, gold_key_idx).

    ng_dict_lst contains exactly cfg.num_dicts - 1 distractors.
    """
    gold_key_idx = (
        cfg.gold_key_idx if cfg.gold_key_idx != -1
        else random.randint(0, cfg.gold_dict_size - 1)
    )

    gold_key = (_generate_subkey(cfg.subkey_size_range),)
    gold_val = _generate_val(cfg.val_size_range)

    gold_dict = {}
    for j in range(cfg.gold_dict_size):
        if j == gold_key_idx:
            gold_dict[gold_key] = gold_val
        else:
            k, v = _generate_ng_kv(gold_key, gold_val, cfg.subkey_size_range, cfg.val_size_range)
            gold_dict[k] = v

    ng_dict_lst = [
        _new_ng_dict(cfg, gold_key, gold_val)
        for _ in range(cfg.num_dicts - 1)
    ]

    return gold_dict, ng_dict_lst, gold_key, gold_val, gold_key_idx

def _key_str(gold_key: tuple) -> str:
    s = str(gold_key)
    s = s.replace("(", "").replace(",)", "")
    return s


def _dict_str(d: dict) -> str:
    s = str(d)
    s = s.replace("(", "").replace(",)", "")
    return s


def build_prompt(dict_lst: List[dict], gold_key: tuple, name_lst: List[int]) -> str:
    formatted = [
        f"Dictionary [{name_lst[i]}] {_dict_str(d)}"
        for i, d in enumerate(dict_lst)
    ]
    return PROMPT_TEMPLATE.format(
        disctionaries="\n".join(formatted),
        gold_key_str=_key_str(gold_key),
    )


def build_answer(gold_key: tuple, gold_val, gold_dict_name: int) -> str:
    return ANSWER_TEMPLATE.format(
        gold_key=_key_str(gold_key),
        gold_value=str(gold_val),
        gold_dict_name=gold_dict_name,
    )


class Task:
    def __init__(self, cfg: DataConfig, task_cfg: TaskConfig, tokenizer):
        self.cfg = cfg
        self.task_cfg = task_cfg
        self.tokenizer = tokenizer
        self._build()

    def _token_count(self, text: str) -> int:
        if self.tokenizer is None:
            return int(len(text.split()) / 0.75)
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def _build(self):
        """
        Build a prompt that satisfies [min_token, max_token).

        Strategy
        --------
        1.  Generate the gold dict + cfg.num_dicts-1 initial distractors.
        2.  If min_token > 0 and the prompt is still too short, keep appending
            fresh distractor dicts one at a time until the prompt crosses
            min_token.  The name_lst grows in tandem so indices stay valid.
        3.  Accept only if the final token count is also < max_token - 100.
            If we overshot max_token while trying to reach min_token, discard
            and retry from scratch (up to 1000 times total).

        When min_token == 0 the loop body is identical to the old behaviour:
        accept the first candidate that fits under max_token.
        """
        for _ in range(1000):
            gold_dict, ng_dict_lst, gold_key, gold_val, _ = build_dicts(self.cfg)

            # Working list: gold is always inserted at gold_dict_idx later;
            # here we just need to measure the base prompt size.
            all_dicts = [gold_dict] + ng_dict_lst
            name_lst = list(range(1, len(all_dicts) + 1))

            # --- grow until min_token is satisfied ---
            while self.task_cfg.min_token > 0:
                prompt = build_prompt(all_dicts, gold_key, name_lst)
                if self._token_count(prompt) >= self.task_cfg.min_token:
                    break
                # Add one more distractor and extend the name list
                all_dicts.append(_new_ng_dict(self.cfg, gold_key, gold_val))
                name_lst.append(len(name_lst) + 1)

            # --- check upper bound ---
            prompt = build_prompt(all_dicts, gold_key, name_lst)
            tok = self._token_count(prompt)
            if tok >= self.task_cfg.max_token - 100:
                # Overshot: retry entirely
                continue
            if self.task_cfg.min_token > 0 and tok < self.task_cfg.min_token:
                # Shouldn't normally happen, but guard anyway
                continue

            # Accept
            self.gold_dict = gold_dict
            # ng_dict_lst = everything except the gold dict
            self.ng_dict_lst = all_dicts[1:]
            self.gold_key = gold_key
            self.gold_val = gold_val
            if self.task_cfg.name_random:
                random.shuffle(name_lst)
            self.name_lst = name_lst
            return

        raise RuntimeError(
            f"Could not build a prompt in [{self.task_cfg.min_token}, "
            f"{self.task_cfg.max_token}) tokens after 1000 tries. "
            "Consider widening the token window or adjusting num_dicts."
        )

    def get_prompt(self, gold_dict_idx: int) -> str:
        dict_lst = copy.deepcopy(self.ng_dict_lst)
        dict_lst.insert(gold_dict_idx, copy.deepcopy(self.gold_dict))
        prompt = build_prompt(dict_lst, self.gold_key, self.name_lst)
        if self._token_count(prompt) >= self.task_cfg.max_token - 30:
            raise ValueError("Prompt exceeds token budget.")
        return prompt

    def get_answer(self, gold_dict_idx: int) -> str:
        # gold_dict_idx is always valid: it is drawn from
        # 0 .. len(name_lst)-1 by apply_position_bias (which uses num_dicts
        # from the *final* list length, set correctly in main()).
        gold_dict_name = self.name_lst[gold_dict_idx]
        return build_answer(self.gold_key, self.gold_val, gold_dict_name)

    def get_messages(self, gold_dict_idx: int) -> list:
        return [
            {"role": "system", "content": "You are a helpful assistant. Be concise."},
            {"role": "user", "content": self.get_prompt(gold_dict_idx)},
            {"role": "assistant", "content": self.get_answer(gold_dict_idx)},
        ]

    @property
    def num_dicts(self) -> int:
        """Total number of dicts in this task (gold + distractors)."""
        return 1 + len(self.ng_dict_lst)


def tokenize_example(messages: list, tokenizer) -> np.ndarray:
    """
    Apply the chat template, then tokenize to a numpy array with:
      [bos_token_id, ...content token ids..., eos_token_id]
    """
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        add_special_tokens=False,
    )
    if tokenizer.bos_token and text.startswith(tokenizer.bos_token):
        text = text[len(tokenizer.bos_token):]

    token_ids = tokenizer.encode(text, add_special_tokens=False)

    bos = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
    eos = [2]
    token_ids = bos + token_ids + eos

    return np.array(token_ids, dtype=np.int32)


def _idx_select_0(n: int, frac: float) -> List[int]:
    if frac > 0.05:
        warnings.warn("frac_0 > 0.05")
    return np.random.choice(n, int(n * frac), replace=False).tolist()


def _idx_select_top5(n: int, frac: float, exclude: Optional[List[int]] = None) -> List[int]:
    if frac > 0.05:
        warnings.warn("frac_top5 > 0.05")
    pool = [i for i in range(n) if exclude is None or i not in exclude]
    return np.random.choice(pool, int(n * frac), replace=False).tolist()


def apply_position_bias(
    gold_idx_lst: List[int],
    num: int,
    num_dicts: int,
    frac_0: float = 0.05,
    frac_top5: float = 0.03,
) -> List[int]:
    """
    Nudge a fraction of samples so their gold dict sits near the start.

    num_dicts here is the *maximum* number of dicts any task can have — used
    only to bound the top-5% cap.  Each individual task's gold_dict_idx is
    already bounded to 0..task.num_dicts-1 when generated; the bias just
    moves some of them closer to 0.
    """
    new_lst = copy.deepcopy(gold_idx_lst)

    sel_0 = _idx_select_0(num, frac_0) if frac_0 > 0 else []
    for i in sel_0:
        new_lst[i] = 0

    sel_top5 = _idx_select_top5(num, frac_top5, sel_0) if frac_top5 > 0 else []
    # Cap relative to num_dicts (not num samples) so indices stay in range.
    top5_cap = max(1, int(num_dicts * 0.05))
    for i in sel_top5:
        new_lst[i] = random.randint(0, top5_cap - 1)

    return new_lst


def _worker_generate_shard(args: Tuple) -> str:
    """
    Runs inside a forked worker that already has the tokenizer loaded via
    _worker_init.  Generates `count` examples and writes them to a temporary
    Megatron indexed dataset.  Returns the path prefix (without extension).
    """
    worker_idx, count, gold_idx_slice, base_seed, data_cfg, task_cfg, prefix = args

    global _WORKER_TOKENIZER
    tokenizer = _WORKER_TOKENIZER

    # Reproducible but distinct seed per worker
    seed = base_seed + worker_idx * 1_000_003
    random.seed(seed)
    np.random.seed(seed % (2 ** 31))

    builder = IndexedDatasetBuilder(prefix + ".bin", dtype=np.int32)

    for i in range(count):
        task = Task(copy.deepcopy(data_cfg), task_cfg, tokenizer)

        # gold_dict_idx must be in [0, task.num_dicts - 1].
        # The pre-computed index may come from a DataConfig with a smaller
        # num_dicts; clamp it to the actual task length just in case.
        gold_dict_idx = min(gold_idx_slice[i], task.num_dicts - 1)

        messages = task.get_messages(gold_dict_idx)
        tokens = tokenize_example(messages, tokenizer)
        builder.add_document(tokens, [len(tokens)])

    builder.finalize(prefix + ".idx")
    return prefix

def _raise_fd_limit() -> None:
    """Raise the open-file-descriptor soft limit to the hard limit."""
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft < hard:
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
        print(f"Raised fd limit: {soft} → {hard}")
    else:
        print(f"fd limit already at maximum ({hard})")


def build_dataset_parallel(
    num: int,
    gold_idx_lst: List[int],
    base_seed: int,
    tokenizer_name: str,
    data_cfg: DataConfig,
    task_cfg: TaskConfig,
    out_prefix: str,
    num_workers: int,
    tmp_dir: str,
    desc: str = "Generating",
) -> None:
    """
    Distribute `num` examples across `num_workers` forked processes.

    Each worker writes its own temporary shard; the main process merges all
    shards into the final output using IndexedDatasetBuilder.add_index().

    Why fork (not spawn)
    --------------------
    spawn starts a fresh interpreter in every worker and triggers a
    simultaneous mass-import of transformers across all workers, instantly
    exhausting the OS open-file-descriptor limit (~1024).  fork inherits the
    parent's already-imported modules, so each worker only needs to load the
    tokenizer weights once via _worker_init — sequentially, not in parallel.
    """
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)

    _raise_fd_limit()

    # Split gold indices evenly across workers
    chunk_sizes = [num // num_workers] * num_workers
    for i in range(num % num_workers):
        chunk_sizes[i] += 1

    gold_slices, offset = [], 0
    for sz in chunk_sizes:
        gold_slices.append(gold_idx_lst[offset: offset + sz])
        offset += sz

    worker_args = [
        (
            worker_idx,
            chunk_sizes[worker_idx],
            gold_slices[worker_idx],
            base_seed,
            data_cfg,
            task_cfg,
            os.path.join(tmp_dir, f"shard_{worker_idx:05d}"),
        )
        for worker_idx in range(num_workers)
        if chunk_sizes[worker_idx] > 0
    ]

    print(f"[{desc}] Spawning {len(worker_args)} workers for {num} examples …")

    ctx = mp.get_context("fork")
    shard_prefixes: List[str] = []

    with ctx.Pool(
        processes=num_workers,
        initializer=_worker_init,
        initargs=(tokenizer_name,),
    ) as pool:
        for prefix in tqdm(
            pool.imap(_worker_generate_shard, worker_args),
            total=len(worker_args),
            desc=f"{desc} shards",
            unit="shard",
        ):
            shard_prefixes.append(prefix)

    # Merge shards into final dataset
    print(f"[{desc}] Merging {len(shard_prefixes)} shards → {out_prefix} …")
    final_builder = IndexedDatasetBuilder(out_prefix + ".bin", dtype=np.int32)
    for prefix in tqdm(shard_prefixes, desc=f"{desc} merge", unit="shard"):
        final_builder.add_index(prefix)
    final_builder.finalize(out_prefix + ".idx")

    # Clean up temp shards
    for prefix in shard_prefixes:
        for ext in (".idx", ".bin"):
            try:
                os.remove(prefix + ext)
            except OSError:
                pass
    try:
        os.rmdir(tmp_dir)
    except OSError:
        pass

    print(f"[{desc}] Done — {num} documents → {out_prefix}.{{idx,bin}}")


def main(
    seed: int,
    num_train: int,
    num_eval: int,
    frac_0: float,
    frac_top5: float,
    alias: str,
    out_dir: str,
    tokenizer_name: str,
    num_workers: int,
    num_dicts: int,
    min_token: int,
    max_token: int,
) -> None:
    if min_token >= max_token:
        raise ValueError(f"--min-token ({min_token}) must be < --max-token ({max_token})")

    random.seed(seed)
    np.random.seed(seed)

    data_cfg = DataConfig(num_dicts=num_dicts)
    task_cfg = TaskConfig(min_token=min_token, max_token=max_token)

    print(
        f"Token window: [{min_token or 0}, {max_token})  |  "
        f"base num_dicts: {num_dicts}  |  workers: {num_workers}"
    )

    # Pre-compute gold positions on the main process.
    # num_dicts is used as an upper bound for the index; the actual task may
    # have more dicts (grown to satisfy min_token), and _worker_generate_shard
    # clamps the index to task.num_dicts - 1 if needed.
    train_gold = [random.randint(0, num_dicts - 1) for _ in range(num_train)]
    eval_gold  = [random.randint(0, num_dicts - 1) for _ in range(num_eval)]

    if frac_0 > 0 or frac_top5 > 0:
        train_gold = apply_position_bias(train_gold, num_train, num_dicts, frac_0, frac_top5)
        eval_gold  = apply_position_bias(eval_gold,  num_eval,  num_dicts, frac_0, frac_top5)

    os.makedirs(out_dir, exist_ok=True)
    tmp_root = os.path.join(out_dir, "_tmp_shards")

    build_dataset_parallel(
        num=num_train,
        gold_idx_lst=train_gold,
        base_seed=seed,
        tokenizer_name=tokenizer_name,
        data_cfg=data_cfg,
        task_cfg=task_cfg,
        out_prefix=f"{out_dir}/{alias}_seed{seed}_train_{num_train}",
        num_workers=num_workers,
        tmp_dir=os.path.join(tmp_root, "train"),
        desc="Train",
    )

    build_dataset_parallel(
        num=num_eval,
        gold_idx_lst=eval_gold,
        base_seed=seed + 999_983,
        tokenizer_name=tokenizer_name,
        data_cfg=data_cfg,
        task_cfg=task_cfg,
        out_prefix=f"{out_dir}/{alias}_seed{seed}_eval_{num_eval}",
        num_workers=max(1, num_workers // 4),
        tmp_dir=os.path.join(tmp_root, "eval"),
        desc="Eval",
    )

    try:
        os.rmdir(tmp_root)
    except OSError:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate wtemplate data as Megatron indexed datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-train", type=int, default=350)
    parser.add_argument("--num-eval", type=int, default=150)
    parser.add_argument("--frac-0", type=float, default=0.05)
    parser.add_argument("--frac-top5", type=float, default=0.03)
    parser.add_argument("--alias", type=str, default="simpledict_ndicts85_34_wtemplate_ordered_5_3")
    parser.add_argument("--out-dir", type=str, default="./dataset")
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=DEFAULT_TOKENIZER,
        help="HuggingFace model name passed to AutoTokenizer.from_pretrained().",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=max(1, mp.cpu_count() - 1),
        help="Number of parallel worker processes.",
    )
    parser.add_argument(
        "--num-dicts",
        type=int,
        default=85,
        help=(
            "Base / minimum number of dictionaries per prompt. "
            "When --min-token > 0 extra distractors are added automatically "
            "until the prompt crosses the minimum token threshold."
        ),
    )
    parser.add_argument(
        "--min-token",
        type=int,
        default=4096,
        help=(
            "Minimum prompt length in tokens (inclusive). "
            "0 means no lower bound (old behaviour). "
            "Example: --min-token 16384 --max-token 32768"
        ),
    )
    parser.add_argument(
        "--max-token",
        type=int,
        default=32768,
        help="Maximum prompt length in tokens (exclusive upper bound).",
    )
    args = parser.parse_args()

    main(
        seed=args.seed,
        num_train=args.num_train,
        num_eval=args.num_eval,
        frac_0=args.frac_0,
        frac_top5=args.frac_top5,
        alias=args.alias,
        out_dir=args.out_dir,
        tokenizer_name=args.tokenizer,
        num_workers=args.num_workers,
        num_dicts=args.num_dicts,
        min_token=args.min_token,
        max_token=args.max_token,
    )