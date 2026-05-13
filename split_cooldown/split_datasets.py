#!/usr/bin/env python3
"""
Split MMap indexed datasets into N non-overlapping datamixes, each
targeting a fixed total token count, with a per-datamix per-output-file
minimum size.

For every .idx/.bin pair found recursively under any --input-dir:

    * non-overlapping  - every doc lives in at most one datamix
    * source-balanced  - each datamix has the same source proportions
                         as the union of inputs (each shard contributes
                         the same fraction to each datamix)
    * size-targeted    - each datamix sums to ~--target-tokens (default 40B)
    * file-floored     - any per-shard output that would be smaller than
                         that datamix's --min-tokens threshold is skipped
                         (those tokens are discarded, NOT redistributed
                         to other datamixes - the 40B target stays honest)

Concretely, with target T_per_dm per datamix and T_total tokens across
all inputs, each shard contributes a fraction f = T_per_dm / T_total to
each datamix.  A datamix is "viable" for a given shard iff
f * shard_tokens >= that datamix's min_tokens; non-viable datamixes
simply don't get a slice of that shard.  The remaining (1 - n_viable*f)
fraction of every shard is leftover and discarded.

Output layout mirrors the input under each datamix folder.  The basename
of each --input-dir is preserved so different modalities don't collide.

Usage:
    python split_datamixes.py \\
        --input-dir /.../text_stage_1 /.../vision_stage_1 /.../audio_stage_1 \\
        --output-dir /.../datamixes \\
        --num-buckets 4 \\
        --target-tokens 40000000000 \\
        --min-tokens 32768 65536 131072 262144 \\
        --seed 42 --workers 32

Defaults: 4 datamixes of 40B each, min-tokens 32k / 64k / 128k / 256k.
"""

import argparse
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from megatron.core.datasets import helpers
from megatron.core.datasets.indexed_dataset import IndexedDataset, IndexedDatasetBuilder

logger = logging.getLogger(__name__)


# Defaults aligned to the user's request when --num-buckets == 4.
DEFAULT_TARGET_TOKENS = 40_000_000_000
DEFAULT_MIN_TOKENS_4 = [32_768, 65_536, 131_072, 262_144]


def unique_labels(paths):
    """Shortest unique suffix of each path, used as output subdirectory."""
    depth = 1
    while True:
        labels = [Path(*p.parts[-depth:]) if depth <= len(p.parts) else p
                  for p in paths]
        if len(set(str(l) for l in labels)) == len(labels):
            return labels
        depth += 1

def fmt_tokens(n):
    if n >= 1e9: return f"{n / 1e9:.3f}B"
    if n >= 1e6: return f"{n / 1e6:.1f}M"
    if n >= 1e3: return f"{n / 1e3:.1f}K"
    return str(int(n))


def split_documents(n_docs, ratios, seed):
    """Randomly assign documents to splits using greedy error sampling.

    Permutes doc indices, then uses Megatron's build_blending_indices to
    minimise the maximum deviation from target ratios.  Returns one sorted
    array of doc indices per split (sorted to keep sequential I/O on .bin).
    """
    rng = np.random.default_rng(seed)
    doc_perm = rng.permutation(n_docs)

    n_splits = len(ratios)
    weights = np.array(ratios, dtype=np.float64)
    split_assignment = np.zeros(n_docs, dtype=np.int16)
    _unused = np.zeros(n_docs, dtype=np.int64)
    helpers.build_blending_indices(
        split_assignment, _unused, weights, n_splits, n_docs, False
    )
    return [np.sort(doc_perm[split_assignment == s]) for s in range(n_splits)]


def write_split(dataset, doc_idx, doc_ids, prefix_out, dtype):
    """Write a subset of documents to a new .idx/.bin pair."""
    os.makedirs(os.path.dirname(prefix_out), exist_ok=True)
    builder = IndexedDatasetBuilder(prefix_out + ".bin", dtype=dtype)
    for d in doc_ids:
        start, end = int(doc_idx[d]), int(doc_idx[d + 1])
        sequences = dataset[start:end]
        lengths = [len(s) for s in sequences]
        builder.add_document(np.concatenate(sequences), lengths)
    builder.finalize(prefix_out + ".idx")


def index_shard_size(prefix_in):
    """Read just the .idx to get this shard's total token count."""
    if not IndexedDataset.exists(prefix_in):
        return prefix_in, 0
    ds = IndexedDataset(prefix_in)
    s = int(ds.sequence_lengths.sum())
    del ds
    return prefix_in, s


def process_shard(prefix_in, prefixes_out, fraction, min_tokens_per_dm, seed):
    """Split one shard so each viable datamix gets `fraction` of the shard.

    A datamix is "viable" for this shard iff fraction * shard_tokens >= its
    min_tokens.  Non-viable datamixes get no slice of this shard.  The
    remainder (1 - n_viable * fraction) is discarded.

    Returns:
        (total_tokens_in_shard,
         [tokens_written_per_datamix],
         discarded_tokens)
    """
    name = Path(prefix_in).name
    n_dm = len(min_tokens_per_dm)

    if not IndexedDataset.exists(prefix_in):
        logger.warning("%s: missing .idx or .bin, skipping", name)
        return 0, [0] * n_dm, 0

    dataset = IndexedDataset(prefix_in)
    doc_idx = dataset.document_indices
    seq_lengths = dataset.sequence_lengths
    dtype = dataset.index.dtype
    n_docs = len(doc_idx) - 1
    total_tokens = int(seq_lengths.sum())

    if n_docs == 0:
        del dataset
        logger.warning("%s: empty, skipping", name)
        return 0, [0] * n_dm, 0

    # Per-doc tokens (vectorized) - used for accounting
    doc_tokens = np.add.reduceat(seq_lengths, doc_idx[:-1].astype(int))

    # Per-shard share each datamix expects
    per_dm_share = fraction * total_tokens

    # Determine viable datamixes for THIS shard
    viable = []
    for i in range(n_dm):
        if per_dm_share >= min_tokens_per_dm[i]:
            viable.append(i)
        else:
            logger.warning("%s: dm%d share %s < min_tokens %d, skipping for this shard",
                           name, i + 1,
                           fmt_tokens(int(per_dm_share)),
                           min_tokens_per_dm[i])

    if not viable:
        del dataset
        logger.warning("%s: %s too small for any datamix, discarding entire shard",
                       name, fmt_tokens(total_tokens))
        return total_tokens, [0] * n_dm, total_tokens

    # Build ratios: each viable datamix gets `fraction`, rest is a discard column
    ratios = [fraction] * len(viable)
    discard_ratio = 1.0 - len(viable) * fraction
    has_discard = discard_ratio > 1e-9
    if has_discard:
        ratios.append(discard_ratio)

    # Renormalize for floating-point safety
    rsum = sum(ratios)
    ratios = [r / rsum for r in ratios]

    splits = split_documents(n_docs, ratios, seed)

    # Write viable splits (skip the trailing discard column if present)
    per_dm_tokens = [0] * n_dm
    parts = []
    for vi, dm_idx in enumerate(viable):
        doc_ids = splits[vi]
        if len(doc_ids) == 0:
            parts.append(f"dm{dm_idx + 1}=0")
            continue
        tok = int(doc_tokens[doc_ids].sum())
        per_dm_tokens[dm_idx] = tok
        write_split(dataset, doc_idx, doc_ids, prefixes_out[dm_idx], dtype)
        parts.append(f"dm{dm_idx + 1}={len(doc_ids)}d/{fmt_tokens(tok)}")

    discarded = 0
    if has_discard and len(splits) > len(viable):
        discard_ids = splits[len(viable)]
        discarded = int(doc_tokens[discard_ids].sum())

    del dataset
    discard_str = f"  [discard {fmt_tokens(discarded)}]" if discarded else ""
    logger.info("%s: %d docs %s -> %s%s",
                name, n_docs, fmt_tokens(total_tokens),
                " ".join(parts), discard_str)
    return total_tokens, per_dm_tokens, discarded


def discover_shards(input_dirs):
    """Recursively yield (shard_prefix, input_root) for every *.idx file."""
    for root in input_dirs:
        root = Path(root).resolve()
        if not root.is_dir():
            logger.warning("Skipping non-directory: %s", root)
            continue
        idxs = sorted(root.rglob("*.idx"))
        if not idxs:
            logger.warning("No .idx files found under %s", root)
            continue
        for idx in idxs:
            yield str(idx)[:-4], root


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--input-dir", nargs="+", required=True,
        help="One or more roots containing .idx/.bin shards (recursively scanned).")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--num-buckets", type=int, default=4,
        help="Number of datamixes (default: 4).")
    ap.add_argument("--target-tokens", type=int, default=DEFAULT_TARGET_TOKENS,
        help="Target tokens per datamix (default: 40_000_000_000). Pass 0 "
             "to fall back to equal split (each datamix gets ~total/N).")
    ap.add_argument("--min-tokens", type=int, nargs="+", default=None,
        help="Per-datamix minimum tokens for any single output binary "
             "file. If a shard's allocated slice for datamix_i would be "
             "below this, the slice is skipped (and discarded). Length "
             "must equal --num-buckets. Default for --num-buckets=4: "
             "32768 65536 131072 262144.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=None,
        help="Parallel workers (default: cpu_count).")
    ap.add_argument("--dry-run", action="store_true",
        help="List shards, total tokens, and computed fraction without writing.")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if args.num_buckets < 1:
        logger.error("--num-buckets must be >= 1"); sys.exit(1)
    if args.target_tokens < 0:
        logger.error("--target-tokens must be >= 0"); sys.exit(1)

    # Resolve --min-tokens default / validate length
    if args.min_tokens is None:
        if args.num_buckets == 4:
            args.min_tokens = list(DEFAULT_MIN_TOKENS_4)
        else:
            logger.error("--min-tokens is required when --num-buckets != 4")
            sys.exit(1)
    if len(args.min_tokens) != args.num_buckets:
        logger.error("--min-tokens has %d entries but --num-buckets=%d",
                     len(args.min_tokens), args.num_buckets)
        sys.exit(1)
    if any(t < 0 for t in args.min_tokens):
        logger.error("--min-tokens values must be >= 0"); sys.exit(1)

    # Reject duplicate basenames among input roots (would collide in output)
    resolved = [Path(d).resolve() for d in args.input_dir]

    labels = unique_labels(resolved)
    root_to_label = dict(zip(resolved, labels))

    output_dir = Path(args.output_dir).resolve()

    logger.info("Inputs:        %s", [str(r) for r in resolved])
    logger.info("Output:        %s", output_dir)
    logger.info("Datamixes:     %d", args.num_buckets)
    logger.info("Target/dm:     %s%s", fmt_tokens(args.target_tokens),
                "  (equal split)" if args.target_tokens == 0 else "")
    for i, t in enumerate(args.min_tokens):
        logger.info("  datamix_%d  min_tokens per output file = %d", i + 1, t)
    logger.info("Seed:          %d", args.seed)

    # Discover shards
    shard_info = list(discover_shards(args.input_dir))
    if not shard_info:
        logger.error("No .idx files found under any --input-dir.")
        sys.exit(1)

    n_workers = min(args.workers or os.cpu_count(), len(shard_info))

    # ----- Phase 1: total token count across all shards -----
    logger.info("Phase 1: indexing %d shards to compute total token count …",
                len(shard_info))
    t1 = time.perf_counter()
    total_input_tokens = 0
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futs = {pool.submit(index_shard_size, p): p for p, _ in shard_info}
        for f in as_completed(futs):
            _, s = f.result()
            total_input_tokens += s
    logger.info("Phase 1: %s total tokens (%.1fs)",
                fmt_tokens(total_input_tokens), time.perf_counter() - t1)

    if total_input_tokens == 0:
        logger.error("Total input is 0 tokens; nothing to do."); sys.exit(1)

    # ----- Compute the per-shard per-datamix fraction -----
    if args.target_tokens == 0:
        fraction = 1.0 / args.num_buckets
        logger.info("Target=0 -> equal split, fraction=%.6f per datamix "
                    "(~%s per datamix)",
                    fraction, fmt_tokens(total_input_tokens / args.num_buckets))
    else:
        fraction = args.target_tokens / total_input_tokens
        if fraction * args.num_buckets > 1.0 + 1e-9:
            logger.warning("Total input %s < %d * target %s; cannot hit "
                           "target. Falling back to equal split: each "
                           "datamix gets ~%s.",
                           fmt_tokens(total_input_tokens), args.num_buckets,
                           fmt_tokens(args.target_tokens),
                           fmt_tokens(total_input_tokens / args.num_buckets))
            fraction = 1.0 / args.num_buckets
        else:
            discard_pct = 100.0 * (1.0 - fraction * args.num_buckets)
            logger.info("Per-shard fraction per datamix: %.6f  "
                        "(%.2f%% of every shard discarded as overflow)",
                        fraction, discard_pct)

    # ----- Build task list -----
    master_rng = np.random.default_rng(args.seed)
    tasks = []
    for prefix_in, root in shard_info:
        rel = root_to_label[root] / Path(prefix_in).relative_to(root)
        prefixes_out = [
            str(output_dir / f"datamix_{i + 1}" / rel)
            for i in range(args.num_buckets)
        ]
        seed = int(master_rng.integers(0, 2**63))
        tasks.append((prefix_in, prefixes_out, fraction, args.min_tokens, seed))

    if args.dry_run:
        logger.info("=" * 70)
        logger.info("DRY RUN — would process %d shards:", len(tasks))
        for prefix_in, prefixes_out, *_ in tasks:
            logger.info("  %s", prefix_in)
            for p in prefixes_out:
                logger.info("    -> %s.{idx,bin}", p)
        return

    # ----- Phase 2: split & write -----
    logger.info("Phase 2: splitting %d shards with %d workers …",
                len(tasks), n_workers)
    t2 = time.perf_counter()
    grand_total = 0
    grand_discard = 0
    dm_totals = [0] * args.num_buckets

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(process_shard, *t): t[0] for t in tasks}
        for future in as_completed(futures):
            try:
                total, per_dm, discarded = future.result()
                grand_total += total
                grand_discard += discarded
                for i, tk in enumerate(per_dm):
                    dm_totals[i] += tk
            except Exception:
                logger.exception("FAILED: %s", futures[future])

    elapsed = time.perf_counter() - t2
    written = sum(dm_totals)
    logger.info("=" * 70)
    logger.info("Total input:       %s", fmt_tokens(grand_total))
    logger.info("Total written:     %s  (%.2f%% of input)",
                fmt_tokens(written),
                100.0 * written / grand_total if grand_total else 0.0)
    logger.info("Total discarded:   %s  (%.2f%% of input)",
                fmt_tokens(grand_discard),
                100.0 * grand_discard / grand_total if grand_total else 0.0)
    target = args.target_tokens if args.target_tokens > 0 \
             else grand_total / args.num_buckets
    for i, tk in enumerate(dm_totals):
        pct = 100.0 * tk / target if target else 0.0
        logger.info("  datamix_%d:  %s  (%.2f%% of %s target, min_file=%d)",
                    i + 1, fmt_tokens(tk), pct,
                    fmt_tokens(target), args.min_tokens[i])
    logger.info("Phase 2 done in %.1fs", elapsed)


if __name__ == "__main__":
    main()