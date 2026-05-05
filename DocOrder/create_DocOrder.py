#!/usr/bin/env python3
"""
DocOrder pipeline for combined easy + hard buckets.

Each output document is a single chat exchange:

    <s><|system_start|>  You are a helpful assistant. Be concise.<|system_end|>
       <|user_start|>    [Section 1]
                         ...shuffled section text...

                         [Section 2]
                         ...shuffled section text...
                         ...
                         <question>           <|user_end|>
       <|assistant_start|> <answer>           <|assistant_end|>
    </s>

There is nothing between <s> and <|system_start|>; the shuffled body
sits inside the user message, immediately followed by the question.

For every sampled document the script:

    1. Strips its leading <s> / trailing </s>.
    2. Splits the remaining content tokens into N roughly-equal chunks
       (N = --easy-sections for docs from the easy root, N = --hard-sections
       for docs from the hard root).
    3. Decodes each chunk back to text and randomly permutes them.
    4. Builds the user message as
           [Section 1]\\n<text>\\n\\n[Section 2]\\n<text>\\n\\n... \\n\\n<question>
       where the section labels are 1-indexed by SHUFFLED position.
    5. Picks one of 10 question phrasings — each one prescribes its own
       answer format — and renders the correct ordering accordingly.
    6. Calls tokenizer.apply_chat_template on
           [system, user(=shuffled_body + question), assistant(=answer)]
       and writes the resulting token sequence to a new binary.

Two source roots (easy + hard) are processed and their documents are
multiplexed into the SAME per-dataset builders.

Output layout (per pair):

    <output_dir>/
        <dataset_name_1>/dump-0/00000_tokens.{idx,bin}
        <dataset_name_2>/dump-0/00000_tokens.{idx,bin}
        ...

Usage:

    python create_DocOrder.py \\
        --hard-root /path/to/mix_8k_16k_cwe/hard \\
        --easy-root /path/to/mix_16k_32k_cwe/easy \\
        --output-dir /path/to/32k_cwe \\
        --hard-sections 8 \\
        --easy-sections 4 \\
        --hard-tokens 1000000000 \\
        --easy-tokens 1000000000 \\
        --seed 42
"""

import argparse
import logging
import os
import time
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from megatron.core.datasets.indexed_dataset import (
    IndexedDataset,
    IndexedDatasetBuilder,
)

logger = logging.getLogger(__name__)

TOKENIZER_NAME = "swiss-ai/Apertus-8B-Instruct-2509"

# Token ids that bookend every document in the binary format
BOS_ID = 1   # <s>
EOS_ID = 2   # </s>

# Minimum number of source tokens per section. Documents smaller than
# n_sections * MIN_TOKENS_PER_SECTION are skipped.
MIN_TOKENS_PER_SECTION = 32


# 10 question phrasings. Each tuple is (question_text, answer_formatter).
# The answer_formatter takes a 1-indexed list of section labels in the
# order they should appear in the original document and renders it in the
# format demanded by the question.
QUESTION_VARIANTS = [
    (
        "The sections of the document above have been rearranged. "
        "Reconstruct the original ordering and respond with the section "
        "numbers as a comma-separated list (e.g. '3,1,4,2'). Output only "
        "the list, with no spaces and no extra text.",
        lambda order: ",".join(str(x) for x in order),
    ),
    (
        "Above you can see a document whose sections appear in the wrong "
        "order. Provide the correct sequence as section numbers separated "
        "by single spaces. Reply with only that line.",
        lambda order: " ".join(str(x) for x in order),
    ),
    (
        "The document above is out of order. Reorder its sections and "
        "reply with a JSON array of section numbers, e.g. [2,4,1,3]. "
        "Output only the array.",
        lambda order: "[" + ",".join(str(x) for x in order) + "]",
    ),
    (
        "Each section of the document above carries a number indicating "
        "its current position. Tell me the order in which they should be "
        "read by listing the section numbers separated by ' -> '. No "
        "explanation.",
        lambda order: " -> ".join(str(x) for x in order),
    ),
    (
        "The text above presents the document's sections out of order. "
        "Output the correct sequence as a hyphen-separated string of "
        "section numbers (for example: 3-1-4-2). Just the string, "
        "nothing else.",
        lambda order: "-".join(str(x) for x in order),
    ),
    (
        "Look at the shuffled sections of the document above. Identify "
        "the proper reading order and answer with the section numbers "
        "wrapped in parentheses, e.g. (2)(4)(1)(3). No commentary.",
        lambda order: "".join(f"({x})" for x in order),
    ),
    (
        "The sections shown above appear in a permuted order. Provide "
        "the correct order as section numbers separated by commas with "
        "a single space after each comma (e.g. '3, 1, 4, 2'). Reply only "
        "with the list.",
        lambda order: ", ".join(str(x) for x in order),
    ),
    (
        "Above is a document whose sections have been scrambled. Restore "
        "the proper ordering and answer with a pipe-separated sequence "
        "of section numbers, like '3|1|4|2'. Output only that string.",
        lambda order: "|".join(str(x) for x in order),
    ),
    (
        "The document's sections are presented out of order above. "
        "Return the correct sequence of section numbers, one per line, "
        "with no other characters.",
        lambda order: "\n".join(str(x) for x in order),
    ),
    (
        "The sections of the document above have been shuffled. Give the "
        "correct order using the format 'Order: N1, N2, N3, ...' where "
        "each Nk is a section number. Output that single line only.",
        lambda order: "Order: " + ", ".join(str(x) for x in order),
    ),
]

SYSTEM_PROMPT = "You are a helpful assistant. Be concise."


def fmt_tokens(n):
    if n >= 1e9: return f"{n / 1e9:.3f}B"
    if n >= 1e6: return f"{n / 1e6:.1f}M"
    if n >= 1e3: return f"{n / 1e3:.1f}K"
    return str(n)


def discover_shard_prefixes(data_dir):
    root = Path(data_dir)
    prefixes = []
    for dump in sorted(root.glob("dump-*")):
        if dump.is_dir():
            for idx in sorted(dump.glob("*_tokens.idx")):
                prefixes.append(str(idx)[:-4])
    return prefixes


def collect_doc_metadata(shard_prefixes):
    """
    Returns (all_docs, dtype) where all_docs is a list of
    (shard_idx, doc_id, n_tokens) tuples covering every document in every
    shard. Token counts are computed from the index, not by loading the
    sequences themselves.
    """
    all_docs = []
    dtype = None
    for si, sp in enumerate(shard_prefixes):
        if not IndexedDataset.exists(sp):
            continue
        ds = IndexedDataset(sp)
        if dtype is None:
            dtype = ds.index.dtype
        doc_idx = ds.document_indices
        seq_lens = ds.sequence_lengths
        n_docs = len(doc_idx) - 1
        if n_docs == 0:
            del ds
            continue
        cumsum = np.concatenate([[0], np.cumsum(seq_lens.astype(np.int64))])
        for did in range(n_docs):
            n_tokens = int(cumsum[doc_idx[did + 1]] - cumsum[doc_idx[did]])
            all_docs.append((si, did, n_tokens))
        del ds
    return all_docs, dtype


def sample_to_token_budget(all_docs, target_tokens, rng):
    """
    Shuffle all_docs uniformly and consume entries until the cumulative
    source-token count meets or exceeds target_tokens. Returns
    (sampled, total_source_tokens) where sampled is a list of
    (shard_idx, doc_id) tuples.
    """
    if not all_docs:
        return [], 0
    indices = np.arange(len(all_docs))
    rng.shuffle(indices)
    sampled = []
    total = 0
    for i in indices:
        si, did, n = all_docs[int(i)]
        sampled.append((si, did))
        total += n
        if total >= target_tokens:
            break
    return sampled, total


def split_into_sections(content_tokens, n_sections):
    """Split a token list into n_sections roughly-equal contiguous chunks."""
    L = len(content_tokens)
    if L < n_sections:
        return None
    section_size = L // n_sections
    chunks = []
    for i in range(n_sections):
        start = i * section_size
        end = (i + 1) * section_size if i < n_sections - 1 else L
        chunks.append(list(content_tokens[start:end]))
    return chunks


def get_doc_tokens(ds, did):
    """Concatenate all sequences for document `did` into a single 1-D int64 array."""
    seq_start = int(ds.document_indices[did])
    seq_end   = int(ds.document_indices[did + 1])
    seqs = ds[seq_start:seq_end]
    if not seqs:
        return np.array([], dtype=np.int64)
    return np.concatenate(seqs).astype(np.int64)


def process_doc(doc_tokens, n_sections, tokenizer, rng):
    """
    Build the new (chat-template-formatted) token sequence for one
    document. Returns None if the document is too short to be meaningfully
    split.
    """
    tokens = doc_tokens.tolist() if hasattr(doc_tokens, "tolist") else list(doc_tokens)
    if not tokens:
        return None

    # Strip leading <s> / trailing </s> from the source document
    has_bos = tokens[0]  == BOS_ID
    has_eos = tokens[-1] == EOS_ID
    content = tokens[1 if has_bos else 0 : (-1 if has_eos else None)]

    if len(content) < n_sections * MIN_TOKENS_PER_SECTION:
        return None

    sections_tokens = split_into_sections(content, n_sections)
    if sections_tokens is None:
        return None

    # Decode each section back to text. skip_special_tokens=True is safe
    # because we already stripped the document's BOS/EOS; any special
    # tokens that survived inside the body would be artifacts.
    section_texts = [
        tokenizer.decode(s, skip_special_tokens=True)
        for s in sections_tokens
    ]

    # Random permutation: perm[shuffled_pos] = original_idx
    perm = list(range(n_sections))
    rng.shuffle(perm)

    # correct_order[k] = SHUFFLED-position label (1-indexed) of the
    # section that originally came k-th. Reading the document in the
    # order [correct_order[0], correct_order[1], ...] recovers the
    # original sequence.
    correct_order = [0] * n_sections
    for shuffled_pos, original_idx in enumerate(perm):
        correct_order[original_idx] = shuffled_pos + 1

    # Build the shuffled body as plain text
    body_parts = []
    for shuffled_pos, original_idx in enumerate(perm):
        label = shuffled_pos + 1
        body_parts.append(f"[Section {label}]\n{section_texts[original_idx]}")
    shuffled_body = "\n\n".join(body_parts)

    # Pick a question variant and render the answer in its format
    variant_idx = int(rng.integers(0, len(QUESTION_VARIANTS)))
    question, answer_fn = QUESTION_VARIANTS[variant_idx]
    answer_str = answer_fn(correct_order)

    # User turn = shuffled body, blank line, then the question
    user_content = shuffled_body + "\n\n" + question

    messages = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": user_content},
        {"role": "assistant", "content": answer_str},
    ]

    full_tokens = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
    )

    # The Apertus chat template emits <s> at the start and </s> at the
    # end, but defensively guarantee that boundary so the binary stays
    # well-formed even if a future template revision drops them.
    if not full_tokens or full_tokens[0] != BOS_ID:
        full_tokens = [BOS_ID] + list(full_tokens)
    if full_tokens[-1] != EOS_ID:
        full_tokens = list(full_tokens) + [EOS_ID]

    return full_tokens


# Per-dataset / per-root drivers

def process_one_dataset(ds_dir, n_sections, target_tokens, output_root,
                        tokenizer, rng, builders, label):
    ds_name = ds_dir.name
    shard_prefixes = discover_shard_prefixes(str(ds_dir))
    if not shard_prefixes:
        logger.warning("[%s/%s] no shards found", label, ds_name)
        return 0, 0

    all_docs, dtype = collect_doc_metadata(shard_prefixes)
    if not all_docs:
        logger.warning("[%s/%s] no documents found", label, ds_name)
        return 0, 0

    sampled, sampled_src_tokens = sample_to_token_budget(all_docs, target_tokens, rng)
    logger.info(
        "[%s/%s] sampled %d docs (~%s source tokens, budget %s)",
        label, ds_name, len(sampled),
        fmt_tokens(sampled_src_tokens), fmt_tokens(target_tokens),
    )

    # Lazily create one builder per dataset; keyed by dataset name so that
    # easy and hard for the same dataset share a builder.
    if ds_name not in builders:
        prefix = os.path.join(output_root, ds_name, "dump-0", "00000_tokens")
        os.makedirs(os.path.dirname(prefix), exist_ok=True)
        builders[ds_name] = {
            "prefix":  prefix,
            "builder": IndexedDatasetBuilder(prefix + ".bin", dtype=dtype),
            "dtype":   dtype,
        }
    info = builders[ds_name]
    builder, out_dtype = info["builder"], info["dtype"]

    # Group by shard for cheap sequential reads
    by_shard = {}
    for si, did in sampled:
        by_shard.setdefault(si, []).append(did)

    n_written = n_skipped = tokens_written = 0

    for si in sorted(by_shard):
        sp = shard_prefixes[si]
        if not IndexedDataset.exists(sp):
            continue
        ds = IndexedDataset(sp)
        for did in sorted(by_shard[si]):
            doc_tokens = get_doc_tokens(ds, did)
            new_tokens = process_doc(doc_tokens, n_sections, tokenizer, rng)
            if new_tokens is None:
                n_skipped += 1
                continue
            arr = np.array(new_tokens, dtype=out_dtype)
            builder.add_document(arr, [len(arr)])
            n_written      += 1
            tokens_written += len(arr)
        del ds

    logger.info(
        "[%s/%s] wrote %d docs / %s tokens (%d skipped — too short)",
        label, ds_name, n_written, fmt_tokens(tokens_written), n_skipped,
    )
    return n_written, tokens_written


def process_root(root_dir, n_sections, target_tokens, output_root,
                 tokenizer, rng, builders, label):
    dataset_dirs = sorted(d for d in Path(root_dir).iterdir() if d.is_dir())
    if not dataset_dirs:
        logger.warning("[%s] no dataset subdirs in %s", label, root_dir)
        return 0, 0
    per_ds_budget = target_tokens // len(dataset_dirs)
    logger.info(
        "[%s] %d datasets under %s — per-dataset budget %s",
        label, len(dataset_dirs), root_dir, fmt_tokens(per_ds_budget),
    )
    total_docs = total_tokens = 0
    for ds_dir in dataset_dirs:
        d, t = process_one_dataset(
            ds_dir, n_sections, per_ds_budget, output_root,
            tokenizer, rng, builders, label,
        )
        total_docs   += d
        total_tokens += t
    return total_docs, total_tokens


def main():
    ap = argparse.ArgumentParser(
        description="Apply the DocOrder task to combined easy + hard buckets.",
    )
    ap.add_argument("--easy-root", required=True,
                    help="Easy bucket root (contains one subdir per source dataset).")
    ap.add_argument("--hard-root", required=True,
                    help="Hard bucket root (contains one subdir per source dataset).")
    ap.add_argument("--output-dir", required=True,
                    help="Output root: one subdir per dataset, each with dump-0/.")
    ap.add_argument("--easy-sections", type=int, default=4)
    ap.add_argument("--hard-sections", type=int, default=8)
    ap.add_argument("--easy-tokens", type=int, default=1_000_000_000,
                    help="Source-token budget across all easy datasets (default 1B).")
    ap.add_argument("--hard-tokens", type=int, default=1_000_000_000,
                    help="Source-token budget across all hard datasets (default 1B).")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    output_dir = str(Path(args.output_dir).resolve())
    os.makedirs(output_dir, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    logger.info("Loading tokenizer %s …", TOKENIZER_NAME)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    builders = {}
    t0 = time.perf_counter()

    logger.info(
        "=== HARD root %s | n_sections=%d | budget=%s ===",
        args.hard_root, args.hard_sections, fmt_tokens(args.hard_tokens),
    )
    h_docs, h_tokens = process_root(
        args.hard_root, args.hard_sections, args.hard_tokens, output_dir,
        tokenizer, rng, builders, label="hard",
    )
    logger.info("HARD total: %d docs / %s tokens", h_docs, fmt_tokens(h_tokens))

    logger.info(
        "=== EASY root %s | n_sections=%d | budget=%s ===",
        args.easy_root, args.easy_sections, fmt_tokens(args.easy_tokens),
    )
    e_docs, e_tokens = process_root(
        args.easy_root, args.easy_sections, args.easy_tokens, output_dir,
        tokenizer, rng, builders, label="easy",
    )
    logger.info("EASY total: %d docs / %s tokens", e_docs, fmt_tokens(e_tokens))

    for name, info in builders.items():
        info["builder"].finalize(info["prefix"] + ".idx")
        logger.info("finalized %s.{bin,idx}", info["prefix"])

    logger.info(
        "ALL DONE — wrote %d docs / %s tokens in %.1fs → %s",
        h_docs + e_docs, fmt_tokens(h_tokens + e_tokens),
        time.perf_counter() - t0, output_dir,
    )


if __name__ == "__main__":
    main()