#!/usr/bin/env python3
"""
Sample N documents from an MMap indexed dataset, then for each document
individually find its most common non-stopword words, pick one from the
top-K at random, and append a word-count QA chat (formatted with the
model's chat template) before the final </s> token.

Each document gets its OWN chosen word based on ITS OWN word frequencies.

Chat template injected per document:
    <system>  You are a helpful assistant. Be concise.
    <user>    Take a look at the text above and tell me how many times
              the word {word} appears.
    <assistant> The word {word} appears {count} times.

where {word} and {count} are both derived from THAT document.

Output layout (flat):
    <output_dir>/<source_name>/dump-0/00000_tokens.{idx,bin}

Usage:
    python create_cwe.py \\
        --source institutional-books-1.0-filtered \\
        --n-docs 10000 \\
        --output-dir /path/to/output \\
        --seed 42
"""

import argparse
import logging
import os
import re
import string
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from megatron.core.datasets.indexed_dataset import IndexedDataset, IndexedDatasetBuilder

logger = logging.getLogger(__name__)


DATASETS = {
    "Biomed-Enriched_preprocessed":
        "/capstor/scratch/cscs/dtamayomela/data/tokenized_data/Apertus-70B-2509/Biomed-Enriched_preprocessed",
    "dolma3_olmocr_science_pdfs-preprocessed":
        "/capstor/scratch/cscs/dtamayomela/data/tokenized_data/Apertus-70B-2509/dolma3_olmocr_science_pdfs-preprocessed",
    "institutional-books-1.0-filtered":
        "/capstor/scratch/cscs/dtamayomela/data/tokenized_data/Apertus-70B-2509/institutional-books-1.0-filtered",
    "finepdfs-edu-multilingual-preprocessed":
        "/capstor/scratch/cscs/dtamayomela/data/corrected_data/finepdfs-edu-multilingual-preprocessed/second_half_longcontext",
    "finepdfs-edu-preprocessed":
        "/capstor/scratch/cscs/dtamayomela/long_context/sampled_buckets/32k_baseline/finepdfs-edu-preprocessed",
    "finepdfs-edu-preprocessed-16k":
        "/capstor/scratch/cscs/dtamayomela/long_context/sampled_buckets_16k/16k_baseline/finepdfs-edu-preprocessed",
    "finetranslations":
        "/capstor/scratch/cscs/dtamayomela/data/corrected_data/finetranslations/second_half_longcontext",
    "swissai-fineweb-2_0_1-quality_10-filterrobots":
        "/capstor/scratch/cscs/dtamayomela/data/corrected_data/swissai-fineweb-2_0_1-quality_10-filterrobots/second_half_longcontext",
    "testing":
        "/capstor/scratch/cscs/dtamayomela/synthetic_data/data_seed_example",
}

TOKENIZER_NAME = "swiss-ai/Apertus-8B-Instruct-2509"

# Token ids that bookend every document in the binary format
BOS_ID = 1   # <s>
EOS_ID = 2   # </s>

# Stopwords to exclude from word-frequency analysis
STOPWORDS = {
    # Articles / conjunctions / prepositions
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "into", "out", "about", "after", "before",
    "between", "through", "over", "under", "upon", "onto", "among",
    # Auxiliary verbs
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "shall", "can",
    # Pronouns
    "he", "she", "they", "we", "you", "i", "me", "him", "her", "us",
    "them", "my", "your", "his", "our", "their", "its",
    "myself", "yourself", "himself", "herself", "itself",
    "themselves", "ourselves",
    # Determiners / quantifiers
    "this", "that", "these", "those", "all", "any", "each", "every",
    "both", "either", "neither", "few", "more", "most", "other",
    "some", "such", "same", "own",
    # Common adverbs / particles
    "not", "no", "nor", "so", "yet", "too", "very", "just", "also",
    "then", "there", "here", "when", "where", "why", "how",
    "what", "which", "who", "whom", "whose",
    # Common short words that survive the length filter but add no signal
    "said", "like", "well", "even", "back", "only", "come", "good",
    "know", "time", "year", "make", "look", "take", "much", "them",
    "than", "then", "want", "does", "from", "with", "have", "this",
    "that", "will", "your", "into", "over", "just", "also", "come",
    "been", "made", "many", "more", "than", "when", "were", "been",
    "they", "their", "what", "there", "about", "would", "could",
    "should", "after", "other", "those", "still", "being",
}

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


def is_valid_word(w, min_word_len=4):
    """
    Return True if w is a meaningful word suitable as the QA target.

    Filters applied (in order):
      1. Minimum length of min_word_len characters — eliminates BPE
         fragments like "se", "re", "ve", "al", "en", "un", etc.
      2. Must not be in STOPWORDS.
      3. Must contain at least one alphabetic character.
      4. Must not be purely punctuation or digits.
    """
    if len(w) < min_word_len:
        return False
    if w.lower() in STOPWORDS:
        return False
    if not any(c.isalpha() for c in w):
        return False
    if all(c in string.punctuation + string.digits for c in w):
        return False
    return True


def pick_word_for_doc(text, top_k, min_word_len, rng):
    """
    Count valid words in `text`, pick one from the top-K at random.
    Returns (chosen_word, count_in_doc), or (None, 0) if no valid words.
    """
    doc_counter = Counter()
    for w in re.findall(r"[A-Za-z]+", text):
        w_lower = w.lower()
        if is_valid_word(w_lower, min_word_len):
            doc_counter[w_lower] += 1

    if not doc_counter:
        return None, 0

    candidates = doc_counter.most_common(top_k)
    chosen_word, chosen_count = candidates[int(rng.integers(0, len(candidates)))]
    return chosen_word, chosen_count


def build_chat_tokens(tokenizer, word, count):
    """
    Build token ids for the QA chat using the model's chat template.
    Returns a 1-D numpy int64 array (no leading BOS / trailing EOS).
    """
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Be concise.",
        },
        {
            "role": "user",
            "content": (
                f"Take a look at the text above and tell me how many times "
                f"the word \"{word}\" appears."
            ),
        },
        {
            "role": "assistant",
            "content": f"The word \"{word}\" appears {count} times.",
        },
    ]
    token_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
    )
    # Strip leading <s> (BOS_ID=1) explicitly — do NOT use tokenizer.bos_token_id
    if token_ids and token_ids[0] == BOS_ID:
        token_ids = token_ids[1:]
    # Strip trailing </s> (EOS_ID=2) only — NOT <|assistant_end|> (token 68).
    if token_ids and token_ids[-1] == EOS_ID:
        token_ids = token_ids[:-1]
    return np.array(token_ids, dtype=np.int64)


def sample_doc_indices(shard_prefixes, n_docs, rng):
    """
    Returns list of (shard_idx, doc_id_within_shard) tuples, length <= n_docs.
    Sampling is uniform over all documents across all shards.
    """
    counts = []
    for prefix in shard_prefixes:
        if not IndexedDataset.exists(prefix):
            counts.append(0)
            continue
        ds = IndexedDataset(prefix)
        counts.append(len(ds.document_indices) - 1)
        del ds

    total = sum(counts)
    if total == 0:
        return []

    actual_n = min(n_docs, total)
    if actual_n < n_docs:
        logger.warning("Only %d docs available (requested %d)", total, n_docs)

    global_ids = rng.choice(total, size=actual_n, replace=False)
    global_ids.sort()

    boundaries = np.concatenate([[0], np.cumsum(counts)])
    result = []
    for gid in global_ids:
        si = int(np.searchsorted(boundaries, gid, side="right")) - 1
        local_id = int(gid - boundaries[si])
        result.append((si, local_id))

    return result


# Phase 2: single pass — decode, pick word, inject chat, write 

def process_and_write(
    shard_prefixes, sampled, tokenizer, output_dir, source_name,
    top_k, min_word_len, rng,
):
    """
    For each sampled doc in a single sequential pass:
      1. Decode the document text.
      2. Count valid words in THAT document and pick one from the top-K.
      3. Build the QA chat using the per-document word and its per-document count.
      4. Insert chat tokens before the trailing </s> and write to the new binary.
    """
    prefix = os.path.join(output_dir, source_name, "dump-0", "00000_tokens")
    os.makedirs(os.path.dirname(prefix), exist_ok=True)

    dtype = None
    for sp in shard_prefixes:
        if IndexedDataset.exists(sp):
            dtype = IndexedDataset(sp).index.dtype
            break
    if dtype is None:
        raise RuntimeError("No valid shards found")

    builder = IndexedDatasetBuilder(prefix + ".bin", dtype=dtype)

    by_shard = {}
    for order_idx, (si, did) in enumerate(sampled):
        by_shard.setdefault(si, []).append((did, order_idx))

    total_docs   = 0
    total_tokens = 0
    skipped_docs = 0

    for si in sorted(by_shard):
        sp = shard_prefixes[si]
        if not IndexedDataset.exists(sp):
            logger.warning("Shard %d missing, skipping", si)
            continue
        ds = IndexedDataset(sp)
        doc_idx = ds.document_indices

        entries = sorted(by_shard[si], key=lambda x: x[0])

        i = 0
        while i < len(entries):
            run_start_did = entries[i][0]
            run_end_did   = run_start_did
            j = i
            while j + 1 < len(entries) and entries[j + 1][0] == run_end_did + 1:
                j += 1
                run_end_did = entries[j][0]

            seq_start = int(doc_idx[run_start_did])
            seq_end   = int(doc_idx[run_end_did + 1])
            all_seqs  = ds[seq_start:seq_end]

            for k in range(i, j + 1):
                did, order_idx = entries[k]
                lo = int(doc_idx[did])     - seq_start
                hi = int(doc_idx[did + 1]) - seq_start
                seqs = all_seqs[lo:hi]

                doc_tokens = (
                    np.concatenate(seqs).astype(np.int64)
                    if seqs else np.array([], dtype=np.int64)
                )

                if len(doc_tokens) > 16_384:
                    logger.debug(
                        "Skipping doc (shard=%d, doc=%d): %d tokens > 16k limit.",
                        si, did, len(doc_tokens),
                    )
                    skipped_docs += 1
                    continue
                    
                # Decode for word analysis (skip special tokens)
                text = tokenizer.decode(doc_tokens, skip_special_tokens=True)

                # Pick the word from THIS document's own frequency distribution
                chosen_word, chosen_count = pick_word_for_doc(
                    text, top_k, min_word_len, rng
                )

                if chosen_word is None:
                    logger.warning(
                        "No valid words in doc (shard=%d, doc=%d) — skipping.", si, did
                    )
                    skipped_docs += 1
                    continue

                # Build chat tokens for this document's word and count
                chat_tokens = build_chat_tokens(tokenizer, chosen_word, chosen_count)

                # Insert before trailing </s>
                # Layout: [BOS=1, ..content.., EOS=2]
                # Target: [BOS=1, ..content.., <chat>, EOS=2]
                if len(doc_tokens) > 0 and doc_tokens[-1] == EOS_ID:
                    new_tokens = np.concatenate([
                        doc_tokens[:-1],
                        chat_tokens,
                        doc_tokens[-1:],
                    ])
                else:
                    new_tokens = np.concatenate([doc_tokens, chat_tokens])

                new_tokens_typed = new_tokens.astype(dtype)
                builder.add_document(new_tokens_typed, [len(new_tokens_typed)])
                total_docs   += 1
                total_tokens += len(new_tokens_typed)

            i = j + 1
        del ds

    builder.finalize(prefix + ".idx")
    logger.info(
        "Wrote %d docs, %s tokens → %s  (%d skipped — no valid words)",
        total_docs, fmt_tokens(total_tokens), prefix, skipped_docs,
    )



def main():
    ap = argparse.ArgumentParser(
        description="Sample N docs, inject per-doc word-count QA chat, write new binary."
    )
    ap.add_argument(
        "--source", required=True, choices=list(DATASETS.keys()),
        help="Source dataset name.",
    )
    ap.add_argument(
        "--n-docs", type=int, default=10_000,
        help="Number of documents to sample (default: 10 000).",
    )
    ap.add_argument(
        "--output-dir", required=True,
        help="Root directory for output binaries.",
    )
    ap.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42).",
    )
    ap.add_argument(
        "--top-k", type=int, default=10,
        help="Pick the chosen word from the top-K most frequent in each doc (default: 10).",
    )
    ap.add_argument(
        "--min-word-len", type=int, default=4,
        help="Minimum characters for a word to be eligible (default: 4).",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Run phase 1 only (sampling), print stats, then stop.",
    )
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    data_dir   = DATASETS[args.source]
    output_dir = str(Path(args.output_dir).resolve())
    os.makedirs(output_dir, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    logger.info("Phase 1: discovering shards in %s …", data_dir)
    t0 = time.perf_counter()

    shard_prefixes = discover_shard_prefixes(data_dir)
    if not shard_prefixes:
        logger.error("No shards found under %s", data_dir)
        sys.exit(1)
    logger.info("  Found %d shards", len(shard_prefixes))

    sampled = sample_doc_indices(shard_prefixes, args.n_docs, rng)
    logger.info("  Sampled %d documents (%.1fs)", len(sampled), time.perf_counter() - t0)

    if args.dry_run:
        logger.info("Dry run — stopping before decode/write phase.")
        return

    logger.info(
        "Phase 2: loading tokenizer, decoding, injecting chats, writing …"
    )
    t2 = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    process_and_write(
        shard_prefixes, sampled, tokenizer,
        output_dir, args.source,
        top_k=args.top_k,
        min_word_len=args.min_word_len,
        rng=rng,
    )

    logger.info("Phase 2 done in %.1fs", time.perf_counter() - t2)
    logger.info("All done.")


if __name__ == "__main__":
    main()