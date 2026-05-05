#!/usr/bin/env python3
"""
Multi-turn Counting Word Examples (CWE) injector.

For each document found under one or more input directories, this script:
  1. Decodes the document tokens to text.
  2. Counts non-stopword words (length >= min_word_len) and picks N
     DISTINCT words from the top-K most frequent ones (sampled without
     replacement).
  3. Builds a multi-turn QA chat -- one (user, assistant) turn per word --
     where each turn uses a randomly chosen question template from a pool
     of 10. Each template specifies the exact format the assistant must
     reply in (either "just the integer" or a templated string).
  4. Inserts the resulting chat token ids before the trailing </s> of the
     document.
  5. Writes the new document into a Megatron IndexedDatasetBuilder.

The script is designed to COMBINE two input buckets (a "hard" bucket with
many turns per doc and an "easy" bucket with fewer turns per doc) into a
single output dataset, preserving the per-source subdirectory layout.

Output layout:
    <output_dir>/<sub_source>/dump-0/00000_tokens.{bin,idx}

The bin for each <sub_source> contains, in order:
    - all docs from <hard_input>/<sub_source>/...   (--hard-questions turns)
    - all docs from <easy_input>/<sub_source>/...   (--easy-questions turns)

Example
-------
Combine 8k_16k/hard (10-turn) with 16k_32k/easy (5-turn) into 32k_cwe:

    python create_cwe_multi.py \\
        --hard-input /.../mix_8k_16k_cwe/hard \\
        --easy-input /.../mix_16k_32k_cwe/easy \\
        --output-dir /.../32k_cwe \\
        --hard-questions 10 \\
        --easy-questions 5 \\
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

from megatron.core.datasets.indexed_dataset import (
    IndexedDataset,
    IndexedDatasetBuilder,
)

logger = logging.getLogger(__name__)



TOKENIZER_NAME = "swiss-ai/Apertus-8B-Instruct-2509"

# Token ids that bookend every document in the binary format
BOS_ID = 1   # <s>
EOS_ID = 2   # </s>


# Stopwords excluded from word-frequency analysis (kept identical to the
# single-turn version so behaviour is consistent across pipelines).
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "into", "out", "about", "after", "before",
    "between", "through", "over", "under", "upon", "onto", "among",
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "shall", "can",
    "he", "she", "they", "we", "you", "i", "me", "him", "her", "us",
    "them", "my", "your", "his", "our", "their", "its",
    "myself", "yourself", "himself", "herself", "itself",
    "themselves", "ourselves",
    "this", "that", "these", "those", "all", "any", "each", "every",
    "both", "either", "neither", "few", "more", "most", "other",
    "some", "such", "same", "own",
    "not", "no", "nor", "so", "yet", "too", "very", "just", "also",
    "then", "there", "here", "when", "where", "why", "how",
    "what", "which", "who", "whom", "whose",
    "said", "like", "well", "even", "back", "only", "come", "good",
    "know", "time", "year", "make", "look", "take", "much", "them",
    "than", "then", "want", "does", "from", "with", "have", "this",
    "that", "will", "your", "into", "over", "just", "also", "come",
    "been", "made", "many", "more", "than", "when", "were", "been",
    "they", "their", "what", "there", "about", "would", "could",
    "should", "after", "other", "those", "still", "being",
}


# Ten distinct question phrasings. Each pairs a user prompt with the
# format the assistant must follow. Both fields are Python format strings:
#   - user uses {word}
#   - answer uses {word} and/or {count}
# The placeholders [WORD] and [N] inside user prompts are *descriptive*
# (they tell the assistant what to substitute); they are NOT format codes.
QUESTION_TEMPLATES = [
    {
        "user": 'Looking at the text above, how many occurrences of the word "{word}" can you find? Reply with only the number.',
        "answer": "{count}",
    },
    {
        "user": 'Scan the document and report the frequency of "{word}". Use this exact response template: "The word [WORD] is mentioned [N] times in the text."',
        "answer": 'The word "{word}" is mentioned {count} times in the text.',
    },
    {
        "user": 'I need a tally of "{word}" from the passage above. Format your answer as: "[WORD] -> [N]"',
        "answer": "{word} -> {count}",
    },
    {
        "user": 'Go through the text and count appearances of "{word}". Respond using exactly this layout: "Frequency of [WORD]: [N]"',
        "answer": "Frequency of {word}: {count}",
    },
    {
        "user": 'Could you tell me how often "{word}" shows up above? Just give me the integer, no explanation, no extra words.',
        "answer": "{count}",
    },
    {
        "user": 'Examine the document and quantify how many times "{word}" appears. Use this format for your reply: "Total occurrences: [N]"',
        "answer": "Total occurrences: {count}",
    },
    {
        "user": 'Please report the number of times "{word}" is found in the text. Your answer must follow this template: "[WORD] | count = [N]"',
        "answer": "{word} | count = {count}",
    },
    {
        "user": 'Count the instances of "{word}" in the passage. Reply with just a single integer number, nothing else.',
        "answer": "{count}",
    },
    {
        "user": 'How many hits do you get for "{word}" in the text above? Use this exact response shape: "Found [N] hit(s) for \'[WORD]\'."',
        "answer": "Found {count} hit(s) for '{word}'.",
    },
    {
        "user": 'In the document provided, how frequent is the word "{word}"? Format your answer as: "[N] occurrences of [WORD]"',
        "answer": "{count} occurrences of {word}",
    },
]


def fmt_tokens(n):
    if n >= 1e9: return f"{n / 1e9:.3f}B"
    if n >= 1e6: return f"{n / 1e6:.1f}M"
    if n >= 1e3: return f"{n / 1e3:.1f}K"
    return str(n)


def discover_shard_prefixes(data_dir):
    """
    Find every <prefix>.idx under <data_dir>/dump-*/ and return its prefix
    (path without the .idx extension), sorted.
    """
    root = Path(data_dir)
    prefixes = []
    for dump in sorted(root.glob("dump-*")):
        if dump.is_dir():
            for idx in sorted(dump.glob("*_tokens.idx")):
                prefixes.append(str(idx)[:-4])
    return prefixes


def is_valid_word(w, min_word_len=4):
    """
    Filter for "meaningful" words eligible to be counted.

      1. >= min_word_len chars (drops BPE fragments).
      2. Not in STOPWORDS.
      3. Has at least one alphabetic character.
      4. Not purely punctuation/digits.
    """
    if len(w) < min_word_len:
        return False
    if w in STOPWORDS:
        return False
    if not any(c.isalpha() for c in w):
        return False
    if all(c in string.punctuation + string.digits for c in w):
        return False
    return True


def pick_words_for_doc(text, n_words, top_k, min_word_len, rng):
    """
    Count valid words in `text`. Pick `n_words` DISTINCT words at random
    from the top_k most frequent ones (sampled without replacement).

    Returns
    -------
    list[(word, count)]
        Length == n_words, in random order, OR
    None
        if the doc has fewer than n_words valid distinct words.
    """
    doc_counter = Counter()
    for w in re.findall(r"[A-Za-z]+", text):
        w_lower = w.lower()
        if is_valid_word(w_lower, min_word_len):
            doc_counter[w_lower] += 1

    candidates = doc_counter.most_common(top_k)
    if len(candidates) < n_words:
        return None

    indices = rng.choice(len(candidates), size=n_words, replace=False)
    return [candidates[int(i)] for i in indices]


def build_multi_turn_chat_tokens(tokenizer, words_counts, rng):
    """
    Build token ids for a multi-turn QA chat. One turn (user + assistant)
    per (word, count) pair. Each turn picks a question template at random
    (without replacement when n_turns <= 10).

    Strips leading <s> and trailing </s> so the caller can splice the
    result into a host document cleanly.
    """
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Be concise.",
        },
    ]

    n_templates = len(QUESTION_TEMPLATES)
    n_turns = len(words_counts)
    if n_turns <= n_templates:
        tmpl_indices = rng.choice(n_templates, size=n_turns, replace=False)
    else:
        # Fallback: more turns than templates -> sample with replacement.
        tmpl_indices = rng.choice(n_templates, size=n_turns, replace=True)

    for (word, count), ti in zip(words_counts, tmpl_indices):
        tmpl = QUESTION_TEMPLATES[int(ti)]
        user_msg = tmpl["user"].format(word=word)
        asst_msg = tmpl["answer"].format(word=word, count=count)
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": asst_msg})

    token_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
    )
    # Strip leading <s> and trailing </s> -- explicit ids, NOT
    # tokenizer.bos_token_id / eos_token_id, to match the single-turn
    # script's behaviour exactly.
    if token_ids and token_ids[0] == BOS_ID:
        token_ids = token_ids[1:]
    if token_ids and token_ids[-1] == EOS_ID:
        token_ids = token_ids[:-1]
    return np.array(token_ids, dtype=np.int64)


def get_dtype_from(shard_prefixes):
    """Return the index.dtype from the first openable shard, or None."""
    for sp in shard_prefixes:
        if IndexedDataset.exists(sp):
            return IndexedDataset(sp).index.dtype
    return None



def process_input(
    input_dir, n_questions, tokenizer, builder,
    top_k, min_word_len, rng, dtype, max_doc_tokens, log_every,
    bucket_label,
):
    """
    Process every document under input_dir/dump-*/. For each:
      - decode text
      - pick n_questions distinct words from the top-K most frequent
      - build a multi-turn chat with n_questions turns
      - splice the chat tokens before the trailing </s>
      - append to the builder

    Returns (n_written, n_skipped, total_tokens).
    """
    shard_prefixes = discover_shard_prefixes(input_dir)
    if not shard_prefixes:
        logger.warning("[%s] No shards found under %s", bucket_label, input_dir)
        return 0, 0, 0

    n_written = 0
    n_skipped = 0
    total_tokens = 0

    for sp in shard_prefixes:
        if not IndexedDataset.exists(sp):
            logger.warning("[%s] Shard %s missing files, skipping", bucket_label, sp)
            continue
        ds = IndexedDataset(sp)
        doc_idx = ds.document_indices
        n_docs_shard = len(doc_idx) - 1
        logger.info("[%s]   Shard %s -- %d docs",
                    bucket_label, os.path.basename(sp), n_docs_shard)

        for did in range(n_docs_shard):
            seq_start = int(doc_idx[did])
            seq_end   = int(doc_idx[did + 1])
            seqs = ds[seq_start:seq_end]

            doc_tokens = (
                np.concatenate(seqs).astype(np.int64)
                if seqs else np.array([], dtype=np.int64)
            )

            if max_doc_tokens is not None and len(doc_tokens) > max_doc_tokens:
                n_skipped += 1
                continue

            text = tokenizer.decode(doc_tokens, skip_special_tokens=True)
            words_counts = pick_words_for_doc(
                text, n_questions, top_k, min_word_len, rng,
            )
            if words_counts is None:
                # Not enough distinct valid words -> skip
                n_skipped += 1
                continue

            chat_tokens = build_multi_turn_chat_tokens(
                tokenizer, words_counts, rng,
            )

            # Splice chat in before trailing </s>
            if len(doc_tokens) > 0 and doc_tokens[-1] == EOS_ID:
                new_tokens = np.concatenate([
                    doc_tokens[:-1], chat_tokens, doc_tokens[-1:],
                ])
            else:
                new_tokens = np.concatenate([doc_tokens, chat_tokens])

            new_tokens_typed = new_tokens.astype(dtype)
            builder.add_document(new_tokens_typed, [len(new_tokens_typed)])
            n_written += 1
            total_tokens += len(new_tokens_typed)

            if log_every and n_written % log_every == 0:
                logger.info("[%s]     %d docs written so far (%s tokens)",
                            bucket_label, n_written, fmt_tokens(total_tokens))
        del ds

    return n_written, n_skipped, total_tokens



def main():
    ap = argparse.ArgumentParser(
        description=(
            "Apply multi-turn CWE to two input buckets (hard + easy) and "
            "write a single combined output, preserving the per-source "
            "subdirectory layout."
        ),
    )
    ap.add_argument("--hard-input", required=True,
                    help="Path containing per-source subdirs for the 'hard' "
                         "bucket (default: 10 turns per doc).")
    ap.add_argument("--easy-input", required=True,
                    help="Path containing per-source subdirs for the 'easy' "
                         "bucket (default: 5 turns per doc).")
    ap.add_argument("--output-dir", required=True,
                    help="Output root. One subdir per source will be created.")
    ap.add_argument("--hard-questions", type=int, default=10,
                    help="Q/A turns per doc for the hard bucket (default: 10).")
    ap.add_argument("--easy-questions", type=int, default=5,
                    help="Q/A turns per doc for the easy bucket (default: 5).")
    ap.add_argument("--top-k", type=int, default=15,
                    help="Sample words from the top-K most frequent valid "
                         "words in each doc (default: 15). Must be >= "
                         "max(--hard-questions, --easy-questions).")
    ap.add_argument("--min-word-len", type=int, default=4,
                    help="Minimum word length (chars) to be eligible "
                         "(default: 4).")
    ap.add_argument("--max-doc-tokens", type=int, default=None,
                    help="If set, skip docs longer than this many tokens. "
                         "Default: no limit (long-context buckets need this).")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed (default: 42).")
    ap.add_argument("--sources", nargs="*", default=None,
                    help="Optional list of sub-source names to restrict to. "
                         "Default: all sources found in either input.")
    ap.add_argument("--log-every", type=int, default=2000,
                    help="Log progress every N docs written (default: 2000). "
                         "Pass 0 to disable per-batch logging.")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # Sanity checks
    if args.top_k < max(args.hard_questions, args.easy_questions):
        logger.error(
            "--top-k (%d) must be >= max(--hard-questions, --easy-questions) (%d)",
            args.top_k, max(args.hard_questions, args.easy_questions),
        )
        sys.exit(1)

    hard_input  = Path(args.hard_input).resolve()
    easy_input  = Path(args.easy_input).resolve()
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if not hard_input.is_dir():
        logger.error("Hard input does not exist: %s", hard_input); sys.exit(1)
    if not easy_input.is_dir():
        logger.error("Easy input does not exist: %s", easy_input); sys.exit(1)

    hard_sources = {p.name for p in hard_input.iterdir() if p.is_dir()}
    easy_sources = {p.name for p in easy_input.iterdir() if p.is_dir()}

    if args.sources:
        sources = sorted(set(args.sources))
    else:
        sources = sorted(hard_sources | easy_sources)

    logger.info("Configuration:")
    logger.info("  hard input : %s  (%d turns/doc)", hard_input, args.hard_questions)
    logger.info("  easy input : %s  (%d turns/doc)", easy_input, args.easy_questions)
    logger.info("  output     : %s", output_root)
    logger.info("  sources    : %s", sources)
    logger.info("  top-k=%d  min-word-len=%d  seed=%d",
                args.top_k, args.min_word_len, args.seed)

    logger.info("Loading tokenizer %s ...", TOKENIZER_NAME)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    rng = np.random.default_rng(args.seed)

    grand_docs = 0
    grand_tokens = 0
    grand_skipped = 0

    for src in sources:
        t0 = time.perf_counter()
        logger.info("=" * 70)
        logger.info("Source: %s", src)
        logger.info("=" * 70)

        hard_dir = hard_input / src
        easy_dir = easy_input / src

        hard_prefixes = discover_shard_prefixes(hard_dir) if hard_dir.is_dir() else []
        easy_prefixes = discover_shard_prefixes(easy_dir) if easy_dir.is_dir() else []

        if not hard_prefixes and not easy_prefixes:
            logger.warning("Neither hard nor easy has shards for %s -- skipping.", src)
            continue

        dtype = get_dtype_from(hard_prefixes) or get_dtype_from(easy_prefixes)
        if dtype is None:
            logger.warning("Could not determine dtype for %s, skipping.", src)
            continue

        out_prefix = output_root / src / "dump-0" / "00000_tokens"
        out_prefix.parent.mkdir(parents=True, exist_ok=True)
        builder = IndexedDatasetBuilder(str(out_prefix) + ".bin", dtype=dtype)

        if hard_prefixes:
            logger.info("Processing HARD (%d turns/doc) for %s ...",
                        args.hard_questions, src)
            nw, ns, nt = process_input(
                hard_dir, args.hard_questions, tokenizer, builder,
                args.top_k, args.min_word_len, rng, dtype,
                args.max_doc_tokens, args.log_every, "HARD",
            )
            logger.info("HARD %s: %d written, %d skipped, %s tokens",
                        src, nw, ns, fmt_tokens(nt))
            grand_docs += nw
            grand_tokens += nt
            grand_skipped += ns
        else:
            logger.warning("No HARD input for %s", src)

        if easy_prefixes:
            logger.info("Processing EASY (%d turns/doc) for %s ...",
                        args.easy_questions, src)
            nw, ns, nt = process_input(
                easy_dir, args.easy_questions, tokenizer, builder,
                args.top_k, args.min_word_len, rng, dtype,
                args.max_doc_tokens, args.log_every, "EASY",
            )
            logger.info("EASY %s: %d written, %d skipped, %s tokens",
                        src, nw, ns, fmt_tokens(nt))
            grand_docs += nw
            grand_tokens += nt
            grand_skipped += ns
        else:
            logger.warning("No EASY input for %s", src)

        builder.finalize(str(out_prefix) + ".idx")
        logger.info("Finalised %s.{bin,idx} in %.1fs",
                    out_prefix, time.perf_counter() - t0)

    logger.info("=" * 70)
    logger.info("ALL DONE")
    logger.info("  Total docs written : %d", grand_docs)
    logger.info("  Total tokens       : %s", fmt_tokens(grand_tokens))
    logger.info("  Total skipped      : %d", grand_skipped)


if __name__ == "__main__":
    main()