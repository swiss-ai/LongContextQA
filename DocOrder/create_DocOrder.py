"""
Sample N documents from an MMap indexed dataset, then for each document:
  1. Detect section boundaries via newline-separator tokens.
  2. Randomly pick M sections (M drawn uniformly from [min_sections, max_sections])
     and shuffle them, prepending "The following text is Section j\n" headers.
  3. Embed the shuffled document INSIDE the user turn of a section-ordering QA
     chat (formatted with the model's chat template).
 
The assistant answer lists the ORIGINAL section indices in the order they should
appear, e.g. "The correct order of the sections is [3, 1, 2]."
 
Output layout (flat):
    <output_dir>/<source_name>/dump-0/00000_tokens.{idx,bin}
 
The final token sequence looks like:
    <s>
    <|system_start|>You are a helpful assistant. Be concise.<|system_end|>
    <|developer_start|>Deliberation: disabled\nTool Capabilities: disabled<|developer_end|>
    <|user_start|>
    The sections of this document may appear in a different order than in the
    original source; please list them in the order in which they should appear.
    <shuffled document text>
    <|user_end|>
    <|assistant_start|>The correct order of the sections is [2, 1, 4, 3].<|assistant_end|>
    </s>
 
Usage:
    python create_dso.py \\
        --source testing \\
        --n-docs 10000 \\
        --output-dir /path/to/output \\
        --seed 42
"""
 
import argparse
import logging
import os
import sys
import time
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
    "finepdfs-edu-preprocessed-general":
        "/capstor/scratch/cscs/dtamayomela/data/corrected_data/finepdfs-edu-preprocessed/second_half_longcontext",
    "finetranslations":
        "/capstor/scratch/cscs/dtamayomela/data/corrected_data/finetranslations/second_half_longcontext",
    "swissai-fineweb-2_0_1-quality_10-filterrobots":
        "/capstor/scratch/cscs/dtamayomela/data/corrected_data/swissai-fineweb-2_0_1-quality_10-filterrobots/second_half_longcontext",
    "testing":
        "/capstor/scratch/cscs/dtamayomela/synthetic_data/data_seed_example",
    "simple_tokenization":
        "/capstor/scratch/cscs/dtamayomela/data/simple_tokenization/tokenized_output",
}
 
TOKENIZER_NAME = "swiss-ai/Apertus-8B-Instruct-2509"
 
BOS_ID = 1   # <s>
EOS_ID = 2   # </s>
 
  
def fmt_tokens(n):
    if n >= 1e9: return f"{n / 1e9:.3f}B"
    if n >= 1e6: return f"{n / 1e6:.1f}M"
    if n >= 1e3: return f"{n / 1e3:.1f}K"
    return str(n)
 
 
def discover_shard_prefixes(data_dir):
    """
    Discover all IndexedDataset shard prefixes under data_dir.
 
    Accepted naming conventions (both relative to a dump-* subdirectory):
      - <name>_tokens.idx   (standard Megatron layout)
      - <name>.idx          (flat layout, e.g. tokenized_output.idx)
    """
    root = Path(data_dir)
    prefixes = []
    for dump in sorted(root.glob("dump-*")):
        if not dump.is_dir():
            continue
        seen = set()
        for idx in sorted(dump.glob("*_tokens.idx")):
            prefix = str(idx)[:-4]
            if prefix not in seen:
                seen.add(prefix)
                prefixes.append(prefix)
        for idx in sorted(dump.glob("*.idx")):
            prefix = str(idx)[:-4]
            if prefix not in seen:
                seen.add(prefix)
                prefixes.append(prefix)
    return prefixes
 
 
def sample_doc_indices(shard_prefixes, n_docs, rng):
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
 
 
def build_newline_token_set(tokenizer):
    """
    Return a plain Python int set of token ids that correspond to one or more
    consecutive newline characters (i.e. section separators).
    """
    anchor = "hello"
    sep_token_ids = set()
    anchor_ids = tokenizer(anchor)["input_ids"]
    if anchor_ids and anchor_ids[0] == BOS_ID:
        anchor_ids = anchor_ids[1:]
    anchor_tok = int(anchor_ids[0])
    for n in range(1, 9):
        text = anchor + "\n" * n + anchor
        ids = tokenizer(text)["input_ids"]
        body = ids[1:] if (ids and ids[0] == BOS_ID) else ids
        positions = [i for i, t in enumerate(body) if int(t) == anchor_tok]
        if len(positions) >= 2:
            for mid in body[positions[0] + 1: positions[1]]:
                sep_token_ids.add(int(mid))
    return sep_token_ids
 
 
def split_into_sections(doc_tokens, sep_token_ids):
    """
    Split doc_tokens on any token in sep_token_ids.
 
    Returns:
        sections   – list of np.int64 arrays (content tokens, no separator)
        separators – list of int token ids (length = len(sections) - 1)
        leading_bos  – list of ints ([] or [BOS_ID])
        trailing_eos – list of ints ([] or [EOS_ID])
    """
    token_list = [int(t) for t in doc_tokens]
 
    leading_bos: list = []
    trailing_eos: list = []
    if token_list and token_list[0] == BOS_ID:
        leading_bos = token_list[:1]
        token_list = token_list[1:]
    if token_list and token_list[-1] == EOS_ID:
        trailing_eos = token_list[-1:]
        token_list = token_list[:-1]
 
    sections: list = []
    separators: list = []
    current: list = []
    for tok in token_list:
        if tok in sep_token_ids:
            sections.append(np.array(current, dtype=np.int64))
            separators.append(tok)
            current = []
        else:
            current.append(tok)
    sections.append(np.array(current, dtype=np.int64))
 
    return sections, separators, leading_bos, trailing_eos
 
  
def section_header_tokens(tokenizer, section_number):
    """
    Tokenise "\nThe following text is Section {section_number}:\n" without BOS.
    """
    text = f"\nThe following text is Section {section_number}:\n"
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    return np.array(ids, dtype=np.int64)
 
  
def tokens_to_array(parts):
    """Flatten a list of (int | list[int] | np.ndarray) into one np.int64 array."""
    arrays = []
    for p in parts:
        if isinstance(p, np.ndarray):
            if p.size:
                arrays.append(p.astype(np.int64))
        elif isinstance(p, list):
            if p:
                arrays.append(np.array(p, dtype=np.int64))
        else:
            arrays.append(np.array([int(p)], dtype=np.int64))
    return np.concatenate(arrays) if arrays else np.array([], dtype=np.int64)
 
 
def shuffle_sections(
    doc_tokens, sep_token_ids, tokenizer, rng,
    min_sections, max_sections,
):
    """
    Partition the document into M chunks, shuffle them, prepend section headers,
    and return the reassembled token sequence together with the answer key.
 
    CHANGED from original: BOS and EOS are STRIPPED from the returned token
    array. The caller is responsible for wrapping everything inside the chat
    template (which provides its own BOS/EOS).
 
    Returns:
        (shuffled_body_tokens, correct_order)
        shuffled_body_tokens - np.int64 array, NO leading BOS, NO trailing EOS.
        correct_order        - list of 1-based ints giving the reading order.
        (None, None)         - if the document has fewer than min_sections sections.
    """
    sections, separators, _leading_bos, _trailing_eos = split_into_sections(
        doc_tokens, sep_token_ids
    )
    # NOTE: We intentionally discard leading_bos and trailing_eos here.
    # The BOS/EOS for the full example will come from apply_chat_template.
 
    n_atomic = len(sections)
    if n_atomic < min_sections:
        return None, None
 
    # Choose M (number of chunks)
    m = int(rng.integers(min_sections, min(max_sections, n_atomic) + 1))
 
    # Choose M-1 cut points from the N-1 inter-section boundaries
    boundary_count = n_atomic - 1
    cut_positions = sorted(
        rng.choice(boundary_count, size=m - 1, replace=False).tolist()
    )
    boundaries = [0] + [p + 1 for p in cut_positions] + [n_atomic]
 
    # Build chunks
    chunks = []
    for k in range(m):
        lo, hi = boundaries[k], boundaries[k + 1]
        chunk_parts = []
        for i in range(lo, hi):
            chunk_parts.append(sections[i])
            if i < hi - 1 and i < len(separators):
                chunk_parts.append(np.array([separators[i]], dtype=np.int64))
        chunks.append(chunk_parts)
 
    # Shuffle chunks (ensure genuine permutation)
    shuffled_order = list(range(m))
    for _ in range(100):
        rng.shuffle(shuffled_order)
        if shuffled_order != list(range(m)):
            break
 
    display_label_for = [0] * m
    for display_pos, orig_idx in enumerate(shuffled_order):
        display_label_for[orig_idx] = display_pos + 1
    correct_order = [display_label_for[k] for k in range(m)]
 
    # Reassemble — NO BOS/EOS; those come from the chat template
    parts = []
    for display_pos, orig_idx in enumerate(shuffled_order):
        label = display_pos + 1
        parts.append(section_header_tokens(tokenizer, label))
        parts.extend(chunks[orig_idx])
 
    return tokens_to_array(parts), correct_order

 
def build_full_example_tokens(tokenizer, shuffled_body_tokens, correct_order):
    """
    Build the complete token sequence for one training example.
 
    Layout:
        <s>
        <|system_start|>You are a helpful assistant. Be concise.<|system_end|>
        <|developer_start|>Deliberation: disabled
        Tool Capabilities: disabled<|developer_end|>
        <|user_start|>
        The sections of this document may appear in a different order than in
        the original source; please list them in the order in which they should
        appear.
        <shuffled document tokens decoded and re-embedded here>
        <|user_end|>
        <|assistant_start|>The correct order of the sections is [2, 1, 4, 3].<|assistant_end|>
        </s>
 
    Strategy
    --------
    We decode the shuffled body tokens back to text and embed that text
    directly inside the user message string before calling
    `apply_chat_template`.  This avoids fragile manual token-level stitching
    and lets the tokenizer handle special-token insertion correctly.
 
    The decode → re-encode round-trip is safe here because:
      - The shuffled body was itself built by concatenating tokenizer outputs,
        so every token id is within vocabulary.
      - We skip_special_tokens=False so structural tokens (if any) survive,
        but BOS/EOS were already stripped by shuffle_sections.
 
    Args:
        tokenizer           - HF tokenizer with apply_chat_template support.
        shuffled_body_tokens - np.int64 array; NO BOS, NO EOS.
        correct_order       - list of 1-based ints, e.g. [3, 1, 2].
 
    Returns:
        np.int64 array of the complete example token ids (includes BOS/EOS
        as produced by apply_chat_template).
    """
    # decode shuffled body back to text 
    # skip_special_tokens=False: we stripped BOS/EOS already, and we want
    # any other vocabulary tokens (e.g. separator tokens) to round-trip.
    shuffled_text = tokenizer.decode(
        shuffled_body_tokens.tolist(),
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
 
    # build the answer string 
    order_str = "[" + ", ".join(str(i) for i in correct_order) + "]"
 
    # assemble messages 
    # The shuffled document is appended to the user message content so it
    # appears INSIDE the <|user_start|>…<|user_end|> span.
    user_instruction = (
        "The sections of this document may appear in a different order "
        "than in the original source; please list them in the order in "
        "which they should appear."
    )
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Be concise.",
        },
        {
            "role": "user",
            # Instruction first, then the document body — both inside the
            # user turn.  A blank line separates instruction from document.
            "content": f"{user_instruction}\n{shuffled_text}",
        },
        {
            "role": "assistant",
            "content": f"The correct order of the sections is {order_str}.",
        },
    ]
 
    token_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
    )

    if not token_ids or token_ids[-1] != EOS_ID:
        token_ids = token_ids + [EOS_ID]
    return np.array(token_ids, dtype=np.int64)
 
  
def process_and_write(
    shard_prefixes, sampled, tokenizer, sep_token_ids,
    output_dir, source_name, rng,
    min_sections, max_sections,
):
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
                if len(doc_tokens) > 32_768 or len(doc_tokens) < 8_000:
                    skipped_docs += 1
                    continue
 
                #  shuffle sections (returns body only, no BOS/EOS) 
                shuffled_body, correct_order = shuffle_sections(
                    doc_tokens, sep_token_ids, tokenizer, rng,
                    min_sections, max_sections,
                )
 
                if shuffled_body is None:
                    logger.warning(
                        "Too few sections in doc (shard=%d, doc=%d) — skipping.",
                        si, did,
                    )
                    skipped_docs += 1
                    continue
 
                # --- build full example: document embedded inside user turn --
                # CHANGED: replaces the old two-step (prepend shuffled tokens,
                # then append chat tokens).  Now the shuffled text is decoded
                # and placed inside the user message before apply_chat_template
                # is called, so the final sequence is:
                #
                #   BOS  system  developer  user[instruction + doc]  assistant  EOS
                #
                new_tokens = build_full_example_tokens(
                    tokenizer, shuffled_body, correct_order
                )
 
                new_tokens_typed = new_tokens.astype(dtype)
                builder.add_document(new_tokens_typed, [len(new_tokens_typed)])
                total_docs   += 1
                total_tokens += len(new_tokens_typed)
 
            i = j + 1
        del ds
 
    builder.finalize(prefix + ".idx")
    logger.info(
        "Wrote %d docs, %s tokens → %s  (%d skipped — too few sections)",
        total_docs, fmt_tokens(total_tokens), prefix, skipped_docs,
    )
 
 
# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
 
def main():
    ap = argparse.ArgumentParser(
        description="Sample N docs, shuffle sections, embed doc in user turn, write new binary."
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
        "--min-sections", type=int, default=2,
        help="Minimum sections required and minimum M for shuffle (default: 2).",
    )
    ap.add_argument(
        "--max-sections", type=int, default=6,
        help="Maximum M sections to include in each shuffle task (default: 6).",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Run phase 1 only (sampling), print stats, then stop.",
    )
    args = ap.parse_args()
 
    if args.min_sections < 2:
        ap.error("--min-sections must be at least 2")
    if args.max_sections < args.min_sections:
        ap.error("--max-sections must be >= --min-sections")
 
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
 
    logger.info("Phase 2: loading tokenizer and detecting separator tokens …")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    sep_token_ids = build_newline_token_set(tokenizer)
    logger.info("  Separator token ids: %s", sorted(sep_token_ids))
 
    logger.info("Phase 2: shuffling sections, injecting chats, writing …")
    t2 = time.perf_counter()
 
    process_and_write(
        shard_prefixes, sampled, tokenizer, sep_token_ids,
        output_dir, args.source, rng,
        min_sections=args.min_sections,
        max_sections=args.max_sections,
    )
 
    logger.info("Phase 2 done in %.1fs", time.perf_counter() - t2)
    logger.info("All done.")
 
 
if __name__ == "__main__":
    main()