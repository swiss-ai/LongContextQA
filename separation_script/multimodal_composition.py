#!/usr/bin/env python3
"""
Sample documents from MMap indexed datasets into sequence-length buckets.

Each source is processed exactly once: a single worker reads every shard
sequentially and fans out selected documents to all sampling configs that
need them.  No temp files, no merge step - minimum possible I/O.

Source spec accepts three forms:
    "Source": int                                - single draw, mix default bucket
    "Source": {"target": N, "bucket": "name"}    - single draw, override bucket
    "Source": [ {target,bucket}, {target,bucket} ] - multiple draws concatenated

When the same (Source, bucket) is requested by several mixes, the script
runs ONE random permutation of that pool and slices it into disjoint chunks
per mix - so no document is repeated across mixes for shared pools.

Output layout:
    <output_dir>/<bucket_dir>/<source_name>/dump-0/00000_tokens.{idx,bin}

Usage:
    python create_lct_buckets.py --output-dir /path/to/output --seed 42
    python create_lct_buckets.py --output-dir /path/to/output --dry-run
"""

import argparse
import logging
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

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
        "/capstor/scratch/cscs/dtamayomela/data/corrected_data/finepdfs-edu-preprocessed/second_half_longcontext",
    "finetranslations":
        "/capstor/scratch/cscs/dtamayomela/data/corrected_data/finetranslations/second_half_longcontext",
    "swissai-fineweb-2_0_1-quality_10-filterrobots":
        "/capstor/scratch/cscs/dtamayomela/data/corrected_data/swissai-fineweb-2_0_1-quality_10-filterrobots/second_half_longcontext",
    "Audio":
        "/capstor/store/cscs/swissai/infra01/audio-datasets/Apertus1p5_lct_tokenized",
    "Image":
        "/capstor/store/cscs/swissai/infra01/vision-datasets/Apertus1p5_lcp_tokenized",
}

# Half-open intervals [lo, hi) on per-document token count.
BUCKET_BOUNDS = {
    "lt_16k":    (   1_000,  16_384),   # short-context pool, used for image top-up
    "8k_16k":    (   8_192,  16_384),   # short-context text pool for cwe/hard
    "16k_32k":   (  16_384,  32_768),
    "32k_64k":   (  32_768,  65_536),
    "64k_128k":  (  65_536, 131_072),
    "128k_256k": ( 131_072, 262_144),
    "any":       (   1_000, 1_000_000_000),
}

# Each mix targets 20B tokens with composition:
#   33% multimodal - image 80% / audio 20%   (target: image 5.33B, audio 1.33B)
#   66% text       - English 60% / multilingual 40%  (target: EN 8.00B, ML 5.33B)
#
# Image availability in the native long-context buckets (4.2 / 2.5 / 1.8B)
# is well below the 5.33B target, so each long-context mix tops up image
# tokens by drawing from the <16k pool (13.4B available).
#
# Total <16k image consumed: 1.13 + 2.83 + 3.53 = 7.49B (of 13.4B available).
# Disjoint sampling across mixes is enforced automatically by the runner.
#
# English text (8.00B): FinePDFs-Edu, OlmoCR-Science, Biomed-Enriched,
#                       Institutional-Books
# Multilingual (5.33B): FineWeb-2, FineTranslations, FinePDFs-Edu-ML
#
# Long-context buckets shift EN allocation toward Institutional-Books and
# FinePDFs-Edu where most long documents live, and ML toward FineWeb-2 and
# FinePDFs-Multilingual since FineTranslations has a thin long-context tail.
#
# Each long-context bucket produces THREE output mixes:
#   <name>             - 17B (10.33B text + 6.67B multimodal)
#   <name>_cwe/easy    - 1.5B text-only, same source ratios as text in <name>
#   <name>_cwe/hard    - 1.5B text-only, same source ratios as text in <name>
#
# Plus one extra short-context mix:
#   mix_8k_16k_cwe/hard - 1.5B text-only, drawn from the 8k–16k bucket
#
# Splitting works because the runner samples disjoint document slices for any
# (source, bucket) pool requested by multiple mixes. So <name>, easy, and hard
# share statistics (length distribution, source mix) without overlapping docs.
# Total drawn per text pool is unchanged from the single-mix design
# (mixed text + easy text + hard text == original text target).

SAMPLING_CONFIG = {
    # ===================
    # 16k–32k bucket
    # ===================

    # 17B mixed (text + multimodal). Text scaled by 0.775 from baseline.
    "mix_16k_32k": {
        "bucket_dir": "mix_16k_32k",
        "bucket": "16k_32k",
        "sources": {
            # Multilingual text (4.14B)
            "swissai-fineweb-2_0_1-quality_10-filterrobots": 1_380_000_000,
            "finetranslations":                              1_380_000_000,
            "finepdfs-edu-multilingual-preprocessed":        1_380_000_000,
            # English text (6.20B)
            "finepdfs-edu-preprocessed":                     2_840_000_000,
            "dolma3_olmocr_science_pdfs-preprocessed":       2_840_000_000,
            "Biomed-Enriched_preprocessed":                    520_000_000,
            # Audio (1.33B)
            "Audio":                                         1_330_000_000,
            # Image (5.33B = 4.20B native + 1.13B from <16k)
            "Image": [
                {"target": 4_200_000_000, "bucket": "16k_32k"},
                {"target": 1_130_000_000, "bucket": "lt_16k"},
            ],
        },
    },
    # 1.5B text-only siblings (3B total). Halved from the previous _text config.
    # easy and hard get disjoint random slices of the same pool.
    "mix_16k_32k_cwe_easy": {
        "bucket_dir": "mix_16k_32k_cwe/easy",
        "bucket": "16k_32k",
        "sources": {
            # Multilingual text (0.60B)
            "swissai-fineweb-2_0_1-quality_10-filterrobots":   200_000_000,
            "finetranslations":                                200_000_000,
            "finepdfs-edu-multilingual-preprocessed":          200_000_000,
            # English text (0.91B)
            "finepdfs-edu-preprocessed":                       415_000_000,
            "dolma3_olmocr_science_pdfs-preprocessed":         415_000_000,
            "Biomed-Enriched_preprocessed":                     75_000_000,
        },
    },
    "mix_16k_32k_cwe_hard": {
        "bucket_dir": "mix_16k_32k_cwe/hard",
        "bucket": "16k_32k",
        "sources": {
            "swissai-fineweb-2_0_1-quality_10-filterrobots":   200_000_000,
            "finetranslations":                                200_000_000,
            "finepdfs-edu-multilingual-preprocessed":          200_000_000,
            "finepdfs-edu-preprocessed":                       415_000_000,
            "dolma3_olmocr_science_pdfs-preprocessed":         415_000_000,
            "Biomed-Enriched_preprocessed":                     75_000_000,
        },
    },

    # ===================
    # 32k–64k bucket
    # ===================

    "mix_32k_64k": {
        "bucket_dir": "mix_32k_64k",
        "bucket": "32k_64k",
        "sources": {
            # Multilingual text (4.14B)
            "swissai-fineweb-2_0_1-quality_10-filterrobots": 1_550_000_000,
            "finepdfs-edu-multilingual-preprocessed":        1_810_000_000,
            "finetranslations":                                780_000_000,
            # English text (6.19B)
            "dolma3_olmocr_science_pdfs-preprocessed":       2_710_000_000,
            "finepdfs-edu-preprocessed":                     2_710_000_000,
            "institutional-books-1.0-filtered":                540_000_000,
            "Biomed-Enriched_preprocessed":                    230_000_000,
            # Audio (1.33B)
            "Audio":                                         1_330_000_000,
            # Image (5.33B = 2.50B native + 2.83B from <16k)
            "Image": [
                {"target": 2_500_000_000, "bucket": "32k_64k"},
                {"target": 2_830_000_000, "bucket": "lt_16k"},
            ],
        },
    },
    "mix_32k_64k_cwe_easy": {
        "bucket_dir": "mix_32k_64k_cwe/easy",
        "bucket": "32k_64k",
        "sources": {
            # Multilingual text (0.60B)
            "swissai-fineweb-2_0_1-quality_10-filterrobots":   225_000_000,
            "finepdfs-edu-multilingual-preprocessed":          260_000_000,
            "finetranslations":                                110_000_000,
            # English text (0.90B)
            "dolma3_olmocr_science_pdfs-preprocessed":         395_000_000,
            "finepdfs-edu-preprocessed":                       395_000_000,
            "institutional-books-1.0-filtered":                 80_000_000,
            "Biomed-Enriched_preprocessed":                     35_000_000,
        },
    },
    "mix_32k_64k_cwe_hard": {
        "bucket_dir": "mix_32k_64k_cwe/hard",
        "bucket": "32k_64k",
        "sources": {
            "swissai-fineweb-2_0_1-quality_10-filterrobots":   225_000_000,
            "finepdfs-edu-multilingual-preprocessed":          260_000_000,
            "finetranslations":                                110_000_000,
            "dolma3_olmocr_science_pdfs-preprocessed":         395_000_000,
            "finepdfs-edu-preprocessed":                       395_000_000,
            "institutional-books-1.0-filtered":                 80_000_000,
            "Biomed-Enriched_preprocessed":                     35_000_000,
        },
    },

    # ===================
    # 64k–128k bucket
    # ===================

    "mix_64k_128k": {
        "bucket_dir": "mix_64k_128k",
        "bucket": "64k_128k",
        "sources": {
            # Multilingual text (4.14B)
            "swissai-fineweb-2_0_1-quality_10-filterrobots": 1_940_000_000,
            "finepdfs-edu-multilingual-preprocessed":        1_940_000_000,
            "finetranslations":                                260_000_000,
            # English text (6.20B)
            "finepdfs-edu-preprocessed":                     3_260_000_000,
            "institutional-books-1.0-filtered":              2_710_000_000,
            "dolma3_olmocr_science_pdfs-preprocessed":         230_000_000,
            # Audio (1.33B)
            "Audio":                                         1_330_000_000,
            # Image (5.33B = 1.80B native + 3.53B from <16k)
            "Image": [
                {"target": 1_800_000_000, "bucket": "64k_128k"},
                {"target": 3_530_000_000, "bucket": "lt_16k"},
            ],
        },
    },
    "mix_64k_128k_cwe_easy": {
        "bucket_dir": "mix_64k_128k_cwe/easy",
        "bucket": "64k_128k",
        "sources": {
            # Multilingual text (0.60B)
            "swissai-fineweb-2_0_1-quality_10-filterrobots":   280_000_000,
            "finepdfs-edu-multilingual-preprocessed":          280_000_000,
            "finetranslations":                                 35_000_000,
            # English text (0.90B)
            "finepdfs-edu-preprocessed":                       470_000_000,
            "institutional-books-1.0-filtered":                395_000_000,
            "dolma3_olmocr_science_pdfs-preprocessed":          35_000_000,
        },
    },
    "mix_64k_128k_cwe_hard": {
        "bucket_dir": "mix_64k_128k_cwe/hard",
        "bucket": "64k_128k",
        "sources": {
            "swissai-fineweb-2_0_1-quality_10-filterrobots":   280_000_000,
            "finepdfs-edu-multilingual-preprocessed":          280_000_000,
            "finetranslations":                                 35_000_000,
            "finepdfs-edu-preprocessed":                       470_000_000,
            "institutional-books-1.0-filtered":                395_000_000,
            "dolma3_olmocr_science_pdfs-preprocessed":          35_000_000,
        },
    },

    # ===================
    # 128k–256k bucket
    # ===================

    "mix_128k_256k": {
        "bucket_dir": "mix_128k_256k",
        "bucket": "128k_256k",
        "sources": {
            # Multilingual text (4.13B)
            "finepdfs-edu-multilingual-preprocessed":        2_330_000_000,
            "swissai-fineweb-2_0_1-quality_10-filterrobots": 1_650_000_000,
            "finetranslations":                                150_000_000,
            # English text (6.21B)
            "institutional-books-1.0-filtered":              3_490_000_000,
            "finepdfs-edu-preprocessed":                     2_640_000_000,
            "Biomed-Enriched_preprocessed":                     80_000_000,
            # Audio (1.33B), Image (5.33B all native)
            "Audio":                                         1_330_000_000,
            "Image":                                         5_330_000_000,
        },
    },
    "mix_128k_256k_cwe_easy": {
        "bucket_dir": "mix_128k_256k_cwe/easy",
        "bucket": "128k_256k",
        "sources": {
            # Multilingual text (0.60B)
            "finepdfs-edu-multilingual-preprocessed":          335_000_000,
            "swissai-fineweb-2_0_1-quality_10-filterrobots":   240_000_000,
            "finetranslations":                                 25_000_000,
            # English text (0.90B)
            "institutional-books-1.0-filtered":                505_000_000,
            "finepdfs-edu-preprocessed":                       385_000_000,
            "Biomed-Enriched_preprocessed":                     10_000_000,
        },
    },
    "mix_128k_256k_cwe_hard": {
        "bucket_dir": "mix_128k_256k_cwe/hard",
        "bucket": "128k_256k",
        "sources": {
            "finepdfs-edu-multilingual-preprocessed":          335_000_000,
            "swissai-fineweb-2_0_1-quality_10-filterrobots":   240_000_000,
            "finetranslations":                                 25_000_000,
            "institutional-books-1.0-filtered":                505_000_000,
            "finepdfs-edu-preprocessed":                       385_000_000,
            "Biomed-Enriched_preprocessed":                     10_000_000,
        },
    },

    # ===================
    # 8k–16k bucket (extra short-context text-only mix, hard variant only)
    # ===================
    # 1.5B text-only at 8k–16k. Source ratios follow the 16k_32k baseline
    # since that's the closest neighbouring length distribution.
    # Text pool here is independent from the (Image, lt_16k) pool used for
    # multimodal top-up - different sources.
    "mix_8k_16k_cwe_hard": {
        "bucket_dir": "mix_8k_16k_cwe/hard",
        "bucket": "8k_16k",
        "sources": {
            # Multilingual text (0.60B)
            "swissai-fineweb-2_0_1-quality_10-filterrobots":   200_000_000,
            "finetranslations":                                200_000_000,
            "finepdfs-edu-multilingual-preprocessed":          200_000_000,
            # English text (0.90B)
            "finepdfs-edu-preprocessed":                       415_000_000,
            "dolma3_olmocr_science_pdfs-preprocessed":         415_000_000,
            "Biomed-Enriched_preprocessed":                     70_000_000,
        },
    },
}


def fmt_tokens(n):
    if n >= 1e9:  return f"{n / 1e9:.3f}B"
    if n >= 1e6:  return f"{n / 1e6:.1f}M"
    if n >= 1e3:  return f"{n / 1e3:.1f}K"
    return str(n)


def discover_shard_prefixes(data_dir):
    """Find all shard prefixes (paths to .idx without extension).

    Supports two layouts:
      1. Nested: <root>/dump-*/*_tokens.idx           (text sources)
      2. Flat:   <root>/*.idx                          (Image, Audio)
    """
    root = Path(data_dir)
    prefixes = []
    # Layout 1: dump-*/...
    for dump in sorted(root.glob("dump-*")):
        if dump.is_dir():
            for idx in sorted(dump.glob("*_tokens.idx")):
                prefixes.append(str(idx)[:-4])
    if prefixes:
        return prefixes
    # Layout 2: flat directory of .idx/.bin pairs
    for idx in sorted(root.glob("*.idx")):
        prefixes.append(str(idx)[:-4])
    return prefixes


def compute_doc_token_counts(dataset):
    starts = dataset.document_indices[:-1].astype(np.intp)
    return np.add.reduceat(dataset.sequence_lengths.astype(np.int64), starts)


def normalize_source_spec(spec, default_bucket):
    """Convert a source value into a list of {target, bucket} dicts."""
    if isinstance(spec, int):
        return [{"target": int(spec), "bucket": default_bucket}]
    if isinstance(spec, dict):
        return [{"target": int(spec["target"]),
                 "bucket": spec.get("bucket", default_bucket)}]
    return [{"target": int(s["target"]),
             "bucket": s.get("bucket", default_bucket)} for s in spec]


def index_source(source_name, data_dir):
    """Return (shard_prefixes, records_array[N,3]: shard_idx|doc_id|tokens)."""
    t0 = time.perf_counter()
    shard_prefixes = discover_shard_prefixes(data_dir)
    if not shard_prefixes:
        logger.warning("[%s] no shards under %s", source_name, data_dir)
        return [], np.empty((0, 3), dtype=np.int64)

    chunks = []
    for si, prefix in enumerate(shard_prefixes):
        if not IndexedDataset.exists(prefix):
            continue
        ds = IndexedDataset(prefix)
        tc = compute_doc_token_counts(ds)
        n = len(tc)
        c = np.empty((n, 3), dtype=np.int64)
        c[:, 0] = si
        c[:, 1] = np.arange(n, dtype=np.int64)
        c[:, 2] = tc
        chunks.append(c)
        del ds

    records = np.concatenate(chunks) if chunks else np.empty((0, 3), dtype=np.int64)
    logger.info("[%s] %d docs, %d shards, %.1fs",
                source_name, len(records), len(shard_prefixes),
                time.perf_counter() - t0)
    return shard_prefixes, records


def process_source(source_name, shard_prefixes, config_specs, output_dir):
    """Read every shard of one source ONCE, write to all config outputs.

    Args:
        source_name:    Name of the source dataset.
        shard_prefixes: List of shard path prefixes.
        config_specs:   List of dicts, each with:
            - config_name:  str (for logging)
            - bucket_dir:   str (output subdirectory)
            - target_tokens: int
            - docs_by_shard: dict[int, np.ndarray]  (shard_idx → sorted doc IDs)
        output_dir:     Root output directory.
    """
    t0 = time.perf_counter()

    # Detect dtype from first available shard
    dtype = None
    for sp in shard_prefixes:
        if IndexedDataset.exists(sp):
            dtype = IndexedDataset(sp).index.dtype
            break
    if dtype is None:
        logger.warning("[%s] no valid shards", source_name)
        return

    # Open one builder per config
    builders = []
    for cfg in config_specs:
        prefix = os.path.join(output_dir, cfg["bucket_dir"], source_name,
                              "dump-0", "00000_tokens")
        os.makedirs(os.path.dirname(prefix), exist_ok=True)
        builders.append({
            "builder":  IndexedDatasetBuilder(prefix + ".bin", dtype=dtype),
            "prefix":   prefix,
            "name":     cfg["config_name"],
            "target":   cfg["target_tokens"],
            "n_docs":   0,
            "n_tokens": 0,
        })

    # Scan each shard once
    for si, shard_prefix in enumerate(shard_prefixes):
        if not IndexedDataset.exists(shard_prefix):
            continue

        # Build reverse map: doc_id → [builder indices]
        doc_to_builders = defaultdict(list)
        for ci, cfg in enumerate(config_specs):
            for did in cfg["docs_by_shard"].get(si, []):
                doc_to_builders[did].append(ci)

        if not doc_to_builders:
            continue

        ds = IndexedDataset(shard_prefix)
        doc_idx = ds.document_indices
        all_docs = sorted(doc_to_builders)

        # Read contiguous runs for sequential I/O
        i = 0
        while i < len(all_docs):
            run_start = all_docs[i]
            run_end = run_start
            while i + 1 < len(all_docs) and all_docs[i + 1] == run_end + 1:
                i += 1
                run_end = all_docs[i]
            i += 1

            seq_start = int(doc_idx[run_start])
            seq_end = int(doc_idx[run_end + 1])
            all_seqs = ds[seq_start:seq_end]

            for d in range(run_start, run_end + 1):
                lo = int(doc_idx[d]) - seq_start
                hi = int(doc_idx[d + 1]) - seq_start
                seqs = all_seqs[lo:hi]
                lengths = [len(s) for s in seqs]
                data = np.concatenate(seqs)
                tok = sum(lengths)

                for ci in doc_to_builders[d]:
                    builders[ci]["builder"].add_document(data, lengths)
                    builders[ci]["n_docs"] += 1
                    builders[ci]["n_tokens"] += tok

        del ds

    # Finalise all builders
    elapsed = time.perf_counter() - t0
    for b in builders:
        b["builder"].finalize(b["prefix"] + ".idx")
        pct = 100.0 * b["n_tokens"] / b["target"] if b["target"] else 0
        logger.info("[%s/%s] %d docs, %s tokens (%.1f%% of target)",
                    b["name"], source_name, b["n_docs"],
                    fmt_tokens(b["n_tokens"]), pct)

    logger.info("[%s] finished in %.1fs", source_name, elapsed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    output_dir = str(Path(args.output_dir).resolve())
    os.makedirs(output_dir, exist_ok=True)

    needed_sources = set()
    for bcfg in SAMPLING_CONFIG.values():
        needed_sources.update(bcfg["sources"])
    for s in sorted(needed_sources):
        if s not in DATASETS:
            logger.error("'%s' not in DATASETS", s)
            sys.exit(1)

    logger.info("Phase 1: indexing %d sources …", len(needed_sources))
    t1 = time.perf_counter()
    catalogue = {}

    with ProcessPoolExecutor(max_workers=len(needed_sources)) as pool:
        futs = {pool.submit(index_source, s, DATASETS[s]): s
                for s in sorted(needed_sources)}
        for f in as_completed(futs):
            s = futs[f]
            sp, rec = f.result()
            catalogue[s] = {"shard_prefixes": sp, "records": rec}

    logger.info("Phase 1 done in %.1fs\n", time.perf_counter() - t1)

    logger.info("=" * 90)
    for bname, bcfg in SAMPLING_CONFIG.items():
        default_bucket = bcfg["bucket"]
        mix_total = 0
        for spec in bcfg["sources"].values():
            for d in normalize_source_spec(spec, default_bucket):
                mix_total += d["target"]
        logger.info("%-25s  default=%s  target %s",
                    bname, default_bucket, fmt_tokens(mix_total))

        for src, spec in bcfg["sources"].items():
            for d in normalize_source_spec(spec, default_bucket):
                target = d["target"]
                bucket_name = d["bucket"]
                lo, hi = BUCKET_BOUNDS[bucket_name]
                rec = catalogue[src]["records"]
                if len(rec) == 0:
                    logger.info("  %-50s [%s]  0 / %s",
                                src, bucket_name, fmt_tokens(target))
                    continue
                mask = (rec[:, 2] >= lo) & (rec[:, 2] < hi)
                avail = int(rec[mask, 2].sum())
                cov = 100.0 * avail / target if target else 0
                logger.info("  %-50s [%-9s]  %s / %s  (%.0f%%, %d docs)",
                            src, bucket_name, fmt_tokens(avail),
                            fmt_tokens(target), cov, int(mask.sum()))
        logger.info("")

    if args.dry_run:
        logger.info("Dry run - done.")
        return

    # Phase 3a: Sampling with disjoint partitioning per (src,bucket)
    #
    # Group all draw requests by (src, bucket). For each group, do ONE
    # random permutation and slice it into disjoint chunks per request,
    # so two mixes pulling from the same pool never see the same doc.
    logger.info("Phase 3: sampling + writing …")
    t3 = time.perf_counter()
    master_rng = np.random.default_rng(args.seed)

    # Step 1: collect every (mix, src, bucket) draw request
    pool_requests = defaultdict(list)   # (src, bucket) -> [(mix_name, target), ...]
    for bname, bcfg in SAMPLING_CONFIG.items():
        for src, spec in bcfg["sources"].items():
            for d in normalize_source_spec(spec, bcfg["bucket"]):
                pool_requests[(src, d["bucket"])].append((bname, d["target"]))

    # Step 2: sample disjoint slices for every pool
    # selections[(mix_name, src, bucket)] = (sel_indices, filtered_records)
    selections = {}
    for (src, bucket_name), reqs in pool_requests.items():
        rec = catalogue[src]["records"]
        lo, hi = BUCKET_BOUNDS[bucket_name]

        if len(rec) == 0:
            for bname, _ in reqs:
                logger.warning("[%s/%s/%s] no records", bname, src, bucket_name)
            continue

        mask = (rec[:, 2] >= lo) & (rec[:, 2] < hi)
        filtered = rec[mask]
        if len(filtered) == 0:
            for bname, _ in reqs:
                logger.warning("[%s/%s/%s] empty bucket",
                               bname, src, bucket_name)
            continue

        tok = filtered[:, 2]
        available = int(tok.sum())
        total_target = sum(t for _, t in reqs)

        seed = int(master_rng.integers(0, 2**63))
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(filtered))
        cum = np.cumsum(tok[perm])

        if total_target > available:
            scale = available / total_target
            logger.warning(
                "[%s/%s] requested %s exceeds available %s - scaling all "
                "draws by %.2fx", src, bucket_name,
                fmt_tokens(total_target), fmt_tokens(available), scale)
        else:
            scale = 1.0

        pos = 0
        for bname, target in reqs:
            scaled = target * scale
            if pos >= len(filtered):
                sel = np.empty(0, dtype=np.int64)
            else:
                target_cum = (cum[pos - 1] if pos > 0 else 0) + scaled
                cut = min(int(np.searchsorted(cum, target_cum, side="left")) + 1,
                          len(filtered))
                cut = max(cut, pos)
                sel = np.sort(perm[pos:cut])
                pos = cut
            selections[(bname, src, bucket_name)] = (sel, filtered)
            sampled = int(filtered[sel, 2].sum()) if len(sel) else 0
            logger.info("[%s/%s/%s] sampled %d docs, ~%s tokens "
                        "(target %s)",
                        bname, src, bucket_name, len(sel),
                        fmt_tokens(sampled), fmt_tokens(target))

    # Step 3: combine draws per (mix, src) into source_tasks
    source_tasks = defaultdict(list)   # source_name → [config_spec, ...]
    for bname, bcfg in SAMPLING_CONFIG.items():
        for src, spec in bcfg["sources"].items():
            draws = normalize_source_spec(spec, bcfg["bucket"])
            combined = defaultdict(set)
            total_target = 0
            for d in draws:
                total_target += d["target"]
                key = (bname, src, d["bucket"])
                if key not in selections:
                    continue
                sel, filtered = selections[key]
                for i in sel:
                    si, did = int(filtered[i, 0]), int(filtered[i, 1])
                    combined[si].add(did)

            if not combined:
                continue
            docs_by_shard = {
                si: np.array(sorted(dids), dtype=np.int64)
                for si, dids in combined.items()
            }
            source_tasks[src].append({
                "config_name":   bname,
                "bucket_dir":    bcfg["bucket_dir"],
                "target_tokens": total_target,
                "docs_by_shard": docs_by_shard,
            })

    # Phase 3b: One worker per source 
    n_workers = len(source_tasks)
    logger.info("\nDispatching %d source workers …", n_workers)

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futs = {}
        for src, specs in source_tasks.items():
            sp = catalogue[src]["shard_prefixes"]
            futs[pool.submit(process_source, src, sp, specs, output_dir)] = src

        for f in as_completed(futs):
            src = futs[f]
            try:
                f.result()
            except Exception:
                logger.exception("FAILED: %s", src)

    logger.info("\nAll done in %.1fs", time.perf_counter() - t3)


if __name__ == "__main__":
    main()