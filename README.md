# LongContextQA

## Introduction

To improve retrieval efficiency, the first tasks that we are introducing into pre-training are:

1. [**Artificial needles**](https://arxiv.org/pdf/2406.19292): A large dictionary that teaches the model to retrieve information.
2. [**CWE**](https://arxiv.org/pdf/2512.13961) (Common Words Extraction): Word counting, which allows the model to understand how many times a specific token appears.
3. **Document ordering**: Shuffle the input sequence and assign paragraph numbers. The task consists of regrouping the paragraphs. The difficulty can be increased by splitting the document into more sections. I haven’t found a related paper, but this seems worth testing.

To improve packing efficiency, each task includes samples spanning a range of sequence lengths. For example, in the 64k context setting, samples range from 16k to 64k tokens. Within this range, shorter sequences (16k–32k) are made more difficult to balance the training signal:

**CWE**

* Hard: 10 QA pairs
* Easy: 5 QA pairs

**Document ordering**

* Hard: 8 shuffled sections
* Easy: 4 shuffled sections

## Long Context Pipeline

Please note that the final token count changed slightly in some scripts. However, the general pipeline can be described by the following diagram:

![Synthetic LC pipeline](./images/synthetic_LC.png)

We first need to separate the available data into three subsets:

* Long Context Data (17B).
* Long Context Data for easy samples (1.5B).
* Long Context Data for hard samples (1.5B).

The 3B-token subset is used as the seed corpus for the synthetic tasks: CWE (3B tokens) and Document Ordering (2B tokens, reused with masking and shuffling). An additional 1B tokens are incorporated to further enrich the training mixture.

First, execute the bash script `separation_script`, and then run the remaining scripts accordingly.

## Cooldown Subsample

The exact scripts used to split the cooldown datasets into four folders each containing ~40B tokens are available in the `split_cooldown` folder.
