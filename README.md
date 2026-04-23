# LongContextQA

To improve the efficiency of retrieval, the firsts tasks that we are inserting into pre-training are:

1. [**Artificial needles**](https://arxiv.org/pdf/2406.19292): Large dictionary that teaches the model to retrieve information.
2. [**CWE](https://arxiv.org/pdf/2512.13961)** (Common Words Extraction): Word counting, this allows the model to understand how many times a specific token appears.
3. **Document ordering**: Shuffle the input sequence and assign paragraph numbers, the task will consist on reaggrupating the paragraphs. You can increase the difficulty by splitting more the document. I haven’t found a related paper, but worth testing.

TODOs:
- [MRCR data](https://arxiv.org/pdf/2409.12640v2).
- Need for full document attention: Sysnthetic non-verifiable data?

## Warnings

- This repository is under development, the scripts also use Megatron-LM indexed dataset outputs given that I am mainly interested in pre-training for now, adapting the code to post training would require some time.