# Masked Language Model Scoring

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

This package uses masked LMs like [BERT](https://arxiv.org/abs/1810.04805), [RoBERTa](https://arxiv.org/abs/1907.11692), and [XLM](https://papers.nips.cc/paper/8928-cross-lingual-language-model-pretraining.pdf) to [score sentences](#scoring) and [rescore n-best lists](#rescoring) via *pseudo-log-likelihood scores*, which are computed by masking individual words. We also support autoregressive LMs like [GPT-2](https://openai.com/blog/better-language-models/). Example uses include:
- [Speech Recognition](examples/asr-librispeech-espnet): Rescoring an ESPnet LAS model (LibriSpeech)
- [Machine Translation](examples/nmt-tedtalks-ace): Rescoring a Transformer NMT model (IWSLT'15 en-vi)
- [Linguistic Acceptability](examples/lingacc-blimp): Unsupervised ranking within linguistic minimal pairs (BLiMP)

**Paper:** Julian Salazar, Davis Liang, Toan Q. Nguyen, Katrin Kirchhoff. "[Masked Language Model Scoring](https://arxiv.org/abs/1910.14659)", ACL 2020.

## Installation

Python 3.6+ is required. Clone this repository and install:
```bash
pip install -e .
pip install torch mxnet-cu102mkl  # Replace w/ your CUDA version; mxnet-mkl if CPU only.
```
Some models are via [GluonNLP](https://github.com/dmlc/gluon-nlp) and others are via [🤗 Transformers](https://github.com/huggingface/transformers), so for now we require both [MXNet](https://mxnet.apache.org/) and [PyTorch](https://pytorch.org/).

## Quick start

To integrate scoring with your application:
```python
from mlm.scorers import MLMScorer
from mlm.models import get_pretrained
import mxnet as mx
ctxs = [mx.gpu()]
model, vocab, tokenizer = get_pretrained(ctxs, 'bert-base-en-cased')
scorer = MLMScorer(model, vocab, tokenizer, ctxs)
# [-12.410664200782776]
print(scorer.score_sentences(["Hello world!"]))
# [[None, -6.126736640930176, -5.501412391662598, -0.7825151681900024, None]]
print(scorer.score_sentences(["Hello world!"], per_token=True))
```

## Scoring

Run `mlm score --help` to see supported models, etc. See `examples/demo/format.json` for the file format. For inputs, "score" is optional. Outputs will add "score" fields containing PLL scores.

There are three score types, depending on the model:
- Pseudo-log-likelihood score (PLL): BERT, RoBERTa, multilingual BERT, XLM
- Maskless PLL score: same (add `--no-mask`)
- Log-probability score: GPT-2

We score hypotheses for 3 utterances of LibriSpeech `dev-other` on GPU 0 using BERT base (uncased):
```bash
mlm score \
    --mode hyp \
    --model bert-base-en-uncased \
    --max-utts 3 \
    --gpus 0 \
    examples/asr-librispeech-espnet/data/dev-other.am.json \
    > examples/demo/dev-other-3.lm.json
```

## Rescoring

One can rescore n-best lists via log-linear interpolation. Run `mlm rescore --help` to see all options. Input one is a file with original scores; input two are scores from `mlm score`.

We rescore acoustic scores (from `dev-other.am.json`) using BERT's scores (from previous section), under different LM weights:
```bash
for weight in 0 0.5 ; do
    echo "lambda=${weight}"; \
    mlm rescore \
        --model bert-base-en-uncased \
        --weight ${weight} \
        examples/asr-librispeech-espnet/data/dev-other.am.json \
        examples/demo/dev-other-3.lm.json \
        > examples/demo/dev-other-3.lambda-${weight}.json
done
```
The original WER is 12.2% while the rescored WER is 8.5%.

## Maskless finetuning

One can finetune masked LMs to give usable PLL scores without masking. See [LibriSpeech maskless finetuning](examples/asr-librispeech-espnet/README.md#maskless-finetuning).

## Development

Run `pip install -e .[dev]` to install extra testing packages. Then:

- To run unit tests and coverage, run `pytest --cov=src/mlm` in the root directory.
