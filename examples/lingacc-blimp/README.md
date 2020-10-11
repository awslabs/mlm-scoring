# Linguistic Acceptability > BLiMP

We use PLLs to perform unsupervised acceptability judgments on the [Benchmark of Linguistic Minimal Pairs (BLiMP)](https://arxiv.org/abs/1912.00582) (Warstadt et al., 2020). Here, BERT and RoBERTa consistently outperform GPT-2 and largely narrow the gap with human performance.

## Data

```sh
# Download data
git clone https://github.com/alexwarstadt/blimp
# Convert data from JSONL to sentences for scoring; creates data/
./convert_to_txt.py
# Convert data from JSONL to dataset for PPPL computation; creates data-concat/
./convert_to_dataset.py
```

## Ranking

First, we score each sentence:
```sh
for file in data/* ; do
    for model in gpt2-345m-en-cased roberta-large-en-cased ; do
        echo "Scoring pairs in ${file}..."
        mkdir -p output/${model}/
        mlm score \
            --mode ref \
            --model ${model} \
            --gpus 0 \
            --split-size 500 \
            ${file} \
            > output/${model}/$(basename ${file} .txt).lm.json
    done
done
```

Then, we compute accuracies (how often the good sentence was ranked over the bad):
```sh
echo "GPT-2 (345M)"
./accuracy.py output/gpt2-345m-en-cased
echo "RoBERTa (large)"
./accuracy.py output/roberta-large-en-cased
```
These give 82.6% and 86.5%, respectively. Human performance is 88.6%.

See [the paper](https://www.aclweb.org/anthology/2020.acl-main.240/) for complete results. After the paper, we found `distilbert-base-cased` gives 78.3% and `albert-xxlarge-v2` gives 84.4%; details in [Issue #2](https://github.com/awslabs/mlm-scoring/issues/2).

## Pseudo-perplexities

This gives token-level PPPLs of 59.2 on the acceptable sentences and 111.2 on the unacceptable ones:
```sh
for suffix in good.txt bad.txt ; do
    for model in bert-base-en-cased ; do
        echo "Scoring ${suffix}..."
        mlm score \
            --mode ref \
            --model ${model} \
            --gpus 0 \
            --split-size 1500 \
            data-concat/${suffix} \
            > output/${model}.${suffix}
    done
done
```
