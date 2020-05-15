# Linguistic Acceptability > BLiMP

We use LPLs to perform unsupervised acceptability judgements on the [Benchmark of Linguistic Minimal Pairs (BLiMP)](https://arxiv.org/abs/1912.00582) (Warstadt et al., 2020).

## Data

```sh
# Download data
git clone https://github.com/alexwarstadt/blimp
# Convert data from JSONL to sentences for scoring; creates data/
python3 convert_to_txt.py
# Convert data from JSONL to dataset for PPPL computation; creates data-concat/
python3 convert_to_dataset.py
```

## Scoring

**TODO:** Commands