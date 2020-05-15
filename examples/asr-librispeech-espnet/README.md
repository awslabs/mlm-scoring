# ASR > LibriSpeech > ESPnet

We include the 100-best LibriSpeech decoding outputs from "[Effective Sentence Scoring Method using Bidirectional Language Model for Speech Recognition](https://arxiv.org/abs/1905.06655)" (Shin et al., 2019) with the authors' permission; please cite their work if reusing their lists. The outputs come from a 5-layer encoder, 1-layer decoder BLSTMP model implemented in ESPnet. The files are `data/*.am.json`.

The split sizes are per 16GB Tesla V100 GPUs; in our case (`p3.8xlarge`) there are four. Scale appropriately for your per-GPU memory.

**TODO: Model artifacts.**

## Scoring

```bash
# Stock BERT/RoBERTa base
for set in dev-clean dev-other test-clean test-other ; do
    for model in bert-base-en-uncased bert-base-en-cased roberta-base-en-cased ; do
        mkdir -p output/${model}/
        echo ${set} ${model}
        mlm score \
            --mode hyp \
            --model ${model} \
            --gpus 0,1,2,3 \
            --split-size 2000 \
            --eos \
            data/${set}.am.json \
            > output/${model}/${set}.lm.json
    done
done
# Trained BERT base
for set in dev-clean dev-other test-clean test-other ; do
    for model in bert-base-en-uncased ; do
        mkdir -p output/${model}-380k/
        echo ${set} ${model}-380k
        mlm score \
            --mode hyp \
            --model ${model} \
            --gpus 0,1,2,3 \
            --split-size 2000 \
            --weights params/bert-base-en-uncased-380k.params \
            --eos \
            data/${set}.am.json \
            > output/${model}-380k/${set}.lm.json
    done
done
```

## Reranking

```bash
# Stock BERT/RoBERTa on development set
for set in dev-clean ; do
    for model in bert-base-en-uncased bert-base-en-cased roberta-base-en-cased ; do
        for weight in $(seq 0 0.05 1.0) ; do
            echo ${set} ${model} ${weight}; \
            mlm rescore \
                --model ${model} \
                --weight ${weight} \
                data/${set}.am.json \
                output/${model}/${set}.lm.json \
                > output/${model}/${set}.lambda-${weight}.json
        done
    done
done
# Once you have the best hyperparameter, evaluate test
for set in test-clean ; do
    for tup in bert-base-en-uncased,,0.35 bert-base-en-cased,,0.35 ; do
        IFS="," read model suffix weight <<< "${tup}"
        echo ${set} ${model}${suffix} ${weight}
        mlm rescore \
            --model ${model} \
            --weight ${weight} \
            data/${set}.am.json \
            output/${model}${suffix}/${set}.lm.json \
            > output/${model}${suffix}/${set}.lambda-${weight}.json
        done
    done
done
```

## Maskless finetuning

**Note:** Paper results are from a domain-adapted BERT.

We first download the normalized text corpus:
```bash
scripts/librispeech-download-text.sh data-distill/
```
We then score the corpus with a masked LM. We print 8 commands, one per GPU:
```bash
model=bert-base-en-uncased
split_size=12
for tup in 0,00,09 1,10,19 2,20,29 3,30,39 4,40,49 5,50,59 6,60,69 7,70,79 ; do
    IFS="," read gpu start end <<< ${tup}
    echo "scripts/librispeech-score.sh data-distill/ output-distill/${model} ${start} ${end} ${gpu} ${split_size} ${model}"
done
```
Modify GPU splits as desired, and e.g., run on different `screen`s.

For now, one must concatenate the used parts and scores into a single file:
```bash
model=bert-base-en-uncased
cat data-distill/part.* > output-distill/part.all
cat output-distill/${model}/part.*.ref.scores > output-distill/${model}/part.all.ref.scores
```

We then finetune BERT towards these sentence scores:
```bash
# `--corpus-dir output-distill` corresponds to reading from `output-distill/part.all`
model=bert-base-en-uncased
mkdir -p output-distill/${model}/params-1e-5_8gpu_384/ 
mlm finetune \
    --model ${model} \
    --gpus 0,1,2,3,4,5,6,7 \
    --eos \
    --corpus-dir output-distill \
    --score-dir output-distill/${model} \
    --output-dir output-distill/${model}/params-1e-5_8gpu_384/ \
    --split-size 30
```
Parameters will be saved to `output-distill/${model}/params-1e-5_8gpu_384/`.

We then score using these weights. Note the flags `--weights` and `--no-mask`. This runs much faster than masked scoring:
```bash
model=bert-base-en-uncased
for set in dev-clean ; do
    echo ${set} ${model}
    mlm score \
        --mode hyp \
        --model ${model} \
        --gpus 0 \
        --weights output-distill/${model}/params-1e-5_8gpu_384/epoch-10.params \
        --eos \
        --no-mask \
        --split-size 500 \
        data/${set}.am.json \
        > output-distill/${model}/params-1e-5_8gpu_384/${set}.lm.json
done
```

Finally, rerank:
```bash
for set in dev-clean ; do
    for weight in $(seq 0 0.05 1.0) ; do
        echo ${set} ${model} ${weight} 
        mlm rescore \
            --model ${model} \
            --weight ${weight} \
            data/${set}.am.json \
            output-distill/${model}/params-1e-5_8gpu_384/${set}.lm.json \
            > output-distill/${model}/params-1e-5_8gpu_384/${set}.lambda-${weight}.json
    done
done
```

## Binning

**TODO** To compute cross-entropy statistics:
```bash
mlm bin
```
