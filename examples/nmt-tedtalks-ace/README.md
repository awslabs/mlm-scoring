# NMT > TED Talks / IWSLT'15 > Ace

We use LPLs to rerank n-best lists produced from the state-of-the-art Transformer low-resource baselines also presented in [Transformers without Tears: Improving the Normalization of Self-Attention](https://arxiv.org/abs/1910.05895) (Nguyen and Salazar, 2019).

You can use the recipes in the [Ace codebase](https://github.com/tnq177/transformers_without_tears), which implements these baselines and produces n-best lists in the expected format.

We include our trained model's 100-best lists on IWSLT'15 English-Vietnamese at `data/en2vi/nbest/*`. The reference translation pairs can be found at https://github.com/tnq177/transformers_without_tears/blob/master/data/en2vi/test.vi.

## Scoring

```sh
for set in dev ; do
    for tup in "en,vi" ; do
        IFS="," read src tgt <<< ${tup}
        pair="${src}2${tgt}"
        PREFIX_DIR=data/${pair}/nbest
        for model in bert-base-multi-uncased ; do
        	mkdir -p output/${model}
            echo ${set} ${pair} ${model}
            mlm score \
                --mode hyp \
                --model ${model} \
                --gpus 0 \
                --split-size 500 \
                ${PREFIX_DIR}/${set}.${src}.bpe.beam_trans.nobpe \
                > output/${model}/${set}-${pair}.lm.json
        done
    done
done
```

## Rescoring

```sh
for set in dev ; do
    for tup in \
        "en,vi,bert-base-multi-uncased,0.15" ; do
        IFS="," read src tgt model weight <<< "${tup}"
        pair="${src}2${tgt}"
        PREFIX_DIR=data/${pair}/nbest
        echo ${set} ${tup} 
        mlm rescore \
            --model ${model} \
            --weight ${weight} \
            --ref-file ${PREFIX_DIR}/${set}.${tgt} \
            --ln 1.0 \
            ${PREFIX_DIR}/${set}.${src}.bpe.beam_trans.nobpe \
            output/${model}/${set}-${pair}.lm.json \
            > output/${model}/${set}-${pair}.lambda-${weight}.json
        done
    done
done
```

This will also output the BLEU score after rescoring.
