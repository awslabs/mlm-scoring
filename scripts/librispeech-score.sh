#!/usr/bin/env bash

# set -e
set -x

source_dir=$1
target_dir=$2
start=$3
end=$4
gpus=$5
split_size=$6
model=$7

if [ "$8" != "" ]; then
    model_weights_arg="--weights $8"
else
    model_weights_arg=""
fi

### TODO: Scale better so that split sizes are not absurdly low

for x in `seq -w ${start} ${end}`
do
    mkdir -p ${target_dir}
    mlm score ${model_weights_arg} \
        --mode ref \
        --model ${model} \
        --gpus ${gpus} \
        --split-size ${split_size} \
        ${source_dir}/part.${x} \
        > ${target_dir}/part.${x}.ref.scores \
        2> >(tee ${target_dir}/part.${x}.ref.log >&2)
done
