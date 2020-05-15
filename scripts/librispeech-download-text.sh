#!/usr/bin/env bash

set -e
set -x

target_dir=$1

mkdir -p ${1}
wget -O ${1}/librispeech-lm-norm.txt.gz http://www.openslr.org/resources/11/librispeech-lm-norm.txt.gz
gunzip ${1}/librispeech-lm-norm.txt.gz
# To avoid tripping up cased models
tr '[:upper:]' '[:lower:]' < ${1}/librispeech-lm-norm.txt > ${1}/librispeech-lm-norm.lower.txt
# Split to a number that's divisible by 10, and 4/8/16 GPUs ;)
split --numeric-suffixes --suffix-length 2 --number l/80 ${1}/librispeech-lm-norm.lower.txt ${1}/part.
# Clean up
rm ${1}/librispeech-lm-norm.txt ${1}/librispeech-lm-norm.lower.txt
echo "There should be 80 parts in ${1}; I found $(ls -1q ${1}/part.* | wc -l)."
