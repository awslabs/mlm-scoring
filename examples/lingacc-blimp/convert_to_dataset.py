#!/usr/bin/env python3

import argparse
import logging
import json
from pathlib import Path

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Convert BLiMP files into text")
    parser.add_argument('--input-dir', type=str, default='blimp', help="BLiMP directory")
    parser.add_argument('--output-dir', type=str, default='data-concat', help="Output directory for .txt files")
    args = parser.parse_args()

    input_dir = Path(args.input_dir) / 'data'
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    good_file = output_dir / 'good.txt'
    bad_file = output_dir / 'bad.txt'
    with good_file.open('wt') as f_good, bad_file.open('wt') as f_bad:
        for jsonl in input_dir.glob('*.jsonl'):
            logging.warn("{}".format(jsonl))
            lines = [json.loads(line) for line in jsonl.read_text().split('\n') if len(line.strip())]
            for line in lines:
                f_good.write(line['sentence_good'] + '\n')
                f_bad.write(line['sentence_bad'] + '\n')
