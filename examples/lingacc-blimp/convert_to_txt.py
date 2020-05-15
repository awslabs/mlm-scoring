#!/usr/bin/env python3

import argparse
import logging
import json
from pathlib import Path

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Convert BLiMP files into text")
    parser.add_argument('--input-dir', type=str, default='blimp', help="BLiMP directory")
    parser.add_argument('--output-dir', type=str, default='data', help="Output directory for .txt files")
    args = parser.parse_args()

    input_dir = Path(args.input_dir) / 'data'
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for jsonl in input_dir.glob('*.jsonl'):
        output_file = output_dir / jsonl.with_suffix('.txt').name
        logging.warn("{} --> {}".format(jsonl, output_file))
        lines = [json.loads(line) for line in jsonl.read_text().split('\n') if len(line.strip())]
        with output_file.open('wt') as f_out:
            for line in lines:
                # Adopt the convention that odd lines = good, even lines = bad
                f_out.write(line['sentence_good'] + '\n')
                f_out.write(line['sentence_bad'] + '\n')
