import argparse
from contextlib import contextmanager
import logging
import math
import os
from pathlib import Path
import random
import sys
from typing import List, TextIO

import numpy as np
# MXNet-based
import gluonnlp as nlp
import mxnet as mx
# PyTorch-based
import torch
import transformers

from .loaders import Predictions, Corpus, ScoredCorpus
from .models import get_pretrained, SUPPORTED
from .models.bert import BERTRegression, BertForMaskedLMOptimized
from .scorers import LMScorer, LMBinner, MLMScorer, MLMScorerPT, MLMBinner, RegressionFinetuner, RegressionScorer


def _shared_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--gpus', type=str, default='-1',
                        help="Comma-delimited list of GPUs to use (-1 is CPU)")
    parser.add_argument('--max-utts', type=int,
                        help="maximum utterances to parse")
    parser.add_argument('--model', type=str,
                        help="Model to (re)score   -   HuggingFace models: 'bert-*', 'xlm-*'   -   GluonNLP models: {}".format(SUPPORTED))
    parser.add_argument('--weights', type=str, default=None,
                        help="Model weights to load")


@contextmanager
def _stdout_to_stderr():
    old_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        yield
    finally:
        sys.stdout = old_stdout


# Converts a list "0,2,7" to [mx.gpu(0), mx.gpu(2), mx.gpu(7)]
# (if equals -1, then run on cpu(0)
SEED = 0
def setup_ctxs(gpu_str: str) -> List[mx.Context]:

    random.seed(SEED)
    np.random.seed(SEED)
    mx.random.seed(SEED)

    ids = [int(id) for id in gpu_str.split(',')]
    if len(ids) == 1 and ids[0] < 0:
        ctxs = [mx.cpu(0)]
    else:
        for id in ids:
            mx.random.seed(SEED, mx.gpu(id))
        ctxs = [mx.gpu(id) for id in ids]

    # Following GluonNLP's scripts/language_model/large_word_language_model.py
    # https://mxnet.incubator.apache.org/faq/env_var.html
    os.environ['MXNET_GPU_MEM_POOL_TYPE'] = 'Round'
    os.environ['MXNET_CPU_PARALLEL_RAND_COPY'] = str(len(ctxs))
    os.environ['MXNET_CPU_WORKER_NTHREADS'] = str(len(ctxs))

    return ctxs


def main() -> None:
    """Defines arguments for all subcommands"""
    parser = argparse.ArgumentParser(description="Masked Language Model Scoring")
    subparsers = parser.add_subparsers(help="Run 'mlm {subcommand} -h' for details")

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')

    # score
    parser_score = subparsers.add_parser('score', help='Scores JSON or TXT files of sentences')
    _shared_args(parser_score)
    parser_score.add_argument('--mode', type=str, choices=['ref', 'hyp'],
                        help="Scoring references (.txt, .json 'refs') vs. hypotheses (.json 'hyp_*')")
    parser_score.add_argument('--temp', type=float, default=1.0,
                        help="softmax temperature")
    parser_score.add_argument('--split-size', type=int, default=500,
                        help="split size (per GPU)")
    parser_score.add_argument('--no-mask', action='store_true',
                        help="Instead of making masked copies, do not mask")
    parser_score.add_argument('--tgt', type=str, default='en',
                        help="Code to use for language embeddings, where appropriate")
    parser_score.add_argument('--eos', action='store_true',
                        help="append '.' (this can help mitigate train-test disparity)")
    parser_score.add_argument('--detok', action='store_true',
                        help="perform Moses English detokenization on hypotheses before scoring")

    capitalize_parser = parser_score.add_mutually_exclusive_group(required=False)
    capitalize_parser.add_argument('--capitalize', dest='capitalize', action='store_true')
    capitalize_parser.add_argument('--no-capitalize', dest='capitalize', action='store_false')
    parser_score.set_defaults(capitalize=None)
    parser_score.add_argument('--whole-word-mask', action='store_true',
                        help="mask whole words")
    parser_score.add_argument('--per-token', action='store_true',
                        help="output lists of per-token scores (slower)")
    parser_score.add_argument('infile', nargs='?', type=argparse.FileType('rt'),
                        help="File to score (.json = ESPNet JSON, otherwise newline-separated text). Loads whole file into memory!")
    parser_score.set_defaults(func=cmd_score)

    # bin (same arguments as score; when stable, make flag)
    parser_bin = subparsers.add_parser('bin', help='Computes bin statistics when scoring')
    _shared_args(parser_bin)
    parser_bin.add_argument('--mode', type=str, choices=['ref', 'hyp'],
                        help="Scoring references (.txt, .json 'refs') vs. hypotheses (.json 'hyp_*')")
    parser_bin.add_argument('--temp', type=float, default=1.0,
                        help="softmax temperature")
    parser_bin.add_argument('--split-size', type=int, default=1000,
                        help="split size (per GPU)")
    parser_bin.add_argument('--no-mask', action='store_true',
                        help="Instead of making masked copies, do not mask")
    parser_bin.add_argument('--eos', action='store_true',
                        help="append '.' (this can help mitigate train-test disparity)")
    capitalize_parser = parser_bin.add_mutually_exclusive_group(required=False)
    capitalize_parser.add_argument('--capitalize', dest='capitalize', action='store_true')
    capitalize_parser.add_argument('--no-capitalize', dest='capitalize', action='store_false')
    parser_bin.set_defaults(capitalize=None)
    parser_bin.add_argument('--whole-word-mask', action='store_true',
                        help="mask whole words")
    parser_bin.add_argument('infile', nargs='?', type=argparse.FileType('rt'),
                        help="File to score (.json = ESPNet JSON, otherwise newline-separated text). Loads whole file into memory!")
    parser_bin.add_argument('counts_file', nargs='?', type=str,
                        help="where to dump the counts per bin")
    parser_bin.add_argument('sums_file', nargs='?', type=str,
                    help="where to dump the sums per bin")
    parser_bin.set_defaults(func=cmd_bin)

    # rescore
    parser_rescore = subparsers.add_parser('rescore', help='Rescores two files together')
    _shared_args(parser_rescore)
    parser_rescore.add_argument('--weight', type=str, default='0.3',
                    help="AM score is (1-sum(weight)), LM scores are weights delimited by commas")
    parser_rescore.add_argument('--ref-file', type=argparse.FileType('rt'),
                    help="Specify an alternative reference file to FILE_AM")
    parser_rescore.add_argument('--ln', type=float, default=None,
                        help="apply GNMT normalization with this scale to each >>LM<< score")
    parser_rescore.add_argument('--ln-type', type=str, choices=['gnmt', 'length'], default='gnmt',
                        help="type of normalization to apply")
    parser_rescore.add_argument('file_am', type=argparse.FileType('rt'),
                        help="File with AM scores (.json = JSON)")
    parser_rescore.add_argument('file_lm', type=str,
                        help="File(s) with LM scores (.json = JSON), delimited by commas")
    parser_rescore.set_defaults(func=cmd_rescore)

    # finetune
    parser_finetune = subparsers.add_parser('finetune', help='Finetune to scoring without masks')
    _shared_args(parser_finetune)
    parser_finetune.add_argument('--corpus-dir', type=str, required=True,
                        help="Directory of part.*")
    parser_finetune.add_argument('--score-dir', type=str, required=True,
                        help="Directory of part.*.ref.scores")
    parser_finetune.add_argument('--output-dir', type=str, required=True,
                        help="Directory to output .param files")
    parser_finetune.add_argument('--freeze', type=int, default=0,
                        help="Number of initial layers to freeze")

    # TODO: deduplicate
    parser_finetune.add_argument('--eos', action='store_true',
                        help="append '.' (this can help mitigate train-test disparity)")
    capitalize_parser = parser_finetune.add_mutually_exclusive_group(required=False)
    capitalize_parser.add_argument('--capitalize', dest='capitalize', action='store_true')
    capitalize_parser.add_argument('--no-capitalize', dest='capitalize', action='store_false')
    parser_finetune.set_defaults(capitalize=None)
    parser_finetune.add_argument('--whole-word-mask', action='store_true',
                        help="mask whole words")
    parser_finetune.add_argument('--split-size', type=int, default=1000,
                    help="split size (per GPU)")


    parser_finetune.set_defaults(func=cmd_finetune)

    args = parser.parse_args()
    args.func(args)


def cmd_score(args: argparse.Namespace) -> None:
    """mlm score command
    """

    # Get model
    ctxs = setup_ctxs(args.gpus)
    weights_file = Path(args.weights) if isinstance(args.weights, str) else None
    # Redirect console output from GluonNLP downloading models
    with _stdout_to_stderr():
        model, vocab, tokenizer = get_pretrained(ctxs, args.model, weights_file, regression=args.no_mask)

    # Set scorer
    if MLMScorer._check_support(model):
        # GluonNLP
        scorer = MLMScorer(model, vocab, tokenizer, eos=args.eos, wwm=args.whole_word_mask, capitalize=args.capitalize, ctxs=ctxs)
    elif MLMScorerPT._check_support(model):
        # Transformers
        scorer = MLMScorerPT(model, vocab, tokenizer, eos=args.eos, wwm=args.whole_word_mask, capitalize=args.capitalize, ctxs=ctxs, lang=args.tgt)
    elif RegressionScorer._check_support(model):
        scorer = RegressionScorer(model, vocab, tokenizer, eos=args.eos, wwm=args.whole_word_mask, capitalize=args.capitalize, ctxs=ctxs)
    elif LMScorer._check_support(model):
        assert not args.whole_word_mask
        assert not args.no_mask
        scorer = LMScorer(model, vocab, tokenizer, eos=args.eos, capitalize=args.capitalize, ctxs=ctxs)
    else:
        raise ValueError(f"Model '{model.__class__.__name__}' not supported by any scorer.")

    # What data do we use?
    if args.mode == 'hyp':

        preds = Predictions.from_file(args.infile, max_utts=args.max_utts)
        # We 'deserialize' the predictions into a corpus, for better batching
        corpus = preds.to_corpus()

        logging.warn("# of input sequences: {}".format(len(preds)))
        logging.warn("# of hypotheses: {}".format(len(corpus)))

    elif args.mode == 'ref':

        corpus = Corpus.from_file(args.infile, max_utts=args.max_utts)
        logging.warn("# sentences: {}".format(len(corpus)))

    # === START SHARED COMPUTATION ===

    # A scorer takes a corpus and produces a list of scores in order of the corpus
    if args.detok:
        corpus_for_scoring = corpus.detok_copy()
    else:
        corpus_for_scoring = corpus
    scores, true_tok_lens = scorer.score(corpus_for_scoring, ratio=1, split_size=args.split_size, per_token=args.per_token)
    scored_corpus = ScoredCorpus.from_corpus_and_scores(corpus, scores)

    num_words_list, max_sent_len = corpus.get_num_words()
    if args.eos:
        logging.warn("Adding EOSes '.' to (P)PPL computation")
        num_words_list = [x+1 for x in num_words_list]
    num_words_total = sum(num_words_list)
    if args.eos:
        logging.warn("# words (excluding EOS '.'): {}".format(num_words_total))
    else:
        logging.warn("# words: {}".format(num_words_total))
    logging.warn("longest sentence: {}".format(max_sent_len))

    num_toks_total = sum(true_tok_lens)
    if args.eos:
        logging.warn("# tokens (including EOS '.'): {}".format(num_toks_total))
    else:
        logging.warn("# tokens: {}".format(num_toks_total))


    if not args.per_token:
        plls = np.array(scores)
        pppl_tok_micro = np.exp(- plls.sum() / num_toks_total).item()
        logging.warn("Token-level (P)PPL: {}".format(pppl_tok_micro))

        # pppl_tok_macro = np.exp(- (plls / np.array(true_tok_lens)).mean())
        # logging.warn("Token-level (P)PPL, macro: {}".format(pppl_tok_macro))

        pppl_word_micro = math.exp((num_toks_total / num_words_total) * math.log(pppl_tok_micro))
        logging.warn("Word-normalized (P)PPL: {}".format(pppl_word_micro))

        # pppl_word_macro = np.exp(- (plls / np.array(num_words_list)).mean())
        # logging.warn("Word-normalized (P)PPL, macro: {}".format(pppl_word_macro))

    # === END SHARED COMPUTATION ===

    # How do we output?
    if args.mode == 'hyp':

        preds = scored_corpus.to_predictions()
        preds.to_json(sys.stdout)

    # otherwise we just print a list of log likelihoods
    elif args.mode == 'ref':

        scored_corpus.to_file(sys.stdout, scores_only=True)



def cmd_bin(args: argparse.Namespace) -> None:
    """mlm score command
    """

    # Get model
    ctxs = setup_ctxs(args.gpus)
    weights_file = Path(args.weights) if isinstance(args.weights, str) else None
    model, vocab, tokenizer = get_pretrained(ctxs, args.model, weights_file, regression=args.no_mask)

    # Define output files
    counts_file = Path(args.counts_file)
    sums_file = Path(args.sums_file)

    # Set binner
    if isinstance(model, nlp.model.BERTModel):
        assert not args.whole_word_mask
        assert not args.no_mask
        binner = MLMBinner(model, vocab, tokenizer, eos=args.eos, capitalize=args.capitalize, ctxs=ctxs)
    elif isinstance(model, BERTRegression):
        raise ValueError("Not supported")
    else:
        assert not args.whole_word_mask
        assert not args.no_mask
        binner = LMBinner(model, vocab, tokenizer, eos=args.eos, capitalize=args.capitalize, ctxs=ctxs)

    # What data do we use?
    if args.mode == 'hyp':

        raise ValueError("Not supported")

    elif args.mode == 'ref':

        corpus = Corpus.from_file(args.infile, max_utts=args.max_utts)
        logging.warn("# sentences: {}".format(len(corpus)))

    # === START SHARED COMPUTATION ===

    # A binner takes a corpus and produces a list of bin counts and scores
    bin_counts, bin_sums = binner.bin(corpus, ratio=1, split_size=args.split_size)
    logging.warning("Saving bin counts to '{}'".format(counts_file))
    np.save(counts_file, bin_counts)
    logging.warning("Saving bin sums to '{}'".format(sums_file))
    np.save(sums_file, bin_sums)


def cmd_rescore(args: argparse.Namespace) -> None:
    """rescore command

    You have two files with the following schema:
    {
        "<UTT_ID>": {
            "ref": {
                "score": 0.111
            },
            "hyp_1": {
                "score": 3.15
            },
            ...
        },
        ...
    }

    """

    model_list = args.model.split(',')
    pretrained_tup_list = [get_pretrained([mx.cpu(0)], model) for model in model_list]

    file_lm_list = args.file_lm.split(',')
    weight_list = [float(x) for x in args.weight.split(',')]

    assert len(pretrained_tup_list) == len(weight_list)

    preds_am = Predictions.from_file(args.file_am, max_utts=args.max_utts)
    preds_lm_list = [Predictions.from_file(Path(file_lm).open('r'), max_utts=args.max_utts, vocab=vocab, tokenizer=tokenizer) for file_lm, (_, vocab, tokenizer) in zip(file_lm_list, pretrained_tup_list)]

    # Preserves input order, but slower
    shared_keys = list(preds_am.keys())
    for preds_lm in preds_lm_list:
        # isdigit() suggests we're in automatic ID mode; cast to int
        preds_lm_keys = set(((int(key) if key.isdigit() else key) for key in preds_lm.keys()))
        shared_keys = [key for key in shared_keys if key in preds_lm_keys]

    logging.warn("{} shared keys found, rescoring these...".format(len(shared_keys)))

    preds_new = Predictions()
    for utt_id in shared_keys:
        hyps_am = preds_am[utt_id]
        hyps_lm_list = [preds_lm[str(utt_id)] for preds_lm in preds_lm_list]
        new_hyps = hyps_am.rescore(hyps_lm_list, scales=weight_list, ln=args.ln, ln_type=args.ln_type)
        preds_new[utt_id] = new_hyps

    preds_new.to_json(sys.stdout)

    # Compute WER after rescoring using the first file, if possible
    ref_file = args.ref_file
    if ref_file is None:
        ref_file = args.file_am
        ref_file.seek(0)

    if Path(ref_file.name).suffix == '.json':
        my_wer = _wer(ref_file, preds_new)
        logging.warn("WER: {}%".format(my_wer*100))
    else:
        my_bleu = _mbleu(ref_file, preds_new)


def _wer(file_ref: TextIO, preds_hyps: Predictions) -> float:
    data = Corpus.from_file(file_ref)

    num_utts = len(preds_hyps)
    logging.warn("{} predictions to compute total WER on...".format(num_utts))
    assert num_utts > 0

    my_wer = data.wer_score(preds_hyps)

    return my_wer


def _mbleu(file_ref: TextIO, preds_hyps: Predictions) -> float:
    data = Corpus.from_file(file_ref)

    num_utts = len(preds_hyps)
    logging.warn("{} predictions to compute multi-BLEU on...".format(num_utts))
    assert num_utts > 0

    my_bleu = data.mbleu_score(preds_hyps)

    return my_bleu


def cmd_finetune(args: argparse.Namespace) -> None:

    # Get model
    ctxs = setup_ctxs(args.gpus)
    weights_file = Path(args.weights) if isinstance(args.weights, str) else None
    model, vocab, tokenizer = get_pretrained(ctxs, args.model, weights_file, regression=True, freeze=args.freeze)
    # model.hybridize(static_alloc=True)

    for corpus_file in sorted(Path(args.corpus_dir).glob('part.*')):
        score_file = Path(args.score_dir) / (corpus_file.name + '.ref.scores')
        if not score_file.is_file():
            raise ValueError("Corpus file '{}' found but score file '{}' not found".format(corpus_file, score_file))
        logging.warn("Loading corpus from '{}' and scores from '{}'".format(corpus_file, score_file))
        scored_corpus = ScoredCorpus.from_files(corpus_file, score_file, max_utts=args.max_utts)
        # TODO: More than one file
        break

    logging.warn("# of hypotheses: {}".format(len(scored_corpus)))

    ### FINETUNING LOOP

    tuner = RegressionFinetuner(model, vocab, tokenizer, eos=args.eos, wwm=args.whole_word_mask, capitalize=args.capitalize, ctxs=ctxs)
    tuner.tune(scored_corpus, ratio=1, split_size=args.split_size, output_dir=Path(args.output_dir))
