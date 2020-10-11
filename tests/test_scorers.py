import pytest

import mxnet as mx
from mxnet.gluon.data import Dataset

from mlm.loaders import Corpus
from mlm.models import get_pretrained
from mlm.scorers import LMScorer, MLMScorer, MLMScorerPT


# The ASR case, where we append . as an EOS

def _get_scorer_and_corpus_eos():
    ctxs = [mx.cpu()]
    model, vocab, tokenizer = get_pretrained(ctxs, 'bert-base-en-uncased')
    scorer_mx = MLMScorer(model, vocab, tokenizer, ctxs, eos=True, wwm=False)
    model, vocab, tokenizer = get_pretrained(ctxs, 'bert-base-uncased')
    scorer_pt = MLMScorerPT(model, vocab, tokenizer, ctxs, eos=True, wwm=False)
    corpus = Corpus.from_dict({'utt': {'ref': "I am Sam"}})
    return scorer_mx, scorer_pt, corpus


def test_mlmscorer_corpus_to_dataset():
    scorer_mx, scorer_pt, corpus = _get_scorer_and_corpus_eos()
    dataset = scorer_mx.corpus_to_dataset(corpus)
    assert isinstance(dataset, Dataset)
    # Our three tokens, plus the EOS
    assert len(dataset) == 4


def test_mlmscorer_score_eos():
    scorer_mx, scorer_pt, corpus = _get_scorer_and_corpus_eos()
    scores, _ = scorer_mx.score(corpus)
    assert len(scores) == 1
    assert pytest.approx(scores[0], abs=0.0001) == -13.3065947
    scores, _ = scorer_pt.score(corpus)
    assert len(scores) == 1
    assert pytest.approx(scores[0], abs=0.0001) == -13.3065947


# The general case

def test_mlmscorer_score_sentences():

    TEST_CASES = (
        # README examples
        ('bert-base-en-cased', MLMScorer, (None, -6.126666069030762, -5.50140380859375, -0.7823182344436646, None)),
        ('bert-base-cased', MLMScorerPT, (None, -6.126738548278809, -5.501765727996826, -0.782496988773346, None)),
        ('gpt2-117m-en-cased', LMScorer, (-8.293947219848633, -6.387561798095703, -1.3138668537139893)),
        # etc.
        ('albert-base-v2', MLMScorerPT, (None, -16.480087280273438, -12.897505760192871, -4.277405738830566, None)),
        ('distilbert-base-cased', MLMScorerPT, (None, -5.1874895095825195, -6.390861511230469, -3.8225560188293457, None)),
    )

    for name, scorer_cls, expected_scores in TEST_CASES:
        model, vocab, tokenizer = get_pretrained([mx.cpu()], name)
        scorer = scorer_cls(model, vocab, tokenizer, [mx.cpu()])
        scores = scorer.score_sentences(["Hello world!"], per_token=True)[0]
        expected_total = 0
        for score, expected_score in zip(scores, expected_scores):
            if score is None and expected_score is None:
                continue
            assert pytest.approx(score, abs=0.0001) == expected_score
            expected_total += expected_score
        score_total = scorer.score_sentences(["Hello world!"], per_token=False)[0]
        assert pytest.approx(score_total, abs=0.0001) == expected_total
