import pytest

import mxnet as mx
from mxnet.gluon.data import Dataset

from mlm.loaders import Corpus
from mlm.models import get_pretrained
from mlm.scorers import MLMScorer, MLMScorerPT



def _get_scorer_and_corpus():
    ctxs = [mx.cpu()]
    model, vocab, tokenizer = get_pretrained(ctxs, 'bert-base-en-uncased')
    scorer_mx = MLMScorer(model, vocab, tokenizer, ctxs, eos=True, wwm=False)
    model, vocab, tokenizer = get_pretrained(ctxs, 'bert-base-uncased')
    scorer_pt = MLMScorerPT(model, vocab, tokenizer, ctxs, eos=True, wwm=False)
    corpus = Corpus.from_dict({'utt': {'ref': "I am Sam"}})
    return scorer_mx, scorer_pt, corpus


def test_mlmscorer_corpus_to_dataset():
    scorer_mx, scorer_pt, corpus = _get_scorer_and_corpus()
    dataset = scorer_mx.corpus_to_dataset(corpus)
    assert isinstance(dataset, Dataset)
    # Our three tokens, plus the EOS
    assert len(dataset) == 4


def test_mlmscorer_score():
    scorer_mx, scorer_pt, corpus = _get_scorer_and_corpus()
    scores, _ = scorer_mx.score(corpus)
    assert len(scores) == 1
    assert pytest.approx(scores[0], abs=0.0001) == -13.3065947
    scores, _ = scorer_pt.score(corpus)
    assert len(scores) == 1
    assert pytest.approx(scores[0], abs=0.0001) == -13.3065947
