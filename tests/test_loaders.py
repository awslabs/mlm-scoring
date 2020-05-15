import pytest

import json

from mlm.loaders import Corpus, Hypotheses, Predictions, ScoredCorpus


TEST_DICT = {
    'utt1': {
        'ref': "BERT",
        'hyp_1': {'score': -7, 'text': "ERNIE"},
        'hyp_2': {'score': -8.5, 'text': "ELMo"}
    },
    'utt0': {
        'ref': "   This is true.  ",
        'hyp_1': {'score': -5, 'text': "This is grue."},
        'hyp_2': {'score': -6.5, 'text': "This is moo?"}
    }
}

TEST_CORPUS = """    I am line one 
 I am line two 
"""


def _get_json_fp(tmp_path):
    tmp_file = tmp_path / "test.json"
    test_contents = json.dumps(TEST_DICT)
    tmp_file.write_text(test_contents)
    fp = tmp_file.open('rt')
    return fp


def _get_text_fp(tmp_path):
    tmp_file = tmp_path / "test.txt"
    tmp_file.write_text(TEST_CORPUS)
    fp = tmp_file.open('rt')
    return fp


def test_predictions_from_file(tmp_path):
    fp = _get_json_fp(tmp_path)
    preds = Predictions.from_file(fp, max_utts=1)
    # max_utts = 1
    assert len(preds) == 1
    # Should be alphabetical (shouldn't rely on occurrence in file)
    assert 'utt0' in preds
    hyp = preds['utt0']
    assert isinstance(hyp, Hypotheses)
    # hyp_1 has array index 0
    assert hyp.sents[0] == "This is grue."
    assert hyp.scores[1] == -6.5


def test_predictions_from_nmt(tmp_path):

    tmp_file = tmp_path / "test.nobpe"
    test_contents = """This is translation 1 . -5.5 -30
This is translation two ? -6.5 -20

I am a product of beam search . -6.5 -10
I am a product of greedy sampling :( -10.5 -40
"""
    tmp_file.write_text(test_contents)
    with tmp_file.open('rt') as fp:
        preds = Predictions.from_nmt(fp, max_utts=1)

    # max_utts = 1
    assert len(preds) == 1
    # Should be numerical (since no ID)
    assert 0 in preds
    hyp = preds[0]
    assert isinstance(hyp, Hypotheses)
    # hyp_1 has array index 0
    assert hyp.sents[0] == "This is translation 1 ."
    assert hyp.scores[1] == -6.5


def test_predictions_to_corpus(tmp_path):

    preds = Predictions.from_dict(TEST_DICT, max_utts=1)
    corpus = preds.to_corpus()
    assert isinstance(corpus, Corpus)
    assert len(corpus) == 2
    assert corpus['utt0--1'] == "This is grue."
    assert corpus['utt0--2'] == "This is moo?"
    corpus_text = list(corpus.values())
    assert corpus_text[0] == "This is grue."


def test_predictions_to_json(tmp_path):

    preds = Predictions.from_dict(TEST_DICT, max_utts=1)

    tmp_file = tmp_path / 'test.json'
    with tmp_file.open('wt') as fp:
        preds.to_json(fp)
    output = tmp_file.read_text()
    output_dict = json.loads(output)
    assert len(output_dict) == 1
    assert output_dict['utt0']['hyp_1']['text'] == TEST_DICT['utt0']['hyp_1']['text']
    assert output_dict['utt0']['hyp_2']['score'] == TEST_DICT['utt0']['hyp_2']['score']


def test_corpus_from_file(tmp_path):

    fp = _get_json_fp(tmp_path)
    refs = Corpus.from_file(fp, max_utts=1)

    # max_utts = 1
    assert len(refs) == 1
    # Should be alphabetical (shouldn't rely on occurrence in file)
    assert 'utt0' in refs
    sent = refs['utt0']
    # Should be stripped
    assert sent == "This is true."


def test_corpus_from_text(tmp_path):

    fp = _get_text_fp(tmp_path)
    refs = Corpus.from_text(fp, max_utts=1)

    # max_utts = 1
    assert len(refs) == 1
    # Should be numerical (since no ID)
    assert 0 in refs
    sent = refs[0]
    # Should be stripped
    assert sent == "I am line one"


def test_scored_corpus(tmp_path):

    preds = Predictions.from_dict(TEST_DICT)
    corpus = preds.to_corpus()
    scores = [1, 2, 3, 4]
    scored_corpus = ScoredCorpus.from_corpus_and_scores(corpus, scores)

    # to_predictions()
    scored_preds = scored_corpus.to_predictions()
    assert len(scored_preds) == 2
    assert scored_preds['utt0'].sents[1] == "This is moo?"
    assert scored_preds['utt1'].scores[1] == 4

    # to_file()
    tmp_file = tmp_path / 'scores.txt'
    with tmp_file.open('wt') as fp:
        scored_corpus.to_file(fp, scores_only=True)
    output = tmp_file.read_text()
    assert output == "1\n2\n3\n4\n"
    with tmp_file.open('wt') as fp:
        scored_corpus.to_file(fp, scores_only=False)
    output = tmp_file.read_text()
    assert output.startswith("This is grue. 1\n")
