import pytest

# MXNet-based
import gluonnlp as nlp
import mxnet as mx
# PyTorch-based
import torch
import transformers

from mlm.models import get_pretrained
from mlm.models.bert import BertForMaskedLMOptimized


def test_get_pretrained():

    # MXNet: bert-base-en-uncased

    model, vocab, tokenizer = get_pretrained([mx.cpu()], 'bert-base-en-uncased')
    # Check the model
    assert isinstance(model, nlp.model.BERTModel)
    assert len(model.encoder.transformer_cells) == 12
    unk_idx = vocab.token_to_idx[vocab.unknown_token]
    assert pytest.approx(model.word_embed[0].params['bertmodel0_word_embed_embedding0_weight']._data[0][unk_idx,0].asscalar()) == -0.0424806065
    # Check the vocab
    assert vocab.token_to_idx['test'] != unk_idx
    assert vocab.token_to_idx['Test'] == unk_idx
    # Check the tokenizer
    assert tuple(tokenizer("The man jumped up, put his basket on Philammon's head")) == ('the', 'man', 'jumped', 'up', ',', 'put', 'his', 'basket', 'on', 'phil', '##am', '##mon', "'", 's', 'head')

    # PyTorch: bert-base-en-uncased

    model, _, tokenizer = get_pretrained([mx.cpu()], 'bert-base-uncased')
    # Check the model
    assert isinstance(model, BertForMaskedLMOptimized)
    assert len(model.bert.encoder.layer) == 12
    unk_idx = tokenizer.unk_token_id
    assert pytest.approx(model.bert.embeddings.word_embeddings.parameters().__next__()[unk_idx,0].detach().numpy().item()) == -0.0424806065
    # Check the vocab
    assert tokenizer.convert_tokens_to_ids('test') != unk_idx
    assert tokenizer.convert_tokens_to_ids('Test') == unk_idx
    # Check the tokenizer
    assert tuple(tokenizer.tokenize("The man jumped up, put his basket on Philammon's head")) == ('the', 'man', 'jumped', 'up', ',', 'put', 'his', 'basket', 'on', 'phil', '##am', '##mon', "'", 's', 'head')

    # MXNet: bert-base-en-uncased-owt

    model, vocab_new, tokenizer = get_pretrained([mx.cpu()], 'bert-base-en-uncased-owt')
    # Check the model
    assert pytest.approx(model.word_embed[0].params['bertmodel1_word_embed_embedding0_weight']._data[0][0,0].asscalar()) == -0.0361938476
    # Check the vocab
    assert len(vocab_new) == len(vocab)
    # Check the tokenizer
    assert tuple(tokenizer("The man jumped up, put his basket on Philammon's head")) == ('the', 'man', 'jumped', 'up', ',', 'put', 'his', 'basket', 'on', 'phil', '##am', '##mon', "'", 's', 'head')

    # MXNet: bert-large-en-cased

    model, vocab, tokenizer = get_pretrained([mx.cpu()], 'bert-large-en-cased')
    # Check the model
    assert isinstance(model, nlp.model.BERTModel)
    assert len(model.encoder.transformer_cells) == 24
    assert pytest.approx(model.word_embed[0].params['bertmodel2_word_embed_embedding0_weight']._data[0][0,0].asscalar()) == 0.0116166482
    # Check the vocab
    unk_idx = vocab.token_to_idx[vocab.unknown_token]
    assert vocab.token_to_idx['test'] != unk_idx
    assert vocab.token_to_idx['Test'] != unk_idx
    assert vocab.token_to_idx['Test'] != vocab.token_to_idx['test']
    # Check the tokenizer
    assert tuple(tokenizer("The man jumped up, put his basket on Philammon's head")) == ('The', 'man', 'jumped', 'up', ',', 'put', 'his', 'basket', 'on', 'Phil', '##am', '##mon', "'", 's', 'head')

    # MXNet: bert-base-multi-cased

    model, vocab, tokenizer = get_pretrained([mx.cpu()], 'bert-base-multi-cased')
    # Check the model
    assert isinstance(model, nlp.model.BERTModel)
    assert len(model.encoder.transformer_cells) == 12
    assert pytest.approx(model.word_embed[0].params['bertmodel3_word_embed_embedding0_weight']._data[0][0,0].asscalar()) == 0.0518957935
    # Check the vocab
    unk_idx = vocab.token_to_idx[vocab.unknown_token]
    assert vocab.token_to_idx['Test'] != unk_idx
    assert vocab.token_to_idx['これは'] != unk_idx
    # Check the tokenizer
    assert tuple(tokenizer("これはTestですよ。")) == ('これは', '##T', '##est', '##で', '##す', '##よ', '。')
