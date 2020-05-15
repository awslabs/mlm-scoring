import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mxnet as mx
from mxnet.gluon import Block
import gluonnlp as nlp
from gluonnlp.model import get_model as _get_model

from .gpt2 import gpt2_117m, gpt2_345m
from .bert import BERTRegression

# get_model() is from:
# https://github.com/dmlc/gluon-nlp/blob/master/scripts/text_generation/model/__init__.py
def get_model(name: str, **kwargs) -> Tuple[Block, nlp.Vocab]:
    """Returns a pre-defined model by name.

    In addition to the models in GluonNLP model API, this API supports getting GPT-2 models.

    Parameters
    ----------
    name : str
        Name of the model.
    dataset_name : str or None, default None
        The dataset name on which the pre-trained model is trained.
        For language model, options are 'wikitext-2'.
        For ELMo, Options are 'gbw' and '5bw'.
        'gbw' represents 1 Billion Word Language Model Benchmark
        http://www.statmt.org/lm-benchmark/;
        '5bw' represents a dataset of 5.5B tokens consisting of
        Wikipedia (1.9B) and all of the monolingual news crawl data from WMT 2008-2012 (3.6B).
        If specified, then the returned vocabulary is extracted from
        the training set of the dataset.
        If None, then vocab is required, for specifying embedding weight size, and is directly
        returned.
    vocab : gluonnlp.Vocab or None, default None
        Vocabulary object to be used with the language model.
        Required when dataset_name is not specified.
        None Vocabulary object is required with the ELMo model.
    pretrained : bool, default False
        Whether to load the pre-trained weights for model.
    ctx : Context, default CPU
        The context in which to load the pre-trained weights.
    root : str, default '$MXNET_HOME/models' with MXNET_HOME defaults to '~/.mxnet'
        Location for keeping the model parameters.

    Returns
    -------
    gluon.Block, gluonnlp.Vocab, (optional) gluonnlp.Vocab
    """
    models: Dict[str, Block] = {
        'gpt2_117m' : gpt2_117m,
        'gpt2_345m' : gpt2_345m
    }
    name = name.lower()
    if name not in models:
        return _get_model(name, **kwargs)
    return models[name](**kwargs)


SUPPORTED = [
    'bert-base-en-uncased',
    'bert-base-en-cased',
    'roberta-base-en-cased',
    'bert-large-en-uncased',
    'bert-large-en-cased',
    'roberta-large-en-cased',
    'bert-base-en-uncased-owt',
    'bert-base-multi-uncased',
    'bert-base-multi-cased',
    'gpt2-117m-en-cased',
    'gpt2-345m-en-cased'
]


def get_pretrained(ctxs: List[mx.Context], name: str = 'bert-base-en-uncased', params_file: Optional[Path] = None, cased: bool = False, finetune: bool = False, regression: bool = False, freeze: int = 0, root: Optional[Path] = None) -> Tuple[Block, nlp.Vocab, nlp.data.BERTTokenizer]:

    # Name format: model-size-lang-cased/uncased(-dataset / special characteristic)
    # e.g., 'bert-base-en-uncased-owt', 'gpt2-117m-en-cased'
    if name not in SUPPORTED:
        raise ValueError("Model '{}' not recognized".format(name))
    name_parts = name.split('-')
    model_name = name_parts[0]
    size = name_parts[1]
    lang = name_parts[2]
    if name_parts[3] == 'cased':
        cased = True
    elif name_parts[3] == 'uncased':
        cased = False
    dataset = name_parts[4] if len(name_parts) == 5 else None

    if freeze < 0:
        raise ValueError("# of initial layers to freeze must be non-negative")

    if params_file is not None and dataset is not None:
        logging.warning("Model parameters '{}' was provided, ignoring dataset suffix '{}'".format(params_file, dataset))

    if model_name == 'bert':

        if cased:
            dataset_suffix = '_cased'
        else:
            dataset_suffix = '_uncased'

        if size == 'base':
            model_fullname = 'bert_12_768_12'
        elif size == 'large':
            model_fullname = 'bert_24_1024_16'

        if lang == 'en':
            if dataset is None:
                dataset_prefix = 'book_corpus_wiki_en'
            elif dataset == 'owt':
                dataset_prefix = 'openwebtext_book_corpus_wiki_en'
        elif lang == 'multi':
            dataset_prefix = 'wiki_multilingual'

        # Get stock BERT with MLM outputs
        kwargs = {
            'dataset_name': dataset_prefix + dataset_suffix,
            'pretrained': True,
            'ctx': ctxs,
            'use_pooler': False,
            'use_decoder': False,
            'use_classifier': False
        }
        if finetune or regression:
            kwargs['use_pooler'] = True
        else:
            kwargs['use_decoder'] = True
        # Override GluonNLP's default location?
        if root is not None:
            kwargs['root'] = str(root)
        model, vocab = get_model(model_fullname, **kwargs)

        # Freeze initial layers if needed
        for i in range(freeze):
            model.encoder.transformer_cells[i].collect_params().setattr('grad_req', 'null')

        # Wrapper if appropriate
        if regression:
            # NOTE THIS:
            model = BERTRegression(model, dropout=0.1)
            model.regression.initialize(init=mx.init.Normal(1.0), ctx=ctxs)

        # MXNet warning message suggests this when softmaxing in float16
        # But float16 is buggy, so let's halve our inference speed for now :(
        # os.environ['MXNET_SAFE_ACCUMULATION'] = '1'
        # model.cast('float16')

        # Get tokenizer
        tokenizer = nlp.data.BERTTokenizer(vocab, lower=(not cased))

    elif model_name == 'roberta':

        if cased:
            dataset_suffix = '_cased'
        else:
            ValueError('Uncased not supported')

        if size == 'base':
            model_fullname = 'roberta_12_768_12'
        elif size == 'large':
            model_fullname = 'roberta_24_1024_16'

        if lang == 'en' and dataset is None:
            dataset_prefix = 'openwebtext_ccnews_stories_books'
        else:
            ValueError('Dataset not supported')

        # Get stock BERT with MLM outputs
        kwargs = {
            'dataset_name': dataset_prefix + dataset_suffix,
            'pretrained': True,
            'ctx': ctxs,
            'use_pooler': False,
            'use_decoder': False,
            'use_classifier': False
        }
        if finetune or regression:
            kwargs['use_pooler'] = True
        else:
            kwargs['use_decoder'] = True
        # Override GluonNLP's default location?
        if root is not None:
            kwargs['root'] = str(root)
        model, vocab = get_model(model_fullname, **kwargs)

        # Freeze initial layers if needed
        for i in range(freeze):
            model.encoder.transformer_cells[i].collect_params().setattr('grad_req', 'null')

        # Wrapper if appropriate
        if regression:
            ValueError("Not yet tested")
            # NOTE THIS:
            model = BERTRegression(model, dropout=0.1)
            model.regression.initialize(init=mx.init.Normal(1.0), ctx=ctxs)

        # Get tokenizer
        tokenizer = nlp.data.GPT2BPETokenizer()

        # TODO: Have the scorers condition on what the vocab and tokenizer class are
        vocab.cls_token = vocab.bos_token
        vocab.sep_token = vocab.eos_token
        tokenizer.convert_tokens_to_ids = vocab.to_indices

    elif model_name == 'gpt2':

        assert cased
        assert not finetune
        assert not regression
        assert freeze == 0

        if size == '117m':
            model_fullname = 'gpt2_117m'
        elif size == '345m':
            model_fullname = 'gpt2_345m'

        # Get stock GPT-2
        kwargs = {
            'dataset_name': 'openai_webtext',
            'pretrained': True,
            'ctx': ctxs,
        }
        # Override GluonNLP's default location?
        if root is not None:
            kwargs['root'] = str(root)

        model, vocab = get_model(model_fullname, **kwargs)

        # Get tokenizer
        tokenizer = nlp.data.GPT2BPETokenizer()
        # To fit the assumptions of score block
        tokenizer.vocab = vocab
        vocab.cls_token = vocab.eos_token
        vocab.sep_token = vocab.eos_token
        tokenizer.convert_tokens_to_ids = vocab.to_indices


    if params_file is not None:
        model.load_parameters(str(params_file),
                    ctx=ctxs, allow_missing=True, ignore_extra=True, cast_dtype=True)

    return model, vocab, tokenizer
