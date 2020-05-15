import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# MXNet-based
import mxnet as mx
from mxnet.gluon import Block
import gluonnlp as nlp
from gluonnlp.model import get_model as _get_model
# PyTorch-based
import torch
import transformers

from .gpt2 import gpt2_117m, gpt2_345m
from .bert import BERTRegression, BertForMaskedLMOptimized

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

# Shortcodes for MXNet models
# These should not conflict w/ HuggingFace Transformer's shortcodes
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

    if name not in SUPPORTED:
        logging.warn("Model '{}' not recognized as an MXNet model; treating as PyTorch model".format(name))
        model_fullname = name

        if model_fullname.startswith('bert-'):

            if params_file is None:
                model, loading_info = BertForMaskedLMOptimized.from_pretrained(model_fullname, output_loading_info=True)
            else:
                model, loading_info = BertForMaskedLMOptimized.from_pretrained(params_file, output_loading_info=True)

            tokenizer = transformers.BertTokenizer.from_pretrained(model_fullname)
            vocab = None

        elif model_fullname.startswith('xlm-'):

            model, loading_info = transformers.XLMWithLMHeadModel.from_pretrained(model_fullname, output_loading_info=True)
            tokenizer = transformers.XLMTokenizer.from_pretrained(model_fullname)
            vocab = None

            # TODO: The loading code in `transformers` assumes pred_layer is under transformers, so the LM head is not loaded properly. We load manually:
            archive_file = transformers.XLMWithLMHeadModel.pretrained_model_archive_map[model_fullname]
            resolved_archive_file = transformers.file_utils.cached_path(archive_file)
            pretrained_state_dict = torch.load(resolved_archive_file, map_location='cpu')
            new_state_dict = model.state_dict()
            new_state_dict.update(
                {
                    'pred_layer.proj.weight': pretrained_state_dict['pred_layer.proj.weight'],
                    'pred_layer.proj.bias': pretrained_state_dict['pred_layer.proj.bias']
                }
            )
            model.load_state_dict(new_state_dict)

        else:
            raise ValueError("Model '{}' is not currently a supported PyTorch model".format(name))

    # Name format: model-size-lang-cased/uncased(-dataset / special characteristic)
    # e.g., 'bert-base-en-uncased-owt', 'gpt2-117m-en-cased'
    else:
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

        if model_name == 'bert'and size != 'base_bertpr':

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
