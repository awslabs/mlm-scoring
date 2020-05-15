import pytest

import random

import mxnet as mx
import numpy as np

from mlm.cmds import setup_ctxs


def test_setup_ctxs():

    # CPU
    ctxs = setup_ctxs('-1')
    assert len(ctxs) == 1
    assert ctxs[0] == mx.cpu()
    # Test randomness
    assert random.randint(0, 1000000) == 885440
    assert np.random.randint(0, 1000000) == 985772
    assert mx.random.randint(0, 1000000, ctx=ctxs[0])[0] == 656751

    # GPU
    ctxs = setup_ctxs('0,2')
    assert len(ctxs) == 2
    assert ctxs[0] == mx.gpu(0)
    assert ctxs[1] == mx.gpu(2)
    # Test randomness
    for ctx in ctxs:
        assert mx.random.randint(0, 1000000, ctx=ctx)[0] == 248005
