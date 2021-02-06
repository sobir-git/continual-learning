from types import SimpleNamespace
import torch

from exp2.controller import GrowingLinear, GrowingController


def test_growing_linear():
    bsize = 8
    gl = GrowingLinear(0, 0)
    gl.append(3, 2)
    inputs = torch.randn((bsize, 3))
    outputs = gl(inputs)
    assert outputs.shape == (bsize, 2)
    assert len(list(gl.parameters())) == 8

    gl.append(8, 20)
    inputs = torch.randn((bsize, 11))
    outputs = gl(inputs)
    assert outputs.shape == (bsize, 22)
    assert len(list(gl.parameters())) == 8


def test_growing_controller():
    config = SimpleNamespace(lr=0.01)
    gc = GrowingController(config)
    before_n_params = len(list(gc.parameters()))
    assert before_n_params == 10

    gc.extend(list(range(10)), 11)
    assert gc.bn.num_features == 11
    assert len(list(gc.parameters())) == before_n_params

    gc.extend(list(range(10, 16)), 7)
    assert gc.bn.num_features == 18
    assert len(list(gc.parameters())) == before_n_params

    bsize = 8
    clf_inputs = [torch.randn(bsize, 11), torch.randn(bsize, 7)]
    outputs = gc(clf_inputs)
    assert outputs.shape == (bsize, 16)

    target = torch.randint(high=15, size=(bsize,))
    criterion = torch.nn.CrossEntropyLoss()

    assert gc.bn.weight.grad is None
    assert gc.bn.bias.grad is None
    assert gc.linear.l00.weight.grad is None
    assert gc.linear.l01.weight.grad is None
    assert gc.linear.l11.weight.grad is None
    assert gc.linear.l10.weight.grad is None

    gc.set_warmup(True)
    assert not gc.linear.l00.weight.requires_grad
    outputs = gc(clf_inputs)
    loss = criterion(outputs, target)
    loss.backward()

    assert gc.bn.weight.grad is not None
    assert gc.bn.bias.grad is not None
    assert gc.linear.l00.weight.grad is None  # still None because of warmup
    assert gc.linear.l01.weight.grad is not None
    assert gc.linear.l11.weight.grad is not None
    assert gc.linear.l10.weight.grad is not None
