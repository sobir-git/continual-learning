from types import SimpleNamespace
import torch

from exp2.controller import GrowingLinear, GrowingController


def test_growing_linear():
    bsize = 8
    gl = GrowingLinear(0, 0, bias=True)
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


def test_growing_linear_no_bias():
    bsize = 8
    gl = GrowingLinear(0, 0, bias=False)

    gl.append(3, 2)
    inputs = torch.randn((bsize, 3))
    outputs = gl(inputs)
    assert outputs.shape == (bsize, 2)
    assert len(list(gl.parameters())) == 4
    assert gl.b00 is None
    assert gl.b01 is None
    assert gl.b11 is None
    assert gl.b10 is None


def test_growing_controller():
    config = SimpleNamespace(lr=0.01, ctrl_bias=True)
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

    def has_grad(i, j):
        wfrozen = getattr(gc.linear, f'w{i}{j}').grad is not None
        bfrozen = getattr(gc.linear, f'b{i}{j}').grad is not None
        return wfrozen and bfrozen

    assert gc.bn.weight.grad is None
    assert gc.bn.bias.grad is None
    assert not has_grad(0, 0)
    assert not has_grad(0, 1)
    assert not has_grad(1, 1)
    assert not has_grad(1, 0)

    gc.set_warmup(True)
    assert not gc.linear.w00.requires_grad
    outputs = gc(clf_inputs)
    loss = criterion(outputs, target)
    loss.backward()

    assert gc.bn.weight.grad is not None
    assert gc.bn.bias.grad is not None
    assert not has_grad(0, 0)  # still no grad
    assert has_grad(0, 1)
    assert has_grad(1, 1)
    assert has_grad(1, 0)
