import ast

import pytest

from .c2rl_reimplementation_padding import CPUOnly, neutral_pool


def test_neutral_pool_no_padding_and_declared_order_change():
    for value in (-.3, 0., .2):
        assert neutral_pool(value, 2, 2) == pytest.approx(value)
    assert neutral_pool(.21, 1, 2) > neutral_pool(.19, 2, 2)
    assert neutral_pool(.21, 1, 32) < neutral_pool(.19, 2, 32)


def test_cpu_adapter_changes_only_empty_cuda_calls():
    tree = ast.parse('result = x.cuda() + torch.softmax(y, dim=-1).cuda()')
    adapter = CPUOnly()
    actual = adapter.visit(tree)
    expected = ast.parse('result = x + torch.softmax(y, dim=-1)')
    assert ast.dump(actual) == ast.dump(expected)
    assert adapter.replaced == 2
    with pytest.raises(AssertionError):
        CPUOnly().visit(ast.parse('x.cuda(device=1)'))
