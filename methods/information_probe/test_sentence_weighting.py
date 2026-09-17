import torch
from .scoring import channels
from .sentence_weighting import SentenceWeighting, eot_vectors, frozen_channels_and_values, random_conditioning


def fixture():
    torch.manual_seed(91)
    v, t = torch.randn(3, 5, 8), torch.randn(4, 6, 8)
    vm = torch.tensor([[1, 0, 0, 0, 1], [1, 0, 0, 1, 1], [1, 0, 0, 0, 0]])
    tm = torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 0, 0], [1]*6, [1, 1, 0, 0, 0, 0]])
    return v, t, vm, tm


def test_original_channels_and_zero_initialization():
    v, t, vm, tm = fixture()
    a, b, values = frozen_channels_and_values(v, t, vm, tm, 20.)
    aa, bb = channels(v, t, vm, tm, 20.)
    torch.testing.assert_close(a, aa, atol=0, rtol=0)
    torch.testing.assert_close(b, bb, atol=0, rtol=0)
    model = SentenceWeighting(8, 3)
    delta = model(v, vm, eot_vectors(t, tm), values, 20.)
    assert torch.equal(delta, torch.zeros_like(delta))


def test_eot_and_random_control_independent_queries():
    v, t, vm, tm = fixture()
    q = eot_vectors(t, tm)
    torch.testing.assert_close(q[0], torch.nn.functional.normalize(t[0, 2], dim=-1))
    whole, idx = random_conditioning(q, q)
    part, part_idx = random_conditioning(q[:1], q)
    assert torch.equal(part, whole[:1]) and part_idx == idx[:1]
    assert all(not torch.equal(q[i], whole[i]) for i in range(len(q)))


def test_masks_gradients_and_block_invariance():
    v, t, vm, tm = fixture()
    q = eot_vectors(t, tm)
    _, _, values = frozen_channels_and_values(v, t, vm, tm, 20.)
    model = SentenceWeighting(8, 3)
    model(v, vm, q, values, 20.).square().sum().backward()
    model.zero_grad()
    model(v, vm, q, values, 20.).sum().backward()
    assert model.query.weight.grad.norm() > 0
    with torch.no_grad():
        model.query.weight.add_(.01 * model.query.weight.grad)
    model.zero_grad()
    delta = model(v, vm, q, values, 20.)
    delta.sum().backward()
    assert model.key.weight.grad.norm() > 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())
    weights, _ = model.weights(v, vm, q)
    assert torch.equal(weights.masked_select((vm != 0)[:, None].expand_as(weights)), torch.zeros_like(weights.masked_select((vm != 0)[:, None].expand_as(weights))))
    torch.testing.assert_close(weights.sum(-1), torch.ones(3, 4))
    torch.testing.assert_close(delta[:1, :2], model(v[:1], vm[:1], q[:2], values[:1, :2], 20.))
    # Changing another query must not change this query's weights.
    changed = q.clone()
    changed[1:] *= -1
    torch.testing.assert_close(model.weights(v, vm, changed)[0][:, 0], weights[:, 0])
