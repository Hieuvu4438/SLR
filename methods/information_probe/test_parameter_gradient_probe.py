import torch

from methods.information_probe.parameter_gradient_probe import partition_summary, cosine


def test_chunked_chain_rule_and_pathway_partition():
    torch.manual_seed(42)
    weight = torch.randn(3, 4, dtype=torch.double, requires_grad=True)
    x = torch.randn(6, 3, dtype=torch.double)
    y = x@weight
    loss = y.square().mean()+y.mean().square()
    direct, = torch.autograd.grad(loss, weight, retain_graph=True)
    covector, = torch.autograd.grad(loss, y)
    accumulated = torch.zeros_like(weight)
    for start in range(0, 6, 2):
        out = x[start:start+2]@weight
        g, = torch.autograd.grad(out, weight, grad_outputs=covector[start:start+2])
        accumulated += g
    torch.testing.assert_close(accumulated, direct)
    summary = partition_summary(torch.tensor([1., 1.]), torch.tensor([2., 0.]))
    assert summary['cross_term'] == -4
    assert summary['part_signed_projection_onto_total'] == 1
    assert abs(cosine(direct.flatten(), accumulated.flatten())-1) < 1e-12
