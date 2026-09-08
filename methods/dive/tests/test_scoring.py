from __future__ import annotations

import torch
import torch.nn.functional as F

from dive.models.scoring import compose_score, evidence_score_block, evidence_score_chunked


def _fixture(dtype=torch.float32):
    generator = torch.Generator().manual_seed(17)
    u = F.normalize(torch.randn(3, 4, 5, generator=generator, dtype=dtype), dim=-1)
    e = F.normalize(torch.randn(2, 3, 5, generator=generator, dtype=dtype), dim=-1)
    vm = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0], [1, 1, 1, 1]], dtype=torch.bool)
    tm = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool)
    return u, e, vm, tm


def test_padding_invariance_and_variable_block_sizes():
    u, e, vm, tm = _fixture()
    dense, valid = evidence_score_block(u, e, vm, tm)
    padded_u = torch.cat((u, torch.randn(3, 2, 5)), dim=1)
    padded_e = torch.cat((e, torch.randn(2, 4, 5)), dim=1)
    padded_vm = torch.cat((vm, torch.zeros(3, 2, dtype=torch.bool)), dim=1)
    padded_tm = torch.cat((tm, torch.zeros(2, 4, dtype=torch.bool)), dim=1)
    padded, padded_valid = evidence_score_block(padded_u, padded_e, padded_vm, padded_tm)
    chunked, chunked_valid = evidence_score_chunked(
        u, e, vm, tm, video_chunk_size=2, text_chunk_size=1
    )
    torch.testing.assert_close(padded, dense)
    torch.testing.assert_close(chunked, dense)
    assert torch.equal(valid, padded_valid) and torch.equal(valid, chunked_valid)


def test_all_invalid_evidence_uses_exact_baseline_fallback():
    u, e, vm, tm = _fixture()
    vm[1] = False
    e_score, pair_valid = evidence_score_block(u, e, vm, tm)
    assert torch.equal(e_score[1], torch.zeros_like(e_score[1]))
    baseline = torch.randn_like(e_score)
    student = e_score + 0.2
    composed = compose_score(baseline, student, e_score, pair_valid, gamma=0.1)
    torch.testing.assert_close(composed[1], baseline[1])


def test_scores_and_centered_correction_obey_cosine_bounds():
    u, e, vm, tm = _fixture()
    reference, valid = evidence_score_block(u, e, vm, tm)
    student_u = F.normalize(u + 0.1, dim=-1)
    student, _ = evidence_score_block(student_u, e, vm, tm)
    assert bool((reference.abs() <= 1 + 1e-6).all())
    correction = torch.where(valid, 0.5 * (student - reference), 0)
    assert bool((correction.abs() <= 1 + 1e-6).all())
    baseline = torch.randn_like(reference)
    torch.testing.assert_close(compose_score(baseline, student, reference, valid, 0), baseline)


def test_chunked_autograd_matches_dense():
    raw_u = torch.randn(3, 4, 5, dtype=torch.float64, generator=torch.Generator().manual_seed(4), requires_grad=True)
    raw_e = torch.randn(2, 3, 5, dtype=torch.float64, generator=torch.Generator().manual_seed(5), requires_grad=True)
    u = F.normalize(raw_u, dim=-1)
    e = F.normalize(raw_e, dim=-1)
    vm = torch.ones(3, 4, dtype=torch.bool)
    tm = torch.ones(2, 3, dtype=torch.bool)
    dense, _ = evidence_score_block(u, e, vm, tm)
    dense.sum().backward(retain_graph=True)
    dense_u_grad = raw_u.grad.detach().clone()
    dense_e_grad = raw_e.grad.detach().clone()
    raw_u.grad = None
    raw_e.grad = None
    chunked, _ = evidence_score_chunked(
        u, e, vm, tm, video_chunk_size=2, text_chunk_size=1
    )
    chunked.sum().backward()
    torch.testing.assert_close(chunked, dense)
    torch.testing.assert_close(raw_u.grad, dense_u_grad)
    torch.testing.assert_close(raw_e.grad, dense_e_grad)


def test_fp64_gradcheck_for_scorer():
    raw_u = torch.randn(1, 2, 3, dtype=torch.float64, requires_grad=True)
    raw_e = torch.randn(1, 2, 3, dtype=torch.float64, requires_grad=True)
    vm = torch.ones(1, 2, dtype=torch.bool)
    tm = torch.ones(1, 2, dtype=torch.bool)

    def score(a, b):
        return evidence_score_block(
            F.normalize(a, dim=-1), F.normalize(b, dim=-1), vm, tm, validate=False
        )[0]

    assert torch.autograd.gradcheck(score, (raw_u, raw_e), eps=1e-6, atol=1e-5, rtol=1e-4)
