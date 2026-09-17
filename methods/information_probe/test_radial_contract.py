import pytest
import torch

from .radial_contract import fixed_controls, inverse_coefficients


def test_fixed_exact_and_unsupported_controls():
    result = fixed_controls()
    assert result['max_relative_error'] < 1e-10
    assert result['unsupported_cases_refused'] == ['zero_bias', 'rectangular']


def test_zero_gain_is_not_inverted():
    with pytest.raises(ValueError, match='nonzero'):
        inverse_coefficients(torch.eye(3), torch.tensor([1., 0., 1.]), torch.ones(3))


def test_affine_constraint_uses_row_vector_projection_orientation():
    p = torch.tensor([[2., 1., 0.], [0., 3., 1.], [1., 0., 4.]], dtype=torch.float64)
    g = torch.tensor([.5, 1., 2.], dtype=torch.float64)
    beta = torch.tensor([.1, .2, .3], dtype=torch.float64)
    u = torch.tensor([[1., 2., -3.], [4., -1., -3.]], dtype=torch.float64)
    z = (g * u + beta) @ p
    a, b = inverse_coefficients(p, g, beta)
    torch.testing.assert_close(z @ a, b.expand(2), atol=1e-12, rtol=1e-12)
    radius = z.norm(dim=-1)
    torch.testing.assert_close(b / ((z / radius[:, None]) @ a), radius,
                               atol=1e-12, rtol=1e-12)
