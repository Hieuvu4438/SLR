import numpy as np

from methods.information_probe.cohort_dual_audit import assignment, dual_certificate, noisy_assignments


def test_dual_supports_random_and_tied_assignments_without_labels():
    for scores in (np.random.default_rng(42).normal(size=(8, 8)), np.ones((4, 4))):
        report, u, v, p = dual_certificate(scores)
        assert report['row_assigned_nonstrict_max_fraction'] == 1
        assert report['column_assigned_nonstrict_max_fraction'] == 1
        assert abs(report['duality_gap']) < 1e-7
        np.testing.assert_allclose((scores-u[:, None]-v)[np.arange(len(scores)), p], 0, atol=1e-7)


def test_focal_query_unchanged_but_cohort_assignment_reverses():
    a = np.array([[2., 1.], [100., 0.]])
    b = np.array([[2., 1.], [-100., 0.]])
    assert np.array_equal(a[0], b[0])
    assert assignment(a)[0] == 1 and assignment(b)[0] == 0


def test_jitter_reproducible_and_large_margin_assignment_stable():
    s = np.eye(4)*10
    a = noisy_assignments(s, seeds=range(3))
    assert a == noisy_assignments(s, seeds=range(3))
    assert all(x['changed_assignment_rows'] == 0 for x in a)
    assert all(x['promoted_full_gallery_R1_both_directions'] == 100 for x in a)
