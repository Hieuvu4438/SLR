from methods.information_probe.cico_pseudoclip_audit import enumerate_counts, interval_counts, union


def test_short_and_local_materialization_have_no_gap():
    for starts in ([0], [0, 3], [52, 55, 58]):
        a = enumerate_counts(starts, min(starts), max(starts) + 15)
        assert a == interval_counts(starts, min(starts), max(starts) + 15)
        assert a["gap_frame_draws"] == a["any_gap"] == a["all_gap"] == 0
    assert enumerate_counts([0], 0, 15)["starts"] == 1


def test_nonlocal_gap_has_hand_calculated_exposure():
    expected = dict(starts=32, any_gap=31, all_gap=1, gap_frame_draws=256, gap_frames=16)
    assert enumerate_counts([0, 32], 0, 47) == expected
    assert interval_counts([32, 0], 0, 47) == expected


def test_interval_union_and_multiple_gaps_agree_with_frame_enumeration():
    assert union([(5, 9), (0, 6), (12, 12), (10, 11)]) == [(0, 9), (10, 11)]
    for starts in ([0, 30], [0, 30, 90], [0, 2, 31, 32, 75], [6, 11, 55, 160]):
        assert enumerate_counts(starts, min(starts), max(starts) + 15) == interval_counts(
            starts, min(starts), max(starts) + 15)
