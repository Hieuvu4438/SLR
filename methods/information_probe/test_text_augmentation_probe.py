from collections import Counter

from methods.information_probe.text_augmentation_probe import captions
from slr_common.data.tokenize import augment_caption


def test_caption_control_and_deployed_recipe():
    records = [{'caption_model': 'Tomorrow rain arrives from the west', 'pair_id': f'id-{i}'} for i in range(30)]
    assert captions(records, None) == [r['caption_model'] for r in records]
    for seed in (42, 1337, 2026):
        result = captions(records, seed)
        assert result == captions(records, seed)
        assert result == [augment_caption(r['caption_model'], r['pair_id'], seed=seed, epoch=0) for r in records]
        assert all(Counter(t.split()) == Counter(records[i]['caption_model'].split()) for i, t in enumerate(result))


def test_augmentation_does_not_change_empty_or_single_word():
    records = [{'caption_model': text, 'pair_id': str(i)} for i, text in enumerate(['', 'Rain'])]
    assert captions(records, 42) == ['', 'Rain']
