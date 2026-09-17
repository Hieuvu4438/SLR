import torch
from .augmentation_training_probe import batch_record, tensor_digest
from .validate_augmentation_training import matched_batches


def test_digest_and_batch_trace_do_not_consume_rng():
    state = torch.get_rng_state().clone()
    x = torch.tensor([[1, 2], [3, 4]])
    b = {'pair_id': ['a', 'b'], 'h': x.float(), 'valid': torch.ones(2, dtype=torch.bool),
         'clean_text': (x, x, x), 'aug_text': (x.clone(), x, x)}
    r = batch_record(b, 0)
    assert r['clean_sha256'] == r['aug_sha256'] and r['changed_text_row_n'] == 0
    b['aug_text'][0][1, 0] = 7
    r = batch_record(b, 0)
    assert r['changed_text_row_n'] == 1 and r['clean_sha256'] != r['aug_sha256']
    assert torch.equal(state, torch.get_rng_state())
    assert tensor_digest([('x', torch.tensor(1.))]) != tensor_digest([('x', torch.tensor(2.))])
    assert tensor_digest([('x', x)]) != tensor_digest([('x', x.float())])


def test_matched_batches_ignore_only_augmented_inputs():
    a = {'batch_number': 0, 'pair_ids': ['a', 'b'], 'clean_sha256': 'c', 'visual_sha256': 'v', 'aug_sha256': 'x'}
    assert matched_batches([a], [{**a, 'aug_sha256': 'y'}])
    assert not matched_batches([a], [{**a, 'pair_ids': ['b', 'a']}])
    assert not matched_batches([a], [{**a, 'visual_sha256': 'different'}])
    assert not matched_batches([a], [])
