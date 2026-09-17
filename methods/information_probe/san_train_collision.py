"""TRAIN-only string reachability census, without SAN/EDA/model execution."""
from collections import Counter, defaultdict
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import pickle
import pickletools


class PrimitiveUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise ValueError('Class resolution forbidden')

    def persistent_load(self, pid):
        raise ValueError('Persistent objects forbidden')


def one_swap(a, b):
    if len(a) != len(b):
        return False
    diff = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
    return not diff or (len(diff) == 2 and
                        a[diff[0]] == b[diff[1]] and a[diff[1]] == b[diff[0]])


def main():
    root = Path(__file__).resolve().parents[2]
    source = root / 'third_party/SAN/data/labels.train.cleaned'
    compressed = source.read_bytes()
    raw = gzip.decompress(compressed)
    ops = Counter(op.name for op, _, _ in pickletools.genops(raw))
    allowed = {'PROTO', 'FRAME', 'EMPTY_DICT', 'MEMOIZE', 'MARK', 'SHORT_BINUNICODE',
               'BININT1', 'SETITEMS', 'BINGET', 'BININT2', 'BINUNICODE', 'STOP'}
    assert set(ops) <= allowed, ops
    data = PrimitiveUnpickler(io.BytesIO(raw)).load()
    assert isinstance(data, dict) and len(data) == 7096
    texts = [v['text'] for v in data.values()]
    assert all(isinstance(s, str) and s.split() for s in texts)

    # Independent brute-force toy check, including repeated words and no-op.
    from itertools import product
    tested = 0
    for length in (1, 2, 3, 4):
        seqs = list(product(('a', 'b'), repeat=length))
        for a in seqs:
            reached = {a}
            for i in range(length):
                for j in range(length):
                    value = list(a)
                    value[i], value[j] = value[j], value[i]
                    reached.add(tuple(value))
            for b in seqs:
                assert one_swap(a, b) == (b in reached)
                tested += 1

    seq = [tuple(s.split()) for s in texts]
    groups = defaultdict(list)
    for i, tokens in enumerate(seq):
        groups[tuple(sorted(tokens))].append(i)
    records = {}
    for mode in ('arbitrary_permutation', 'one_transposition'):
        k = Counter()
        active_groups = 0
        for ids in groups.values():
            found = False
            for i in ids:
                for j in ids:
                    if texts[i] == texts[j] or texts[j] != ' '.join(seq[j]):
                        continue
                    if mode == 'one_transposition' and not one_swap(seq[i], seq[j]):
                        continue
                    k[i] += 1
                    found = True
            active_groups += int(found)
        def cooccurrence(count):
            return -math.expm1(sum(math.log1p(-count / (len(texts)-1-j))
                                  for j in range(63)))
        bound = .5 * sum(cooccurrence(count) for count in k.values()) / len(texts)
        assert 0 <= bound <= .5 * len(k) / len(texts)
        records[mode] = {'eligible_source_rows': len(k),
                         'eligible_source_percent': 100 * len(k) / len(texts),
                         'ordered_row_pairs': sum(k.values()), 'groups': active_groups,
                         'uniform_B64_augmented_cooccurrence_upper_probability': bound,
                         'uniform_B64_augmented_cooccurrence_upper_percent': 100 * bound,
                         'counterparts_by_source_row_index': dict(sorted(k.items()))}
    maximum = max(Counter(texts).values())
    protocol = root / 'docs/proposal7/evidence/autonomous_search/SAN_train_collision_protocol.md'
    files = [source, Path(__file__), protocol, root / 'third_party/SAN/datasets.py',
             root / 'third_party/SAN/train_vlp_v2.py', root / 'third_party/SAN/train.bash']
    print(json.dumps({'status': 'completed', 'scope': 'SAN TRAIN raw-string reachability only',
                      'n_train': len(texts), 'toy_pairs_verified': tested,
                      'pickle_op_counts': dict(ops),
                      'max_identical_string_multiplicity': maximum,
                      'all_identical_B64_possible_without_replacement': maximum >= 64,
                      'whitespace_normalization_changed_rows': sum(s != ' '.join(t) for s,t in zip(texts,seq)),
                      'records': records,
                      'sha256': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in files},
                      'test_or_dev_labels_accessed': False, 'model_or_tokenizer_executed': False,
                      'optimizer_updates': 0, 'method_go': False}, indent=2))


if __name__ == '__main__':
    main()
