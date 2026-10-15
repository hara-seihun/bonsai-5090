"""Frozen signal-blind schedule contract shared by the panel and CPU proof."""
from dataclasses import dataclass

COUNTS = (394, 338, 232, 239, 1640)
ARRAY_LENGTH = sum(COUNTS)
IDENTITY = 'native7-g1-cap4-histogram-v1'
MASK = (1 << 64) - 1


@dataclass
class SplitMix64:
    state: int

    def next(self):
        self.state = (self.state + 0x9e3779b97f4a7c15) & MASK
        z = self.state
        z = ((z ^ (z >> 30)) * 0xbf58476d1ce4e5b9) & MASK
        z = ((z ^ (z >> 27)) * 0x94d049bb133111eb) & MASK
        return z ^ (z >> 31)

    def bounded(self, bound):
        threshold = ((-bound) & MASK) % bound
        while True:
            value = self.next()
            if value >= threshold:
                return value % bound


def descriptor(seed):
    assert type(seed) is int and 0 <= seed <= 0x7fffffff
    sequence = bytearray(value for value, count in enumerate(COUNTS) for _ in range(count))
    rng = SplitMix64(seed)
    for i in range(ARRAY_LENGTH - 1, 0, -1):
        j = rng.bounded(i + 1)
        sequence[i], sequence[j] = sequence[j], sequence[i]
    return {'id': IDENTITY, 'seed': seed, 'counts': list(COUNTS), 'array_length': ARRAY_LENGTH,
            'algorithm': 'splitmix64-fisher-yates-rejection-v1', 'sequence_hex': sequence.hex(),
            'request_offset': 'fnv1a64-token-le32-xor-seed-splitmix64-mod-v1',
            'consumption': 'prompt-rotated-cyclic-one-per-verifier-round'}


def request_offset(token_ids, seed):
    value = 14695981039346656037
    for token in token_ids:
        assert type(token) is int and 0 <= token <= 0xffffffff
        for byte in token.to_bytes(4, 'little'):
            value = ((value ^ byte) * 1099511628211) & MASK
    return SplitMix64(value ^ seed).next() % ARRAY_LENGTH
