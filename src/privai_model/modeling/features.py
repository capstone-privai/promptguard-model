"""Versioned, data-independent character statistics. No secrets are logged."""
from collections import Counter
from dataclasses import dataclass
import math

DEFAULT_PREFIXES = ('sk-', 'ghp_', 'github_pat_', 'AKIA')
BASE_NAMES = ('log_length', 'entropy_normalized', 'uppercase_ratio', 'lowercase_ratio',
              'digit_ratio', 'special_ratio', 'whitespace_ratio', 'uncased_letter_ratio',
              'class_diversity', 'known_prefix')


@dataclass(frozen=True)
class FeatureSpec:
    prefixes: tuple = DEFAULT_PREFIXES
    length_cap: int = 65536
    version: str = 'character-v1'

    def __post_init__(self):
        if self.version != 'character-v1' or type(self.length_cap) is not int or self.length_cap < 1:
            raise ValueError('invalid character feature specification')
        if any(not isinstance(x, str) or not x for x in self.prefixes) or len(set(self.prefixes)) != len(self.prefixes):
            raise ValueError('prefixes must be unique nonempty strings')

    @property
    def names(self):
        return list(BASE_NAMES) + ['prefix:'+p for p in self.prefixes]

    @property
    def dimension(self):
        return len(self.names)

    def to_dict(self):
        return {'version': self.version, 'prefixes': list(self.prefixes), 'length_cap': self.length_cap,
                'dimension': self.dimension, 'names': self.names,
                'normalization': 'fixed log-length cap; entropy/log2(0x110000); ratios; diversity/6'}

    @classmethod
    def from_dict(cls, data):
        spec = cls(tuple(data['prefixes']), data['length_cap'], data['version'])
        if data.get('dimension', spec.dimension) != spec.dimension or data.get('names', spec.names) != spec.names:
            raise ValueError('feature specification dimension/order mismatch')
        return spec


def feature_spec(config):
    return FeatureSpec(tuple(config.get('feature_prefixes', DEFAULT_PREFIXES)), config.get('feature_length_cap', 65536))


def validate_feature_options(config):
    if type(config.get('character_features', False)) is not bool:
        raise ValueError('character_features must be boolean')
    feature_spec(config)


def character_statistics(value, prefixes=DEFAULT_PREFIXES):
    if not isinstance(value, str):
        raise ValueError('candidate value must be text')
    n = len(value)
    counts = Counter(value)
    entropy = -sum((c/n)*math.log2(c/n) for c in counts.values()) if n else 0.0
    classes = [0]*6  # upper, lower, digit, special, whitespace, uncased Unicode letter
    for char, count in counts.items():
        index = (0 if char.isupper() else 1 if char.islower() else 2 if char.isdigit()
                 else 4 if char.isspace() else 5 if char.isalpha() else 3)
        classes[index] += count
    ratios = [c/n if n else 0.0 for c in classes]
    matches = [float(value.startswith(prefix)) for prefix in prefixes]
    return {'length': n, 'entropy': entropy, 'uppercase_ratio': ratios[0], 'lowercase_ratio': ratios[1],
            'digit_ratio': ratios[2], 'special_ratio': ratios[3], 'whitespace_ratio': ratios[4],
            'uncased_letter_ratio': ratios[5], 'class_diversity': sum(c > 0 for c in classes),
            'known_prefix': float(any(matches)), 'prefix_matches': matches}


def extract_features(value, spec=None):
    spec = spec or FeatureSpec()
    stats = character_statistics(value, spec.prefixes)
    return [math.log1p(min(stats['length'], spec.length_cap))/math.log1p(spec.length_cap),
            stats['entropy']/math.log2(0x110000),
            *(stats[k] for k in BASE_NAMES[2:8]), stats['class_diversity']/6,
            stats['known_prefix'], *stats['prefix_matches']]
