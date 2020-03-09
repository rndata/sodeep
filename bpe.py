"""
Neural Machine Translation of Rare Words with Subword Units
Rico Sennrich and Barry Haddow and Alexandra Birch
https://arxiv.org/abs/1508.07909
"""
import itertools as it
import re
from collections import defaultdict

import toolz as tz

import .corpus


def prepare_vocab(tokens_it):
    vocab = defaultdict(int)
    for tokens in tokens_it:
        for token in tokens:
            if len(token) > 1:
                vocab[" ".join(token)] += 1

    return vocab


def get_stats(vocab):
    pairs= defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for s1, s2 in tz.sliding_window(2, symbols):
            pairs[s1, s2] += freq
    return pairs


def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out


def encode_vocab(vocab, merges=10):
    vocab = dict(vocab)
    for i in range(merges):
        pairs = get_stats(vocab)
        best = max(pairs, key=pairs.get)
        vocab = merge_vocab(best, vocab)

    return vocab

def build_vocab(tokens_it, merges=10):
    vocab = prepare_vocab(tokens_it)
    vocab = encode_vocab(vocab)
    return vocab


def to_tokens(vocab):
    r = set()
    for k in vocab:
        for v in k.split(" "):
            if len(v) > 1:
                r.add(v)
    return sorted(r, key=len, reverse=True)
